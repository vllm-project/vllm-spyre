import json
import sys
from pathlib import Path

# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get('triton'):
        del sys.modules['triton']

import math
import operator
import os
from typing import TYPE_CHECKING, Union

import torch
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None
from vllm.platforms import Platform, PlatformEnum

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)

THREADING_ENVS = [
    "OMP_NUM_THREADS",
    # "TORCHINDUCTOR_COMPILE_THREADS", # vLLM wants this set to 1
    "DT_PARALLEL_THREADS",  # affects the compilation during warmup
    # set these for good measure
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
]

PRE_COMPILE_MODEL_CONFIG_FILENAME = "model_compile.log.json"
PRE_COMPILE_MODEL_CATALOG_FILENAME = "pre_compiled_cache_catalog.json"
DISABLE_COMPILATION_ENV_VAR = "DISABLE_COMPILATION"


# Needed by vllm/model_executor/layers/pooler.py:562
# Copied from vllm/utils/__init__.py
class _StreamPlaceholder:

    def __init__(self):
        self.synchronize = lambda: None


class SpyrePlatform(Platform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"
    # compressed-tensors supported by
    # https://github.com/foundation-model-stack/fms-model-optimizer/blob/main/fms_mo/aiu_addons/__init__.py
    supported_quantization: list[str] = ["gptq", "compressed-tensors"]
    _warmup_shapes: tuple[dict[str, int], ...] | None = None
    _block_size: int = 64  # hardcoded Spyre constraint for now
    _num_spyre_blocks_override: int = -1  # override num of KV cache blocks
    _config: VllmConfig = None

    # TODO: see if this needs to be set
    # See vllm batched_count_greater_than method
    # simple_compile_backend: str = "eager"

    # Needed by vllm/model_executor/layers/pooler.py:562
    current_stream = lambda _: _StreamPlaceholder()

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "spyre"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        """
        Check if the current platform supports async output.
        """
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:

        # In case vllm passes a default vllm_config to us.
        # This happens when get_current_vllm_config is called
        # without setting the vllm config through
        # set_current_vllm_config
        if vllm_config.model_config is None:
            return

        cls._config = vllm_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        is_decoder = model_config.runner_type == "generate"

        is_pooling = model_config.runner_type == "pooling"

        if not bool(int(os.getenv("VLLM_USE_V1", "1"))):
            raise ValueError("vllm-spyre is only supported with vLLM v1. "
                             "Please set VLLM_USE_V1=1")
        elif not is_decoder and not is_pooling:
            raise ValueError("Only the 'generate' and 'pooling' runners are "
                             "supported")

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_spyre.v1.worker."\
                "spyre_worker.SpyreWorker"

        cls._check_threading_config(parallel_config.world_size)

        # set env vars based on the model
        if is_decoder:
            os.environ["FLEX_OVERWRITE_NMB_FRAME"] = "true"
            os.environ["COMPILATION_MODE"] = "offline_decoder"
        if is_pooling:
            os.environ["FLEX_OVERWRITE_NMB_FRAME"] = "false"
            os.environ["COMPILATION_MODE"] = "offline"

        if envs_spyre.VLLM_SPYRE_USE_CB and is_decoder:
            scheduler_config.scheduler_cls = "vllm_spyre.v1.core."\
                "scheduler.ContinuousBatchingSpyreScheduler"
            if envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS:
                raise ValueError("Prompt logprobs not supported with " \
                "continuous batching")
            if (vllm_config.model_config.quantization
                    and vllm_config.scheduler_config.max_num_seqs == 1):
                raise ValueError(
                    "Batch size 1 not supported for fp8 continuous batching.")
        else:
            # Static batching or embedding model.
            # Override --max-num-seqs to the biggest warmup batch size
            # And override --max-model-len to the biggest warmup sequence
            cls._warmup_shapes = None
            spyre_warmup_shapes = cls.get_warmup_shapes(scheduler_config)
            max_batch_size = 0
            max_seq_len = 0
            for shape in spyre_warmup_shapes:
                max_batch_size = max(max_batch_size, shape["batch_size"])
                max_seq_len = max(max_seq_len,
                                  shape["prompt_length"] + shape["new_tokens"])

            if (envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS
                    and max_batch_size > 1):
                raise ValueError(
                    "Prompt logprobs only supported with batch size 1")

            # verify that warmup shapes are not too large
            model_config.get_and_verify_max_len(max_model_len=max_seq_len)

            # override stuff
            model_config.max_model_len = max_seq_len
            scheduler_config.max_num_seqs = max_batch_size

            scheduler_config.scheduler_cls = (
                    "vllm_spyre.v1.core.scheduler."\
                        "StaticBatchingSpyreScheduler")

        # To disable any paged attention ops in the base scheduler, we:
        # - Set the block size (in tokens) to the maximum sequence length
        #       so that the scheduler thinks an entire sequence will fit in
        #       one single block.
        # - Set the number of blocks to the maximum number of sequences, so
        #       the scheduler always thinks there's a block available
        # - Set `max_num_batched_tokens` to the size of a full batch of full
        #       length requests, so that the scheduler will always have token
        #       budget available to schedule a full batch
        if cache_config is not None:
            # overriding number of available Spyre blocks if not None
            if cache_config.num_gpu_blocks_override:
                cls._num_spyre_blocks_override = \
                    cache_config.num_gpu_blocks_override
            # The V1 scheduler actually needs 2 blocks for each sequence...
            cache_config.num_gpu_blocks_override = \
                scheduler_config.max_num_seqs * 2

            cache_config.block_size = model_config.max_model_len
            scheduler_config.max_num_batched_tokens = (
                model_config.max_model_len * scheduler_config.max_num_seqs)

        logger.info(
            "Overriding configurations based on warmup shapes. "
            "max_model_len=%d, max_num_seqs=%d, block_size=%d, "
            "num_gpu_blocks_override=%d, max_num_batched_tokens=%d",
            model_config.max_model_len, scheduler_config.max_num_seqs,
            cache_config.block_size, cache_config.num_gpu_blocks_override,
            scheduler_config.max_num_batched_tokens)

        # set env vars for torch_sendnn to consume
        os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(
            vllm_config.model_config.max_model_len)
        if (envs_spyre.VLLM_SPYRE_USE_CB
                and vllm_config.model_config.max_model_len > 32 * 1024):
            logger.warning(
                'Max context length is too big. Currently only 32K (32768) ' \
                'context length is supported on Spyre for continuous ' \
                'batching. Results might be off!'
            )
        # min value 2 needed for VLLM_DT_MAX_BATCH_SIZE (compiler constraint)
        # Note that we can still have decodes of batch size 1 as the env var
        # only concerns the max batch size.
        os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(
            max(vllm_config.scheduler_config.max_num_seqs, 2))

        # max product of batch size x tkv supported by the Spyre compiler
        if ('granite-3.3-8b-instruct' in model_config.model
                and parallel_config.world_size == 4):
            # hard coded value for tensor parallel size 4 with the below model
            # https://huggingface.co/ibm-granite/granite-3.3-8b-instruct
            os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = str(128 * 1024)
            logger.info("Model granite-3.3-8b-instruct and tensor parallel " \
            "size 4 detected. Using VLLM_DT_MAX_BATCH_TKV_LIMIT = %d",
            128 * 1024)

            # If no HDMA p2psize override was specified, set 256MB
            if not os.getenv("FLEX_HDMA_P2PSIZE", None):
                os.environ["FLEX_HDMA_P2PSIZE"] = str(1024 * 1024 * 256)
                logger.info(
                    "Model granite-3.3-8b-instruct and tensor parallel size 4 "
                    "detected. Using FLEX_HDMA_P2PSIZE = %d",
                    1024 * 1024 * 256)
        else:
            # default value for any other model/ tensor parallel size
            default_max_batch_tkv_limit = \
                vllm_config.model_config.max_model_len * \
                vllm_config.scheduler_config.max_num_seqs
            os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = str(
                default_max_batch_tkv_limit)
            logger.info("No model / tensor parallel size specific value for " \
            "VLLM_DT_MAX_BATCH_TKV_LIMIT found. Using the default value " \
            "(max_model_len * max_batch_size): %d", default_max_batch_tkv_limit)

        # scheduling heuristic: prefill vs decode prioritization
        if envs_spyre.VLLM_SPYRE_N_TOKENS_PREFILL_PRIO == -1:
            logger.info(
                "Env var VLLM_SPYRE_N_TOKENS_PREFILL_PRIO for prefill/decode "
                "balancing unset. Defaulting to -1, which always prioritizes "
                "prefills (no scheduler heuristic/ balancing at all).")
        else:
            logger.info(
                "Env var VLLM_SPYRE_N_TOKENS_PREFILL_PRIO for prefill/decode "
                "balancing is set to %s. This means that prefills using up to "
                " %s tokens will always be prioritized over decodes.",
                envs_spyre.VLLM_SPYRE_N_TOKENS_PREFILL_PRIO,
                envs_spyre.VLLM_SPYRE_N_TOKENS_PREFILL_PRIO)

        cls._handle_disable_compilation(vllm_config, is_decoder)

    @classmethod
    def use_all_gather(cls) -> bool:
        """
        Whether to use allgather in LogitsProcessor to gather the logits.
        """
        return True

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        logger.warning("Pin memory is not supported on Spyre.")
        return False

    @classmethod
    def inference_mode(cls):
        """
        Spyre does not support `torch.inference_mode`.
        This allows to fall back to `torch.no_grad` when inference mode is set.
        """
        return torch.no_grad()

    @classmethod
    def get_warmup_shapes(cls, scheduler_config) -> tuple[dict[str, int], ...]:
        if cls._warmup_shapes is not None:
            return cls._warmup_shapes
        # load warmup shapes and sort by "speed"
        wup_prompt_lens = envs_spyre.VLLM_SPYRE_WARMUP_PROMPT_LENS or []
        if not all(pl % 64 == 0 for pl in wup_prompt_lens):
            raise RuntimeError(
                "All values in VLLM_SPYRE_WARMUP_PROMPT_LENS must be multiples "
                "of 64.")

        wup_batch_sizes = envs_spyre.VLLM_SPYRE_WARMUP_BATCH_SIZES or []
        if len(wup_prompt_lens) != len(wup_batch_sizes):
            raise RuntimeError(
                "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                "VLLM_SPYRE_WARMUP_BATCH_SIZES must have equal length")
        if scheduler_config.runner_type == "pooling":
            wup_new_tokens = [0] * len(wup_prompt_lens)
        else:
            wup_new_tokens = envs_spyre.VLLM_SPYRE_WARMUP_NEW_TOKENS or []
            if len(wup_new_tokens) != len(wup_prompt_lens):
                raise RuntimeError(
                    "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                    "VLLM_SPYRE_WARMUP_NEW_TOKENS must have equal length")

        logger.info("VLLM_SPYRE_WARMUP_PROMPT_LENS = %s", wup_prompt_lens)
        logger.info("VLLM_SPYRE_WARMUP_NEW_TOKENS = %s", wup_new_tokens)
        logger.info("VLLM_SPYRE_WARMUP_BATCH_SIZES = %s", wup_batch_sizes)

        cls._warmup_shapes = tuple(
            sorted([{
                'prompt_length': pl,
                'new_tokens': nt,
                'batch_size': bs
            } for pl, nt, bs in zip(wup_prompt_lens, wup_new_tokens,
                                    wup_batch_sizes)],
                   key=operator.itemgetter('batch_size', 'prompt_length')))
        return cls._warmup_shapes

    @classmethod
    def get_block_size(cls) -> int:
        return cls._block_size

    @classmethod
    def get_num_spyre_blocks_override(cls) -> int:
        return cls._num_spyre_blocks_override

    @classmethod
    def supports_v1(cls, model_config: ModelConfig) -> bool:
        """Returns whether the current platform can support v1 for the supplied
        model configuration.
        """
        return True

    @classmethod
    def validate_request(
        cls,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        processed_inputs: ProcessorInputs | None = None,
    ) -> None:
        """Raises if this request is unsupported on this platform"""
        if isinstance(params, PoolingParams):
            # Only validating generation requests for now
            return None

        if (params.prompt_logprobs is not None
                and not envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS):
            raise ValueError("Prompt logprobs must be enabled with "
                             "`VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS=1`")

        if isinstance(prompt, dict) and "prompt_token_ids" in prompt:
            prompt_len = len(prompt["prompt_token_ids"])
        elif processed_inputs is not None:
            if "encoder" in processed_inputs:
                raise ValueError("Encoder-decoder models not supported ")
            prompt_len = len(processed_inputs["prompt_token_ids"])
        else:
            # We need a prompt length to do any validation here
            return

        max_tokens = 0
        if params is not None and params.max_tokens is not None:
            max_tokens = params.max_tokens

        if envs_spyre.VLLM_SPYRE_USE_CB:
            # For continuous batching, check if the request is within the max
            # context length. This needs to take the padded prompt length
            # into account.

            # ceil division to pad to next block boundary
            prompt_padding_len = math.ceil(
                prompt_len / cls._block_size) * cls._block_size
            # we have to account for the token generated during prefill (-1)
            if (prompt_padding_len + max_tokens - 1
                    > cls._config.scheduler_config.max_model_len):
                raise ValueError(
                    "Could not add request: prompt length is "
                    f"{prompt_len} tokens, which gets padded to "
                    f"{prompt_padding_len} tokens, maximum number of output "
                    f"tokens is {max_tokens} tokens, but max model context "
                    f"length is {cls._config.scheduler_config.max_model_len}.")
        else:
            # For non-continuous batching, check if the request matches a warmup
            # shape
            assert cls._warmup_shapes is not None, "Warmup shapes must be set"
            if len(
                    cls._get_matching_warmup_shapes(
                        prompt_len=prompt_len,
                        max_tokens=max_tokens,
                        warmup_shapes=cls._warmup_shapes)) == 0:
                raise ValueError(
                    "No applicable warmup shape exists for "
                    f"combination of prompt length ({prompt_len} tokens) "
                    "and maximum number of output tokens to be "
                    f"generated ({max_tokens} tokens)")

    @classmethod
    def _get_matching_warmup_shapes(
            cls, prompt_len: int, max_tokens: int,
            warmup_shapes: tuple[dict[str, int], ...]) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request"""
        return [
            shape for shape in warmup_shapes
            if prompt_len <= shape['prompt_length']
            and max_tokens <= shape['new_tokens']
        ]

    @classmethod
    def _check_threading_config(cls, worker_count: int):
        """
        Check parallelism configuration to avoid CPU contention

        Libraries that support multi-threading (eg. OpenMP) default to
        parallelism based on the number of CPUs on the host. This can lead to
        CPU contention in containerized deployments especially when process
        forking is involved. This function provides better default behavior.
        """

        # The quay.io/ibm-aiu/spyre-base image includes shell scripts that
        # automatically set OMP_NUM_THREADS to the result of `nproc --all`.
        #
        # vLLM also already has logic around threading to be aware of,
        #  - sets TORCHINDUCTOR_COMPILE_THREADS=1 (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/env_override.py#L38-L39)
        #  - it will set OMP_NUM_THREADS=1 when using multiple workers (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/executor/multiproc_worker_utils.py#L304)
        #  - has configurations for OMP thread binding (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/envs.py#L435-L438)
        #    - the bind attempts to detect NUMA nodes (https://github.com/vllm-project/vllm/blob/baba0389f7e810a361fff5229ce20c2d5a2b1fac/vllm/v1/worker/cpu_worker.py#L111)

        assert worker_count > 0
        # Always print current env for awareness
        env_map = {env: os.getenv(env) for env in THREADING_ENVS}
        logger.info(
            "Initial threading configurations: %s",
            ' '.join([f"{env}={value}" for env, value in env_map.items()]))

        # Try to determine the CPU time/cores that we are allocated
        cpu_count: float | None = None
        detection_message = ""
        try:
            # try to query cgroup CPU limits
            with open('/sys/fs/cgroup/cpu.max') as f:
                quota_str, period_str = f.read().strip().split()

            if quota_str != 'max':
                quota = int(quota_str)
                period = int(period_str)
                cpu_count = quota / period
                detection_message = f"Detected cgroup CPU limit of {cpu_count}"

        except FileNotFoundError:
            # file may not exist if not running under cgroups v2
            pass
        except Exception as e:
            logger.debug(
                "Error parsing /sys/fs/cgroup/cpu.max to get CPU info",
                exc_info=e)

        # could try `nproc` here, but it is affected by
        # OMP_NUM_THREADS itself

        # try os.cpu_count() to get node CPU count
        if cpu_count is None and (cpu_count_res := os.cpu_count()) is not None:
            cpu_count = float(cpu_count_res)
            detection_message = \
                f"Detected {cpu_count} CPUs from `os.cpu_count()`"

        # NOTE: math.ceil can output a number for each worker that sums
        # to a total greater than cpu_count.
        cpus_per_worker = math.ceil(
            cpu_count / worker_count) if cpu_count is not None else None

        thread_warning = "Excessive threads may result in CPU contention. " \
             + "Note that each worker processes has its own thread pools." \
                if worker_count > 1 else ""
        failed_detection_message = "Unable to detect available CPUs to " \
            "validate threading configuration."

        if envs_spyre.VLLM_SPYRE_UPDATE_THREAD_CONFIG:
            if cpus_per_worker is None:
                raise RuntimeError(
                    f"{failed_detection_message} Use "
                    "VLLM_SPYRE_UPDATE_THREAD_CONFIG=0 and configure manually."
                )

            for env in THREADING_ENVS:
                os.environ[env] = str(cpus_per_worker)

            logger.info(
                "%s for %d workers. Since VLLM_SPYRE_UPDATE_THREAD_CONFIG is "
                "enabled, setting threading configurations to %d",
                detection_message, worker_count, cpus_per_worker)
            return

        # In the case that VLLM_SPYRE_UPDATE_THREAD_CONFIG is not enabled,
        # check configs and maybe log a warning
        if cpus_per_worker is None:
            logger.info("%s %s", failed_detection_message, thread_warning)
            return

        def _float_or_0(s: str) -> float:
            try:
                return float(s)
            except ValueError:
                return 0.0

        if any((value is None or _float_or_0(value) > 1.2 * cpus_per_worker)
               for value in env_map.values()):
            logger.warning(
                "%s %s for %d workers. Recommend setting each threading "
                "configuration to %d. Set VLLM_SPYRE_UPDATE_THREAD_CONFIG=1 "
                "to do this automatically.", thread_warning, detection_message,
                worker_count, cpus_per_worker)

    def get_max_output_tokens(self, prompt_len: int) -> int:
        """Return the size of biggest ```new_tokens``` of the \
            warmup shapes that fits the prompt length"""
        if self._warmup_shapes is None:
            return sys.maxsize

        max_new_tokens = 1
        for shape in self._warmup_shapes:
            if prompt_len <= shape['prompt_length']:
                max_new_tokens = max(max_new_tokens, shape['new_tokens'])

        return max_new_tokens

    @classmethod
    def _handle_disable_compilation(cls, vllm_config: VllmConfig,
                                    is_decoder: bool):
        """
        For decoder models we want to respect the `REQUIRE_PRECOMPILED_DECODERS`
        environment variable, which disallows torch_sendnn from compiling new
        graphs and only allows it to load pre-compiled graphs.
        In order to do this, we must load up some config from the torch_sendnn
        cache and check to make sure that the current vllm config matches,
        otherwise the cached artifacts cannot be used.

        For encoder models, we do not allow disabling compilation.
        """
        req_precompiled_decoder_env_var = "REQUIRE_PRECOMPILED_DECODERS"

        disable_flag = os.getenv(req_precompiled_decoder_env_var, "").lower()
        if disable_flag not in ("1", "true"):
            return

        # If this isn't a decoder model, re-enable compilation
        if not is_decoder:
            logger.info("Unsetting %s because %s is not a decoder model",
                        DISABLE_COMPILATION_ENV_VAR,
                        vllm_config.model_config.model)
            os.environ[DISABLE_COMPILATION_ENV_VAR] = "false"
            return

        # If the user asked to disable compilation, then we need to enforce that
        # they setup their cache
        torch_cache_dir = os.getenv("TORCH_SENDNN_CACHE_DIR", None)
        torch_cache_enabled = bool(
            int(os.getenv("TORCH_SENDNN_CACHE_ENABLE", "0")))

        if not torch_cache_dir or not torch_cache_enabled:
            raise ValueError(
                f"{req_precompiled_decoder_env_var}=1 requires setting"
                " TORCH_SENDNN_CACHE_DIR to a valid path and setting " \
                "TORCH_SENDNN_CACHE_ENABLE=1")

        compilation_config_path = Path(
            torch_cache_dir) / PRE_COMPILE_MODEL_CONFIG_FILENAME
        compilation_catalog_path = Path(
            torch_cache_dir) / PRE_COMPILE_MODEL_CATALOG_FILENAME

        if not compilation_catalog_path.exists() and \
            not compilation_config_path.exists():
            raise ValueError(
                f"{req_precompiled_decoder_env_var}=1 was set, but no "
                f"pre-compiled model config was found in the "
                f"TORCH_SENDNN_CACHE_DIR: {str(compilation_config_path)} or"
                f"{str(compilation_catalog_path)} does not exist")

        if not compilation_catalog_path.is_file() and \
            not compilation_config_path.is_file():
            raise ValueError(
                "{req_precompiled_decoder_env_var}=1 was set, but the "
                "pre-compiled model config is not a file")

        matching_config = None

        # Note: In below implementation we don't tell user exactly what's wrong
        # but we do "warn" them about mismatch and provide the list of supported
        # configuration along with what they have given us.
        if compilation_catalog_path.is_file():
            with open(compilation_catalog_path) as f:
                try:
                    pre_compile_catalog = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Precompiled catalog {str(compilation_catalog_path)}"
                        " is not a valid JSON file") from e
            match_result = cls.__match_from_pre_compile_catalog(
                pre_compile_catalog, vllm_config)

            if match_result == -1:
                # No match found
                logger.warning(
                    "Provided vllm configuration doesn't match any of the "
                    "pre-compiled model configurations. Catalog: \n%s\n "
                    "vllm_config: \n%s", str(compilation_catalog_path),
                    str(vllm_config))

                # Return with warning
                return
            else:
                matching_config = pre_compile_catalog[match_result]

        elif compilation_config_path.is_file():
            with open(compilation_config_path) as f:
                try:
                    compilation_config = json.load(f)
                except json.JSONDecodeError as e:
                    raise ValueError("Precompiled model config "
                                     f"{str(compilation_config_path)} was "
                                     "not valid json") from e
            match_result = cls.__match_from_model_config_file(
                compilation_config, vllm_config)
            if not match_result:
                logger.warning(
                    "Provided vllm configuration doesn't match any of the "
                    "pre-compiled model")
                # Return with warning
                return
            else:
                matching_config = compilation_config

        if matching_config:
            # Check vllm_spyre version
            try:
                from vllm_spyre._version import version as vllm_spyre_version
                if matching_config["vllm_spyre_version"] != vllm_spyre_version:
                    # Can be converted to ValueError if we want to be strict
                    # with checking
                    logger.warning(
                        "Model was compiled on vllm-spyre "
                        "%s but the current vllm_spyre version is %s",
                        matching_config['vllm_spyre_version'],
                        vllm_spyre_version)
            except ImportError:
                logger.warning(
                    "Cannot validate vllm_spyre version against pre-compiled "
                    "model config")

            # Check model name
            model_name = matching_config["data"]["MODEL_NAME"]

            if vllm_config.model_config.model != model_name:
                # We don't have a way to easily ensure that the compiled model
                # is the same as the one that the user is loading. We can only
                # warn here if the names do not match.
                logger.warning(
                    "Configured model name is %s but the pre-compiled model "
                    "config has name %s. Please ensure this is the correct "
                    "model", vllm_config.model_config.model, model_name)

    @classmethod
    def __match_from_pre_compile_catalog(cls, pre_compile_catalog: dict,
                                         vllm_config: VllmConfig) -> int:
        """Function to find the pre-compile model configuration that matches
        the provided vllm_config.
        """

        # Iterate through catalog file to find if any configuration matches,
        # otherwise, return False
        for idx, config in enumerate(pre_compile_catalog):
            # Compare each key-value pair with values in vllm_config
            match_result = cls.__match_from_model_config_file(
                config, vllm_config)
            if match_result:
                return idx
        return -1

    @classmethod
    def __match_from_model_config_file(cls, compilation_config: dict,
                                       vllm_config: VllmConfig) -> bool:
        """Function to validate if vllm configuration provided matches
        pre-compile model configuration
        """

        # Validate configurations
        vllm_configs = compilation_config["data"]

        # TP size
        tp_size = vllm_configs["NUM_AIUS"]
        if vllm_config.parallel_config.tensor_parallel_size != tp_size:
            return False

        if "VLLM_SPYRE_WARMUP_PROMPT_LENS" in vllm_configs:
            if envs_spyre.VLLM_SPYRE_USE_CB:
                return False
            else:
                get_list = lambda x: [int(i) for i in x.split(",")]

                prompt_lens = get_list(
                    vllm_configs["VLLM_SPYRE_WARMUP_PROMPT_LENS"])
                new_tokens = get_list(
                    vllm_configs["VLLM_SPYRE_WARMUP_NEW_TOKENS"])
                batch_sizes = get_list(
                    vllm_configs["VLLM_SPYRE_WARMUP_BATCH_SIZES"])

                if prompt_lens != envs_spyre.VLLM_SPYRE_WARMUP_PROMPT_LENS:
                    return False

                if new_tokens != envs_spyre.VLLM_SPYRE_WARMUP_NEW_TOKENS:
                    return False

                if batch_sizes != envs_spyre.VLLM_SPYRE_WARMUP_BATCH_SIZES:
                    return False
        else:
            if not envs_spyre.VLLM_SPYRE_USE_CB:
                return False

            context_len = vllm_configs["VLLM_DT_MAX_CONTEXT_LEN"]
            batch_size = vllm_configs["VLLM_DT_MAX_BATCH_SIZE"]

            if context_len != vllm_config.model_config.max_model_len:
                return False

            if batch_size != vllm_config.scheduler_config.max_num_seqs:
                return False

        return True

import sys

# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get("triton"):
        del sys.modules["triton"]

import math
import operator
import os
from typing import TYPE_CHECKING, Union

import torch
from transformers.models.granite import GraniteConfig
from vllm.inputs import ProcessorInputs, PromptType
from vllm.logger import init_logger

try:
    # pre 0.11.1 compatibility
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    from vllm.utils.argparse_utils import FlexibleArgumentParser

if TYPE_CHECKING:
    # NB: We can't eagerly import many things from vllm since vllm.config
    # will import this file. These would lead to circular imports
    from vllm.config import ModelConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
    from vllm.sampling_params import SamplingParams
else:
    ModelConfig = None
    VllmConfig = None
    SamplingParams = None
    PoolingParams = None
from vllm.platforms import Platform, PlatformEnum

import vllm_spyre.envs as envs_spyre
from vllm_spyre.compilation_utils import handle_disable_compilation

logger = init_logger(__name__)

THREADING_ENVS = [
    "OMP_NUM_THREADS",
    # "TORCHINDUCTOR_COMPILE_THREADS", # vLLM wants this set to 1
    "DT_PARALLEL_THREADS",  # affects the compilation during warmup
    # set these for good measure
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
]


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
    _config: VllmConfig = None
    _torch_sendnn_version = None

    # Backend for dynamic compilation ops
    # See vllm batched_count_greater_than method
    simple_compile_backend: str = envs_spyre.VLLM_SPYRE_SIMPLE_COMPILE_BACKEND

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
        # 🌶️🌶️🌶️ Patch in our perf logger before the engine is created
        from vllm_spyre.v1.metrics import patch_async_llm_stat_loggers

        patch_async_llm_stat_loggers()

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
            raise ValueError("vllm-spyre is only supported with vLLM v1. Please set VLLM_USE_V1=1")
        elif not is_decoder and not is_pooling:
            raise ValueError("Only the 'generate' and 'pooling' runners are supported")

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm_spyre.v1.worker.spyre_worker.SpyreWorker"

        cls._check_threading_config(parallel_config.world_size)

        # set env vars based on the model
        if is_decoder:
            os.environ["FLEX_OVERWRITE_NMB_FRAME"] = "true"
            os.environ["COMPILATION_MODE"] = "offline_decoder"
        if is_pooling:
            os.environ["FLEX_OVERWRITE_NMB_FRAME"] = "false"
            os.environ["COMPILATION_MODE"] = "offline"

        assert (
            envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL and envs_spyre.VLLM_SPYRE_USE_CB
        ) or not envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL, (
            "Cannot use chunked prefill without continuous batching."
        )

        if envs_spyre.VLLM_SPYRE_USE_CB and is_decoder:
            if envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL:
                scheduler_config.scheduler_cls = (
                    "vllm_spyre.v1.core.scheduler.ChunkedPrefillSpyreScheduler"
                )
            else:
                scheduler_config.scheduler_cls = (
                    "vllm_spyre.v1.core.scheduler.ContinuousBatchingSpyreScheduler"
                )
            if envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS:
                raise ValueError("Prompt logprobs not supported with continuous batching")
            if (
                vllm_config.model_config.quantization
                and vllm_config.scheduler_config.max_num_seqs == 1
            ):
                raise ValueError("Batch size 1 not supported for fp8 continuous batching.")
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
                max_seq_len = max(max_seq_len, shape["prompt_length"] + shape["new_tokens"])

            if envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS and max_batch_size > 1:
                raise ValueError("Prompt logprobs only supported with batch size 1")

            # verify that warmup shapes are not too large
            model_config.get_and_verify_max_len(max_model_len=max_seq_len)

            # override stuff
            model_config.max_model_len = max_seq_len
            scheduler_config.max_num_seqs = max_batch_size

            scheduler_config.scheduler_cls = (
                "vllm_spyre.v1.core.scheduler.StaticBatchingSpyreScheduler"
            )

        # Hardcode some things for granite-3.3-8b-instruct
        if cls.is_granite_3_8b(vllm_config.model_config):
            cls.configure_granite_3_8b(vllm_config)

        # To disable any paged attention ops in the base scheduler, we:
        # - Set the block size (in tokens) to the maximum sequence length
        #       so that the scheduler thinks an entire sequence will fit in
        #       one single block.
        # - Set `max_num_batched_tokens` to the size of a full batch of full
        #       length requests, so that the scheduler will always have token
        #       budget available to schedule a full batch
        if cache_config is not None:
            cache_config.block_size = model_config.max_model_len
            if not envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL:
                scheduler_config.max_num_batched_tokens = (
                    model_config.max_model_len * scheduler_config.max_num_seqs
                )
            else:
                # TODO: ideally, this would be user-configurable from CLI/engine
                # args instead of with the internal env var, but that requires a
                # way to detect if value set by vllm or by the user
                if (chunk_len := os.getenv("VLLM_DT_CHUNK_LEN")) is None:
                    os.environ["VLLM_DT_CHUNK_LEN"] = str(scheduler_config.max_num_batched_tokens)
                else:
                    try:
                        chunk_len_int = int(chunk_len)
                    except (ValueError, TypeError) as e:
                        raise Exception("VLLM_DT_CHUNK_LEN must be an integer") from e

                    logger.info(
                        "VLLM_DT_CHUNK_LEN was provided. Overriding max_num_batched_tokens to %d",
                        chunk_len_int,
                    )
                    scheduler_config.max_num_batched_tokens = chunk_len_int

                assert scheduler_config.max_num_batched_tokens % cls._block_size == 0, (
                    "`max_num_batched_tokens` must"
                    f" be divisible by the block size ({cls._block_size}) "
                    "to enable chunked prefill. It was set to "
                    f"`{scheduler_config.max_num_batched_tokens}`. Please "
                    "set `--max-num-batched-tokens` to a number that satisfies "
                    "this constraint."
                )

        logger.info(
            "Overriding configurations based on warmup shapes. "
            "max_model_len=%d, max_num_seqs=%d, block_size=%d, "
            "max_num_batched_tokens=%d",
            model_config.max_model_len,
            scheduler_config.max_num_seqs,
            cache_config.block_size,
            scheduler_config.max_num_batched_tokens,
        )

        # set env vars for torch_sendnn to consume
        os.environ["VLLM_DT_MAX_CONTEXT_LEN"] = str(vllm_config.model_config.max_model_len)
        if envs_spyre.VLLM_SPYRE_USE_CB and vllm_config.model_config.max_model_len > 32 * 1024:
            logger.warning(
                "Max context length is too big. Currently only 32K (32768) "
                "context length is supported on Spyre for continuous "
                "batching. Results might be off!"
            )
        # min value 2 needed for VLLM_DT_MAX_BATCH_SIZE (compiler constraint)
        # Note that we can still have decodes of batch size 1 as the env var
        # only concerns the max batch size.
        os.environ["VLLM_DT_MAX_BATCH_SIZE"] = str(
            max(vllm_config.scheduler_config.max_num_seqs, 2)
        )

        if not os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"):
            # max product of batch size x tkv supported by the Spyre compiler
            default_max_batch_tkv_limit = (
                vllm_config.model_config.max_model_len * vllm_config.scheduler_config.max_num_seqs
            )

            os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = str(default_max_batch_tkv_limit)
            logger.info(
                "No model / tensor parallel size specific value for "
                "VLLM_DT_MAX_BATCH_TKV_LIMIT found. Using the default value "
                "(max_model_len * max_batch_size): %d",
                default_max_batch_tkv_limit,
            )

        # Compare requested runtime configuration with supported configurations
        # Don't use top-level import to avoid circular import error
        from vllm_spyre.config.runtime_config_validator import validate_runtime_configuration

        warmup_shape_tuples = (
            [(ws["prompt_length"], ws["new_tokens"], ws["batch_size"]) for ws in cls._warmup_shapes]
            if cls._warmup_shapes and not envs_spyre.VLLM_SPYRE_USE_CB
            else None
        )

        validate_runtime_configuration(
            model_config=model_config,
            tp_size=parallel_config.tensor_parallel_size,
            max_model_len=model_config.max_model_len,
            max_num_seqs=scheduler_config.max_num_seqs,
            warmup_shapes=warmup_shape_tuples,
        )

        handle_disable_compilation(vllm_config, is_decoder)

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
                "All values in VLLM_SPYRE_WARMUP_PROMPT_LENS must be multiples of 64."
            )

        wup_batch_sizes = envs_spyre.VLLM_SPYRE_WARMUP_BATCH_SIZES or []
        if len(wup_prompt_lens) != len(wup_batch_sizes):
            raise RuntimeError(
                "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                "VLLM_SPYRE_WARMUP_BATCH_SIZES must have equal length"
            )
        if scheduler_config.runner_type == "pooling":
            wup_new_tokens = [0] * len(wup_prompt_lens)
        else:
            wup_new_tokens = envs_spyre.VLLM_SPYRE_WARMUP_NEW_TOKENS or []
            if len(wup_new_tokens) != len(wup_prompt_lens):
                raise RuntimeError(
                    "The lists in VLLM_SPYRE_WARMUP_PROMPT_LENS and "
                    "VLLM_SPYRE_WARMUP_NEW_TOKENS must have equal length"
                )

        logger.info("VLLM_SPYRE_WARMUP_PROMPT_LENS = %s", wup_prompt_lens)
        logger.info("VLLM_SPYRE_WARMUP_NEW_TOKENS = %s", wup_new_tokens)
        logger.info("VLLM_SPYRE_WARMUP_BATCH_SIZES = %s", wup_batch_sizes)

        cls._warmup_shapes = tuple(
            sorted(
                [
                    {"prompt_length": pl, "new_tokens": nt, "batch_size": bs}
                    for pl, nt, bs in zip(wup_prompt_lens, wup_new_tokens, wup_batch_sizes)
                ],
                key=operator.itemgetter("batch_size", "prompt_length"),
            )
        )
        return cls._warmup_shapes

    @classmethod
    def get_block_size(cls) -> int:
        return cls._block_size

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

        # The PoolingParams import is lazy here because it imports vllm.config,
        # which will in turn import this file again.
        from vllm.pooling_params import PoolingParams

        if isinstance(params, PoolingParams):
            # Only validating generation requests for now
            return None

        # Note: Currently prompt logprobs are not supported, therefore
        # envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS is hardcoded to False
        if params.prompt_logprobs is not None and not envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS:
            raise ValueError("Prompt logprobs are currently not supported.")

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
            prompt_padding_len = math.ceil(prompt_len / cls._block_size) * cls._block_size
            if prompt_padding_len + max_tokens > cls._config.model_config.max_model_len:
                raise ValueError(
                    "Could not add request: prompt length is "
                    f"{prompt_len} tokens, which gets padded to "
                    f"{prompt_padding_len} tokens, maximum number of output "
                    f"tokens is {max_tokens} tokens, but max model context "
                    f"length is {cls._config.model_config.max_model_len}."
                )
        else:
            # For non-continuous batching, check if the request matches a warmup
            # shape
            assert cls._warmup_shapes is not None, "Warmup shapes must be set"
            if (
                len(
                    cls._get_matching_warmup_shapes(
                        prompt_len=prompt_len,
                        max_tokens=max_tokens,
                        warmup_shapes=cls._warmup_shapes,
                    )
                )
                == 0
            ):
                raise ValueError(
                    "No applicable warmup shape exists for "
                    f"combination of prompt length ({prompt_len} tokens) "
                    "and maximum number of output tokens to be "
                    f"generated ({max_tokens} tokens)"
                )

    @classmethod
    def _get_matching_warmup_shapes(
        cls, prompt_len: int, max_tokens: int, warmup_shapes: tuple[dict[str, int], ...]
    ) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request"""
        return [
            shape
            for shape in warmup_shapes
            if prompt_len <= shape["prompt_length"] and max_tokens <= shape["new_tokens"]
        ]

    @classmethod
    def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
        if parser is not None:
            parser.set_defaults(enable_prefix_caching=False)

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
            " ".join([f"{env}={value}" for env, value in env_map.items()]),
        )

        # Try to determine the CPU time/cores that we are allocated
        cpu_count: float | None = None
        detection_message = ""

        if (num_cpu := envs_spyre.VLLM_SPYRE_NUM_CPUS) > 0:
            cpu_count = num_cpu
            detection_message = f"VLLM_SPYRE_NUM_CPUS is set to {cpu_count}"
        else:
            try:
                # try to query cgroup CPU limits
                with open("/sys/fs/cgroup/cpu.max") as f:
                    quota_str, period_str = f.read().strip().split()

                if quota_str != "max":
                    quota = int(quota_str)
                    period = int(period_str)
                    cpu_count = quota / period
                    detection_message = f"Detected cgroup CPU limit of {cpu_count}"

            except FileNotFoundError:
                # file may not exist if not running under cgroups v2
                pass
            except Exception as e:
                logger.debug("Error parsing /sys/fs/cgroup/cpu.max to get CPU info", exc_info=e)

            # try psutil to get physical core count
            if cpu_count is None:
                try:
                    import psutil

                    cpu_count = float(psutil.cpu_count(logical=False))
                    detection_message = (
                        f"Detected {cpu_count} physical CPUs from psutil.cpu_count(logical=False)"
                    )
                except ImportError:
                    logger.info("Install psutil to count physical CPU cores")
                    pass
                except Exception as e:
                    logger.debug("Error using psutil", exc_info=e)

            # could try `nproc` here, but it is affected by
            # OMP_NUM_THREADS itself

            # try os.cpu_count() to get node CPU count
            if cpu_count is None and (cpu_count_res := os.cpu_count()) is not None:
                cpu_count = float(cpu_count_res)
                detection_message = f"Detected {cpu_count} CPUs from `os.cpu_count()`"

        # NOTE: math.ceil can output a number for each worker that sums
        # to a total greater than cpu_count.
        cpus_per_worker = math.ceil(cpu_count / worker_count) if cpu_count is not None else None

        thread_warning = (
            "Excessive threads may result in CPU contention. "
            + "Note that each worker processes has its own thread pools."
            if worker_count > 1
            else ""
        )
        failed_detection_message = (
            "Unable to detect available CPUs to validate threading configuration."
        )

        if envs_spyre.VLLM_SPYRE_UPDATE_THREAD_CONFIG:
            if cpus_per_worker is None:
                raise RuntimeError(
                    f"{failed_detection_message} Set VLLM_SPYRE_NUM_CPUS or "
                    "use VLLM_SPYRE_UPDATE_THREAD_CONFIG=0 and configure "
                    "manually."
                )

            for env in THREADING_ENVS:
                os.environ[env] = str(cpus_per_worker)

            logger.info(
                "%s for %d workers. Since VLLM_SPYRE_UPDATE_THREAD_CONFIG is "
                "enabled, setting threading configurations to %d",
                detection_message,
                worker_count,
                cpus_per_worker,
            )
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

        if any(
            (value is None or _float_or_0(value) > 1.2 * cpus_per_worker)
            for value in env_map.values()
        ):
            logger.warning(
                "%s %s for %d workers. Recommend setting each threading "
                "configuration to %d. Set VLLM_SPYRE_UPDATE_THREAD_CONFIG=1 "
                "to do this automatically.",
                thread_warning,
                detection_message,
                worker_count,
                cpus_per_worker,
            )

    def get_max_output_tokens(self, prompt_len: int) -> int:
        """Return the size of biggest ```new_tokens``` of the \
            warmup shapes that fits the prompt length"""
        if self._warmup_shapes is None:
            # ceil division to pad to next block boundary
            padded_prompt_len = math.ceil(prompt_len / self._block_size) * self._block_size
            max_new_tokens = self._config.model_config.max_model_len - padded_prompt_len
            return max_new_tokens

        max_new_tokens = 1
        for shape in self._warmup_shapes:
            if prompt_len <= shape["prompt_length"]:
                max_new_tokens = max(max_new_tokens, shape["new_tokens"])

        return max_new_tokens

    @classmethod
    def configure_granite_3_8b(cls, vllm_config: VllmConfig):
        """
        Configure hard coded values for the model
        https://huggingface.co/ibm-granite/granite-3.3-8b-instruct
        """
        parallel_config = vllm_config.parallel_config

        if parallel_config.world_size != 4:
            # only override configs for TP=4
            return

        tkv_128k = 128 * 1024
        if not os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"):
            os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = str(tkv_128k)
            logger.info(
                "Model granite-3.3-8b-instruct and tensor parallel "
                "size 4 detected. Using VLLM_DT_MAX_BATCH_TKV_LIMIT = %d",
                tkv_128k,
            )
        elif os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT") != str(tkv_128k):
            logger.warning(
                "VLLM_DT_MAX_BATCH_TKV_LIMIT was set to %s, not "
                "overriding to the granite-3.3-8b-instruct default of %d",
                os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"),
                tkv_128k,
            )

        # If no HDMA p2psize override was specified, set 256MB
        p2psize_256m = 256 * 1024 * 1024
        if not os.getenv("FLEX_HDMA_P2PSIZE"):
            os.environ["FLEX_HDMA_P2PSIZE"] = str(p2psize_256m)
            logger.info(
                "Model granite-3.3-8b-instruct and tensor parallel size 4 "
                "detected. Using FLEX_HDMA_P2PSIZE = %d",
                p2psize_256m,
            )
        elif os.getenv("FLEX_HDMA_P2PSIZE") != str(p2psize_256m):
            logger.warning(
                "FLEX_HDMA_P2PSIZE was set to %s, not using the "
                "granite-3.3-8b-instruct default of %d",
                os.getenv("FLEX_HDMA_P2PSIZE"),
                p2psize_256m,
            )

        # Override the total number of KV cache blocks based on what we know
        # will fit. (Unless user already set `--num-gpu-blocks-override`)
        # TODO: remove this once we have correct free memory info available
        if cls.sendnn_configured() and ((0, 0, 0) < cls.sendnn_version() < (1, 0, 3)):
            # Older versions of torch_sendnn use the previous override of ~2k
            # blocks.
            # NB: A version of (0, 0, 0) means that the version of torch_sendnn
            # could not be determined, and we assume this means we have a dev
            # install of newer code.
            blocks_override = 2080
        else:
            # If torch_sendnn is not configured or we have a newer torch_sendnn
            # install, use the newer 8k override.
            blocks_override = 8192

        if vllm_config.cache_config.num_gpu_blocks_override is None:
            vllm_config.cache_config.num_gpu_blocks_override = blocks_override
            logger.info(
                "Model granite-3.3-8b-instruct and tensor parallel size 4 "
                "detected. Overriding available KV Cache blocks to %d",
                blocks_override,
            )
        elif vllm_config.cache_config.num_gpu_blocks_override != blocks_override:
            logger.warning(
                "--num-gpu-blocks-override was set to %d, not using the "
                "granite-3.3-8b-instruct default of %d",
                vllm_config.cache_config.num_gpu_blocks_override,
                blocks_override,
            )

        # hard-coded value for max_num_batched_tokens with chunked prefill
        if (
            envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL
            and envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn"
            and os.getenv("VLLM_DT_CHUNK_LEN") is None
        ):
            logger.info(
                "Model granite-3.3-8b-instruct and tensor "
                "parallel size 4 with chunked prefill detected. Setting "
                "--max-num-batched-tokens 1024"
            )
            vllm_config.scheduler_config.max_num_batched_tokens = 1024

    @classmethod
    def is_granite_3_8b(cls, model_config: ModelConfig):
        """Returns true if we have a model that looks like
        ibm-granite/granite-3.3-8b-instruct"""
        if not isinstance(model_config.hf_config, GraniteConfig):
            # Not granite at all
            return False

        return (
            model_config.hf_config.num_hidden_layers == 40
            and model_config.hf_config.max_position_embeddings == 131072
            and model_config.hf_config.hidden_size == 4096
            and model_config.hf_config.vocab_size == 49159
            and model_config.hf_config.num_key_value_heads == 8
            and model_config.hf_config.num_attention_heads == 32
        )

    @classmethod
    def sendnn_configured(cls) -> bool:
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn":
            try:
                from torch_sendnn._version import __version__ as version_str

                sem_ver = version_str.split("+")[0]
                cls._torch_sendnn_version = tuple(map(int, sem_ver.split(".")))
                return True
            except ImportError as err:
                raise RuntimeError("sendnn backend requires torch_sendnn") from err
        return False

    @classmethod
    def sendnn_version(cls):
        if cls.sendnn_configured():
            return cls._torch_sendnn_version
        return (0, 0, 0)

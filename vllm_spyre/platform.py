import inspect
import sys

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
from typing import TYPE_CHECKING, Optional, Union

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


class classproperty:

    def __init__(self, func):
        self.func = func

    def __get__(self, instance, owner):
        return self.func(owner)


@property  # type: ignore
def is_v1_compatible(self) -> bool:
    architectures = getattr(self.hf_config, "architectures", [])
    patterns = ["Bert", "Roberta"]
    if any(pat in arch for arch in architectures for pat in patterns):
        return True
    import vllm.model_executor.models as me_models
    return me_models.ModelRegistry.is_v1_compatible(architectures)


class SpyrePlatform(Platform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    _device_type: str = "cpu"
    # compressed-tensors supported by
    # https://github.com/foundation-model-stack/fms-model-optimizer/blob/main/fms_mo/aiu_addons/__init__.py
    supported_quantization: list[str] = ["gptq", "compressed-tensors"]
    _warmup_shapes: Optional[tuple[dict[str, int], ...]] = None
    _block_size: int = 64  # hardcoded Spyre constraint for now
    _num_spyre_blocks_override: int = -1  # override num of KV cache blocks
    _config: VllmConfig = None

    @classproperty
    def device_type(cls):
        # TODO: temporary hack while BertModels
        # inherit SupportsV0Only in vllm upstream.
        import vllm.model_executor.models as me_models
        from vllm.config import ModelConfig

        # no need to patch after the model_config change
        if 'model_config' not in \
                inspect.getfullargspec(me_models.ModelRegistry.is_v1_compatible).args:
            ModelConfig.is_v1_compatible = is_v1_compatible
        return cls._device_type

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "spyre"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """
        Check if the current platform supports async output.
        """
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cls._config = vllm_config
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        if getattr(scheduler_config, "is_multi_step", False):
            raise NotImplementedError

        # Can be simplified after the deprecation of `model_config.task` in
        # vllm > 0.10.0
        is_decoder = "generate" in model_config.supported_tasks if (
            model_config.task == "auto"
            or model_config.task is None) else model_config.task == "generate"

        is_pooling = "embed" in model_config.supported_tasks if (
            model_config.task == "auto"
            or model_config.task is None) else model_config.task == "embed"

        if not bool(int(os.getenv("VLLM_USE_V1", "1"))):
            raise ValueError("vllm-spyre is only supported with vLLM v1. "
                             "Please set VLLM_USE_V1=1")
        elif not is_decoder and not is_pooling:
            raise ValueError("Only the 'generate' and 'embed' tasks are "
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
        else:
            # Static batching or embedding model.
            # Override --max-num-seqs to the biggest warmup batch size
            # And override --max-model-len to the biggest warmup sequence
            cls._warmup_shapes = None
            max_model_len = model_config.max_model_len
            spyre_warmup_shapes = cls.get_warmup_shapes(
                scheduler_config, max_model_len)
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
    def get_warmup_shapes(
            cls,
            scheduler_config,
            max_model_len: int = sys.maxsize) -> tuple[dict[str, int], ...]:
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

        for shape in cls._warmup_shapes:
            max_seq_len = shape["prompt_length"] + shape["new_tokens"]
            if max_seq_len > max_model_len:
                raise RuntimeError(
                    f"Warmup shape [{shape['batch_size']},"
                    f" {shape['prompt_length']}, {shape['new_tokens']}]"
                    f" results in a maximum sequence length of "
                    f"{max_seq_len} which is longer that what the model "
                    f"supports ({max_model_len})")
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
        processed_inputs: Optional[ProcessorInputs] = None,
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
        cpu_count: Optional[float] = None
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

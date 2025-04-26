import operator
from typing import TYPE_CHECKING, Optional, Union

import torch
from vllm.inputs import ProcessorInputs, PromptType
from vllm.inputs.parse import is_token_prompt
from vllm.logger import init_logger
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig
else:
    ModelConfig = None
    VllmConfig = None
import vllm.envs as envs
from vllm.platforms import Platform, PlatformEnum

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)


class SpyrePlatform(Platform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"
    supported_quantization: list[str] = ["gptq"]
    _warmup_shapes: Optional[tuple[dict[str, int], ...]] = None

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
        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config
        model_config = vllm_config.model_config

        if scheduler_config.is_multi_step:
            raise NotImplementedError

        # continuous batching related checks
        if envs_spyre.VLLM_SPYRE_USE_CB and not envs.VLLM_USE_V1:
            raise NotImplementedError(
                "Continuous batching is only implemented for vLLM V1")

        # Near future TODO: vLLM will have an api to check whether v0 or v1 is
        # used that isn't just checking the environment variable

        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm_spyre.v1.worker.spyre_worker.SpyreWorker"
            else:
                parallel_config.worker_cls = \
                    "vllm_spyre.worker.spyre_worker.SpyreWorker"

        if envs.VLLM_USE_V1:
            # As of 0.7.3 the scheduler for V1 isn't actually pluggable like
            # this yet
            if envs_spyre.VLLM_SPYRE_USE_CB:
                scheduler_config.scheduler_cls = \
                    "vllm_spyre.v1.core.scheduler.ContinuousBatchingSpyreScheduler"
            else:
                scheduler_config.scheduler_cls = \
                    "vllm_spyre.v1.core.scheduler.StaticBatchingSpyreScheduler"
        else:
            scheduler_config.scheduler_cls = \
                "vllm_spyre.core.scheduler.SpyreScheduler"

        if not envs_spyre.VLLM_SPYRE_USE_CB:
            # Override --max-num-seqs to the biggest warmup batch size
            # And override --max-model-len to the biggest warmup sequence
            cls._warmup_shapes = None
            spyre_warmup_shapes = cls.get_warmup_shapes(scheduler_config)
            max_batch_size = 0
            max_seq_len = 0
            for shape in spyre_warmup_shapes:
                max_batch_size = max(max_batch_size, shape['batch_size'])
                max_seq_len = max(max_seq_len,
                                  shape['prompt_length'] + shape['new_tokens'])

        if envs.VLLM_USE_V1:
            if envs_spyre.VLLM_SPYRE_USE_CB:
                # For continuous batching we use max_num_seqs to control
                # the max batch size respecting AIU Spyre KV cache size
                scheduler_config.max_num_seqs =\
                    envs_spyre.VLLM_SPYRE_MAX_BATCH_SIZE
                # ToDo: this function check_and_update_config is called twice:
                # 1st time scheduler_config.max_num_seqs is what user sets
                # 2nd time scheduler_config.max_num_seqs is 128
            else:
                # The v0 scheduler will run out of blocks if this is overridden
                scheduler_config.max_num_seqs = max_batch_size

        cache_config = vllm_config.cache_config

        if cache_config and model_config:
            # Cache and model config aren't set in the individual worker procs
            # These are set in the main engine process

            # To disable any paged attention ops in the base scheduler, we both:
            # - Set the block size (in tokens) to the maximum sequence length
            #       so that the scheduler thinks an entire sequence will fit in
            #       one single block.
            # - Set the number of blocks to the maximum number of sequences, so
            #       the scheduler always thinks there's a block available
            if not envs_spyre.VLLM_SPYRE_USE_CB:
                model_config.max_model_len = max_seq_len
            cache_config.block_size = model_config.max_model_len

            if envs.VLLM_USE_V1:
                # The V1 scheduler actually needs 2 blocks for each sequence...
                cache_config.num_gpu_blocks_override = \
                    scheduler_config.max_num_seqs * 2
            else:
                cache_config.num_gpu_blocks_override = \
                    scheduler_config.max_num_seqs

            logger.info(
                "Overriding configurations based on warmup shapes. "
                "max_model_len=%d, max_num_seqs=%d, block_size=%d, "
                "num_gpu_blocks_override=%d", model_config.max_model_len,
                scheduler_config.max_num_seqs, cache_config.block_size,
                cache_config.num_gpu_blocks_override)

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

        if not envs_spyre.VLLM_SPYRE_USE_CB:
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

        if is_token_prompt(prompt):
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
            # context length.
            # The V1 engine will check the prompt length, but not the prompt +
            # max_tokens.
            if (prompt_len + max_tokens
                    > envs_spyre.VLLM_SPYRE_MAX_CONTEXT_LENGTH):
                raise ValueError(
                    "Could not add request: prompt length is "
                    f"{prompt_len} tokens, maximum number of output tokens is "
                    f"{max_tokens} tokens, but max model context length is "
                    f"{envs_spyre.VLLM_SPYRE_MAX_CONTEXT_LENGTH}.")
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

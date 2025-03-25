import operator
import os
from typing import TYPE_CHECKING, Optional

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None
import vllm.envs as envs
from vllm.platforms import Platform, PlatformEnum

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)


class SpyrePlatform(Platform):
    _enum = PlatformEnum.OOT
    device_name: str = "spyre"
    device_type: str = "cpu"
    supported_quantization: list[str] = ["gptq"]

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

        # Near future TODO: vLLM will have an api to check whether v0 or v1 is
        # used that isn't just checking the environment variable

        if parallel_config.worker_cls == "auto":
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm_spyre.v1.worker.spyre_worker.SpyreWorker"

                # Forking is required here because this class is used to set up
                # the warmup shapes, and the workers that are now in separate
                # processes need to retrieve them.
                # If we can refactor the workers to setup the warmup shapes
                # themselves, then we can support process spawning too.
                if envs.VLLM_WORKER_MULTIPROC_METHOD != "fork":
                    logger.warning("V1 integration requires "
                                   "VLLM_WORKER_MULTIPROC_METHOD=fork")
                    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork"
            else:
                parallel_config.worker_cls = \
                    "vllm_spyre.worker.spyre_worker.SpyreWorker"

        if envs.VLLM_USE_V1:
            # As of 0.7.3 the scheduler for V1 isn't actually pluggable like
            # this yet
            scheduler_config.scheduler_cls = \
                "vllm_spyre.v1.core.scheduler.SpyreScheduler"
        else:
            scheduler_config.scheduler_cls = \
                "vllm_spyre.core.scheduler.SpyreScheduler"

        # Override --max-num-seqs to the biggest warmup batch size
        # And override --max-model-len to the biggest warmup sequence
        cls.set_warmup_shapes(scheduler_config)
        max_batch_size = 0
        max_seq_len = 0
        for shape in scheduler_config.spyre_warmup_shapes:
            max_batch_size = max(max_batch_size, shape['batch_size'])
            max_seq_len = max(max_batch_size,
                              shape['prompt_length'] + shape['new_tokens'])

        if envs.VLLM_USE_V1:
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
    def set_warmup_shapes(cls, scheduler_config) -> None:
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

        logger.info("VLLM_SPYRE_WARMUP_PROMPT_LENS = %s", wup_prompt_lens)
        logger.info("VLLM_SPYRE_WARMUP_NEW_TOKENS = %s", wup_new_tokens)
        logger.info("VLLM_SPYRE_WARMUP_BATCH_SIZES = %s", wup_batch_sizes)

        scheduler_config.spyre_warmup_shapes = tuple(
            sorted([{
                'prompt_length': pl,
                'new_tokens': nt,
                'batch_size': bs
            } for pl, nt, bs in zip(wup_prompt_lens, wup_new_tokens,
                                    wup_batch_sizes)],
                   key=operator.itemgetter('batch_size', 'prompt_length')))

import operator
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
    spyre_warmup_shapes: tuple[dict[str, int], ...]

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

        print("\n\nCHECKING AND UPDATING CONFIG\n\n")

        parallel_config = vllm_config.parallel_config
        scheduler_config = vllm_config.scheduler_config

        # if scheduler_config.is_multi_step or envs.VLLM_USE_V1:
        #     raise NotImplementedError

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = \
                "vllm_spyre.worker.spyre_worker.SpyreWorker"
        
            if envs.VLLM_USE_V1:
                parallel_config.worker_cls = \
                    "vllm_spyre.v1.worker.spyre_worker.SpyreWorker"

        # scheduler_config.scheduler_cls = \
        #     "vllm_spyre.core.scheduler.SpyreScheduler"
        cache_config = vllm_config.cache_config
        if cache_config:
            # spyre needs block_size = max_model_len
            vllm_config.cache_config.block_size = \
                vllm_config.model_config.max_model_len
        cls.set_warmup_shapes(scheduler_config)

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

        print("\n\n Setting warmup shapes \n\n\n")

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

        print("[SchedulerConfig] VLLM_SPYRE_WARMUP_PROMPT_LENS =",
              wup_prompt_lens)
        print("[SchedulerConfig] VLLM_SPYRE_WARMUP_NEW_TOKENS =",
              wup_new_tokens)
        print("[SchedulerConfig] VLLM_SPYRE_WARMUP_BATCH_SIZES =",
              wup_batch_sizes)

        cls.spyre_warmup_shapes = tuple(
            sorted([{
                'prompt_length': pl,
                'new_tokens': nt,
                'batch_size': bs
            } for pl, nt, bs in zip(wup_prompt_lens, wup_new_tokens,
                                    wup_batch_sizes)],
                   key=operator.itemgetter('batch_size', 'prompt_length')))
        
        print("\n\n\n Set the warmup shapes \n\n\n")

    @classmethod
    def get_warmup_shapes(cls) -> tuple[dict[str, int], ...]:
        return cls.spyre_warmup_shapes

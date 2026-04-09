"""A Torch Spyre worker class."""

from vllm.config import VllmConfig
from vllm.v1.worker.cpu_worker import CPUWorker

from vllm_spyre_next.custom_ops import register_all


class TorchSpyreWorker(CPUWorker):
    """A worker class that executes the model on a group of Spyre cores."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(vllm_config, local_rank, rank, distributed_init_method, is_driver_worker)

        # Register all the custom ops here when a worker is created.
        # This has to happen before the model is loaded, so that all the layers will be swapped out
        # with the custom implementations for spyre.
        register_all()

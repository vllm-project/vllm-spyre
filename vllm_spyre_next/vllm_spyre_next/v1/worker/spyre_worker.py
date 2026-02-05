"""A Torch Spyre worker class."""

from vllm.v1.worker.worker_base import WorkerBase


class TorchSpyreWorker(WorkerBase):
    """A worker class that executes the model on a group of Spyre cores."""

    raise NotImplementedError

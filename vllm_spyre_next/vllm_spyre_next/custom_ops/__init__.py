"""This module contains all custom ops for spyre"""

from . import silu_and_mul
from . import vocab_parallel_embedding
from . import linear
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_all():
    logger.info("Registering custom ops for spyre_next")
    silu_and_mul.register()
    vocab_parallel_embedding.register()
    linear.register()

    # IR provider registration (triggered by import)
    from . import kernels as _kernels  # noqa: F401

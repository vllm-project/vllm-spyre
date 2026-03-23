"""This module contains all custom ops for spyre"""

from . import cpu_fallback
from . import linear
from . import rms_norm
from . import silu_and_mul
from . import vocab_parallel_embedding
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_all():
    logger.info("Registering custom ops for spyre_next")
    cpu_fallback.register()
    vocab_parallel_embedding.register()
    linear.register()
    rms_norm.register()
    silu_and_mul.register()

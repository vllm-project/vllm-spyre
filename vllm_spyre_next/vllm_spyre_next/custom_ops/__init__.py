"""This module contains all custom ops for spyre"""

from . import rms_norm
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_all():
    logger.info("Registering custom ops for spyre_next")
    rms_norm.register()

"""Custom ops and IR providers for Spyre."""

from . import silu_and_mul
from vllm.logger import init_logger

logger = init_logger(__name__)


def register_all():
    logger.info("Registering custom ops and IR providers for spyre_next")

    # IR provider registration (triggered by import)
    from . import kernels as _kernels  # noqa: F401

    # OOT class registration (triggered by import of @register_oot decorator)
    from . import rms_norm as _rms_norm  # noqa: F401

    # Legacy custom op registration (still needed for silu_and_mul)
    silu_and_mul.register()

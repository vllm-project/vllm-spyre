"""This module contains all custom ops for spyre"""

from . import cpu_fallback
from . import linear
from . import parallel_lm_head
from . import rms_norm
from . import rotary_embedding
from . import silu_and_mul
from . import vocab_parallel_embedding
from vllm.logger import init_logger
from vllm.utils.torch_utils import vllm_lib

logger = init_logger(__name__)

# All custom ops that need dual dispatch (CPU + PrivateUse1).
# Each entry is (op_name, op_func).
_DUAL_DISPATCH_OPS: list[tuple[str, object]] = []


def register_dual_dispatch(op_name: str, op_func: object) -> None:
    """Record an op for dual-key registration after initial register."""
    _DUAL_DISPATCH_OPS.append((op_name, op_func))


def _register_second_dispatch_key():
    """Register all custom ops for the alternate dispatch key.

    direct_register_custom_op registers ops for the platform's dispatch_key
    (currently "CPU"). Spyre also needs PrivateUse1 dispatch because integer
    tensors (input_ids, positions) live on the Spyre device. This adds
    the second key so ops dispatch correctly regardless of input device.
    """
    from vllm.platforms import current_platform
    primary_key = current_platform.dispatch_key

    # Add the other key
    other_key = "PrivateUse1" if primary_key == "CPU" else "CPU"
    for op_name, op_func in _DUAL_DISPATCH_OPS:
        vllm_lib.impl(op_name, op_func, dispatch_key=other_key)
    logger.info("Registered %d ops for secondary dispatch key: %s",
                len(_DUAL_DISPATCH_OPS), other_key)


def register_all():
    logger.info("Registering custom ops for spyre_next")
    cpu_fallback.register()
    vocab_parallel_embedding.register()
    parallel_lm_head.register()
    linear.register()
    rotary_embedding.register()
    rms_norm.register()
    silu_and_mul.register()
    _register_second_dispatch_key()

"""Environment variables for vllm_spyre_next configuration."""

import os
from typing import TYPE_CHECKING, Any, Callable

from vllm.logger import init_logger

if TYPE_CHECKING:
    VLLM_SPYRE_NEXT_RMSNORM_KERNEL: str = "native"
    VLLM_SPYRE_NEXT_RMSNORM_DTYPE_UP_PROMOTION: bool = False

logger = init_logger(__name__)

_cache: dict[str, Any] = {}


def override(name: str, value: str) -> None:
    """Override an environment variable value (for testing)."""
    if name not in environment_variables:
        raise ValueError(f"The variable {name} is not a known setting and cannot be overridden")
    os.environ[name] = value
    _cache[name] = environment_variables[name]()


def clear_env_cache():
    """Clear the environment variable cache."""
    _cache.clear()


environment_variables: dict[str, Callable[[], Any]] = {
    # RMS norm kernel implementation selection
    # Available options:
    # - "native": Transpose-based vLLM implementation (default, recommended)
    # - "functional": torch.nn.functional.rms_norm implementation
    "VLLM_SPYRE_NEXT_RMSNORM_KERNEL": lambda: os.getenv("VLLM_SPYRE_NEXT_RMSNORM_KERNEL", "native"),
    # Enable dtype up-promotion (f16->f32) for RMS norm computation
    # Note: Currently not fully supported by torch-spyre. Setting to 1
    # may cause compilation or runtime errors.
    # Set to 0 (default) to disable, 1 to enable.
    "VLLM_SPYRE_NEXT_RMSNORM_DTYPE_UP_PROMOTION": lambda: bool(
        int(os.getenv("VLLM_SPYRE_NEXT_RMSNORM_DTYPE_UP_PROMOTION", "0"))
    ),
}


def __getattr__(name: str):
    """Lazy evaluation and caching of environment variables."""
    if name in _cache:
        return _cache[name]

    if name in environment_variables:
        value = environment_variables[name]()
        _cache[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# Shared registry: layer_name -> layer instance (for custom op lookup)
_LAYER_REGISTRY: dict[str, Any] = {}
_INSTANCE_COUNTERS: dict[str, int] = {}


def register_layer(instance: Any, prefix: str) -> str:
    """Register a layer instance and return its unique name.

    Used by custom ops that need to look up `self` from a standalone
    function (the custom op runs outside torch.compile and receives
    only a string key).

    Args:
        instance: The layer instance to register.
        prefix: Base name, e.g. "spyre_rmsnorm".

    Returns:
        Unique layer name, e.g. "spyre_rmsnorm_0".
    """
    count = _INSTANCE_COUNTERS.get(prefix, 0)
    name = f"{prefix}_{count}"
    _INSTANCE_COUNTERS[prefix] = count + 1
    _LAYER_REGISTRY[name] = instance
    return name


def get_layer(name: str) -> Any:
    """Look up a registered layer by name."""
    return _LAYER_REGISTRY[name]


def _fake_impl(*args, **kwargs) -> None:
    """No-op fake implementation for shape inference during torch.compile tracing."""
    return


def convert(tensor, device=None, dtype=None):
    """Convert tensor device and/or dtype. No-op when both are None.

    Args:
        tensor: Input tensor, or None (passed through as None).
        device: Target device (None = keep current).
        dtype: Target dtype (None = keep current).

    Returns:
        Converted tensor, or None if input is None.
    """
    if tensor is None:
        return None
    # In case the tensor is on spyre and a dtype change is requested
    # we first need to move it to cpu and then change the dtype.
    if tensor.device.type == "spyre" and dtype is not None:
        tensor = tensor.to(device="cpu")
        # Set the device to spyre to avoid unexpected device change of the return tensor.
        if device is None:
            device = "spyre"

    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor

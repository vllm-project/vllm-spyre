"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

from typing import Any

import torch

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


def _unwrap_subclass(tensor: torch.Tensor) -> torch.Tensor:
    """Strip Python tensor subclass wrapper, returning the plain Spyre tensor.

    torch-spyre's DMA code (copy_device_to_host) does a static_cast to
    SpyreTensorImpl and reads dma_sizes/spyre_layout fields. Tensor
    subclasses created via _make_subclass share the same TensorImpl, BUT
    the copy_ dispatch goes through __torch_function__ which creates
    intermediate tensors with plain TensorImpl (no SpyreTensorImpl).
    This causes the DMA code to read garbage fields and crash.

    Stripping the subclass via as_subclass(torch.Tensor) ensures the
    copy dispatch sees the original SpyreTensorImpl.
    """
    if type(tensor) is not torch.Tensor:
        return tensor.as_subclass(torch.Tensor)
    return tensor


def _get_root_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Walk up the view chain to find the root non-view tensor.

    _make_subclass creates views, so SpyreSafeView wrappers appear as
    views even for full tensors. This walks ._base until we reach a
    non-view tensor that is safe for torch-spyre DMA.
    """
    t = _unwrap_subclass(tensor)
    while t._is_view() and t._base is not None:
        t = _unwrap_subclass(t._base)
    return t


def spyre_to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Safely copy a Spyre tensor (possibly a sliced view) to CPU.

    Handles torch-spyre bugs:
    1. .to("cpu") doesn't work on tensor subclasses (SpyreSafeView)
    2. D2H copy crashes on sliced/non-contiguous views
    3. D2H copy crashes on _make_subclass tensors (missing SpyreTensorImpl)

    Strategy: find the root non-view tensor, copy it to CPU in full,
    then reconstruct the original view (slice) on CPU.

    Remove this function once torch-spyre fixes these issues.
    """
    if tensor is None or tensor.device.type == "cpu":
        return tensor

    src = _unwrap_subclass(tensor)

    # If this is a view (from slicing, _make_subclass, etc.), we cannot
    # D2H copy it directly — torch-spyre crashes on views. Instead, copy
    # the root tensor to CPU and reconstruct the view there.
    if src._is_view() and src._base is not None:
        root = _get_root_tensor(src)
        root = root.contiguous()
        cpu_root = torch.empty(root.shape, dtype=root.dtype, device="cpu")
        cpu_root.copy_(root)
        # Reconstruct the view on CPU using storage_offset + as_strided
        offset = src.storage_offset()
        cpu_view = torch.as_strided(cpu_root, src.shape, src.stride(), offset)
        return cpu_view.contiguous()

    # Non-view tensor: direct copy
    src = src.contiguous()
    cpu_tensor = torch.empty(src.shape, dtype=src.dtype, device="cpu")
    cpu_tensor.copy_(src)
    return cpu_tensor


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
    if tensor.device.type == "spyre":
        # In case the tensor is on spyre, we first need to move it to cpu and then change the dtype.
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
    else:
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if device is not None:
            tensor = tensor.to(device=device)
    return tensor

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


# Library object for registering PrivateUse1 (Spyre) dispatch keys.
# vLLM's direct_register_custom_op only registers the platform dispatch_key
# (CPU). When tensors flow on Spyre, PyTorch dispatches to PrivateUse1 —
# so we must register the same kernels there too.
_spyre_dispatch_lib = torch.library.Library("vllm", "IMPL")


def register_spyre_dispatch(op_name: str, op_func) -> None:
    """Register an op implementation for the PrivateUse1 (Spyre) dispatch key.

    Call this after direct_register_custom_op to ensure the op can be
    dispatched when any argument is a Spyre tensor.
    """
    _spyre_dispatch_lib.impl(op_name, op_func, dispatch_key="PrivateUse1")


def _safe_spyre_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Safely copy a Spyre tensor (possibly a sliced view) to CPU.

    Handles torch-spyre bug where D2H copy crashes on sliced/non-contiguous
    views. Strategy: walk up the view chain to find the root non-view tensor,
    copy it to CPU in full, then reconstruct the original view (slice) on CPU.

    Remove this function once torch-spyre fixes view copy issues.
    """
    # If this is a view (from slicing), we cannot D2H copy it directly —
    # torch-spyre crashes on views. Copy the root tensor to CPU and
    # reconstruct the view there.
    if tensor._is_view() and tensor._base is not None:
        # Walk up view chain to find the root non-view tensor
        t = tensor
        while t._is_view() and t._base is not None:
            t = t._base
        t = t.contiguous()
        cpu_root = torch.empty(t.shape, dtype=t.dtype, device="cpu")
        cpu_root.copy_(t)
        # Reconstruct the view on CPU using storage_offset + as_strided
        offset = tensor.storage_offset()
        cpu_view = torch.as_strided(cpu_root, tensor.shape, tensor.stride(), offset)
        return cpu_view.contiguous()

    # Non-view tensor: direct copy
    src = tensor.contiguous()
    cpu_tensor = torch.empty(src.shape, dtype=src.dtype, device="cpu")
    cpu_tensor.copy_(src)
    return cpu_tensor


def convert(tensor, device=None, dtype=None):
    """Convert tensor device and/or dtype. No-op when both are None.

    When moving from Spyre to CPU, uses a safe D2H copy path that handles
    torch-spyre's bug where D2H crashes on sliced views. Dtype conversion
    is always done on CPU (torch-spyre doesn't support cross-dtype copy_).

    Args:
        tensor: Input tensor, or None (passed through as None).
        device: Target device (None = keep current). Can be a string,
            torch.device, or None.
        dtype: Target dtype (None = keep current).

    Returns:
        Converted tensor, or None if input is None.
    """
    if tensor is None:
        return None

    target = torch.device(device) if isinstance(device, str) else device

    if tensor.device.type == "spyre":
        # Spyre -> CPU: use safe D2H copy (handles view crash bug).
        # Spyre -> Spyre dtype change: move to CPU first (torch-spyre
        # doesn't support cross-dtype copy_).
        if target is not None and target.type != "spyre":
            tensor = _safe_spyre_to_cpu(tensor)
        elif target is None and dtype is not None:
            # dtype change only — must go through CPU
            tensor = _safe_spyre_to_cpu(tensor)
        elif target is not None and target.type == "spyre":
            pass  # already on spyre, no device move needed
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
    else:
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        if target is not None and tensor.device.type != target:
            tensor = tensor.to(device=target)
    return tensor

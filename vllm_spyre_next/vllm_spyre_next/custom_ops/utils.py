"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

from typing import Any

import torch
import torch.utils._pytree as pytree
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger

logger = init_logger(__name__)


def convert_for_spyre(*args, dtype=torch.float16):
    """Transfer tensors from CPU to Spyre device, potentially with dtype conversion.

    Args:
        *args: Variable number of arguments containing tensors or nested structures
               (lists, tuples, dicts) with tensors
        dtype: Target dtype for the tensors (default: torch.float16)

    Returns:
        Converted structure with all tensors on Spyre device and with potential dtype conversion

    Example:
        >>> x = torch.randn(10, 20)  # CPU tensor, any dtype
        >>> x_spyre = convert_for_spyre(x)
        >>> # x_spyre is now on Spyre device in float16
    """

    def _convert(arg):
        return (
            arg.to(dtype=dtype).to(device=torch.device("spyre"))
            if isinstance(arg, torch.Tensor)
            else arg
        )

    return pytree.tree_map(_convert, args)[0]


def convert_from_spyre(*args, dtype=torch.float16, device="cpu"):
    """Transfer tensors from Spyre to device, potentially with dtype conversion.

    Args:
        *args: Variable number of arguments containing tensors or nested structures
               (lists, tuples, dicts) with tensors
        dtype: Target dtype for the tensors (default: torch.float16)
        device: Target device for the tensors (default: "cpu")

    Returns:
        Converted structure with all tensors on Spyre device and with potential dtype conversion

    Example:
        >>> x = torch.randn(10, 20)  # CPU tensor, any dtype
        >>> x_spyre = convert_for_spyre(x)
        >>> # x_spyre is now on Spyre device in float16
    """

    def _convert(arg):
        return (
            arg.to(device=torch.device(device)).to(dtype=dtype)
            if isinstance(arg, torch.Tensor)
            else arg
        )

    return pytree.tree_map(_convert, args)[0]


def register_in_static_context(instance: Any, prefix_base: str) -> str:
    """Register layer in static_forward_context with unique prefix.

    Necessary for custom ops that bypass torch.compile and are retrieved via
    get_forward_context().no_compile_layers during forward passes.

    Args:
        instance: Layer instance to register
        prefix_base: Base name (e.g., "spyre_rmsnorm")

    Returns:
        Unique prefix assigned to this instance
    """
    compilation_config = get_current_vllm_config().compilation_config
    cls = instance.__class__

    if not hasattr(cls, "_instance_counter"):
        cls._instance_counter = 0

    prefix = f"{prefix_base}_{cls._instance_counter}"
    cls._instance_counter += 1

    if prefix in compilation_config.static_forward_context:
        raise ValueError(f"Duplicate layer name: {prefix}")

    compilation_config.static_forward_context[prefix] = instance
    return prefix


def dispatch_forward_impl(layer_name: str, *args, **kwargs):
    """Look up a custom op layer by name and call its forward_impl.

    This is the shared dispatch logic for all Spyre custom ops. Each op
    defines a typed op_func wrapper (required by torch's infer_schema)
    that delegates here.

    Args:
        layer_name: Key into forward_context.no_compile_layers
        *args, **kwargs: Forwarded to layer.forward_impl
    """
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(*args, **kwargs)


def fake_impl(*args, **kwargs) -> None:
    """Shared no-op fake implementation for all Spyre custom ops.

    Used for shape inference during torch.compile tracing. The op schema
    is already defined by the typed op_func, so this doesn't need typed
    parameters.
    """
    return

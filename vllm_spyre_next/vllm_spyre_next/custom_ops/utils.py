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


def create_rmsnorm_op_pair():
    """Create custom op pair for RMSNorm operation."""

    def op_func(
        x: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
        residual: torch.Tensor | None = None,
    ) -> None:
        forward_context = get_forward_context()
        layer = forward_context.no_compile_layers[layer_name]
        layer.forward_impl(x, output, residual)

    def fake_impl(
        x: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
        residual: torch.Tensor | None = None,
    ) -> None:
        return

    return op_func, fake_impl


def create_siluandmul_op_pair():
    """Create custom op pair for SiluAndMul operation."""

    def op_func(
        x: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
    ) -> None:
        forward_context = get_forward_context()
        layer = forward_context.no_compile_layers[layer_name]
        layer.forward_impl(x, output)

    def fake_impl(
        x: torch.Tensor,
        output: torch.Tensor,
        layer_name: str,
    ) -> None:
        return

    return op_func, fake_impl

"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

import torch
import torch.utils._pytree as pytree
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

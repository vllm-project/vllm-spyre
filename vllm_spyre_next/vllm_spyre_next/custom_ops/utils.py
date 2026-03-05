"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

import torch
import torch.utils._pytree as pytree
from vllm.logger import init_logger

logger = init_logger(__name__)


def convert_for_spyre(*args, **kwargs):
    """Transfer tensors from CPU to Spyre device, potentially with dtype conversion.

    Central utility for preparing inputs before Spyre kernel execution.
    Recursively processes nested data structures (lists, tuples, dicts) using
    PyTorch's pytree utilities to convert all tensors to Spyre device format.

    Conversion Process:
        1. Dtype conversion: Any dtype -> float16, or provided dtype (Spyre requirement)
        2. Device transfer: CPU/CUDA -> Spyre device
        3. Non-tensor passthrough: Other types remain unchanged

    Args:
        *args: Variable number of arguments containing tensors or nested structures
               (lists, tuples, dicts) with tensors

    Returns:
        Converted structure with all tensors on Spyre device and with potential dtype conversion

    Example:
        >>> x = torch.randn(10, 20)  # CPU tensor, any dtype
        >>> x_spyre = convert_for_spyre(x)
        >>> # x_spyre is now on Spyre device in float16

    Note:
        - Used by SpyreRMSNorm.forward_native and other Spyre custom ops
        - Output must be transferred back to CPU after computation
    """

    dtype = torch.float16
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
        del kwargs["dtype"]

    if len(kwargs) > 0:
        raise ValueError(
            f"`convert_for_spyre` only supports `dtype` kwarg, but these are provided in addition: {kwargs.keys()}"
        )

    def _convert(arg):
        return (
            arg.to(dtype=dtype).to(device=torch.device("spyre"))
            if isinstance(arg, torch.Tensor)
            else arg
        )

    return pytree.tree_map(_convert, args)[0]


def convert_from_spyre(*args, **kwargs):
    """Transfer tensors from Spyre to device, potentially with dtype conversion.

    Central utility for preparing inputs before Spyre kernel execution.
    Recursively processes nested data structures (lists, tuples, dicts) using
    PyTorch's pytree utilities to convert all tensors to Spyre device format.

    Conversion Process:
        1. Dtype conversion: Any dtype -> float16, or provided dtype (Spyre requirement)
        2. Device transfer: CPU/CUDA -> Spyre device
        3. Non-tensor passthrough: Other types remain unchanged

    Args:
        *args: Variable number of arguments containing tensors or nested structures
               (lists, tuples, dicts) with tensors

    Returns:
        Converted structure with all tensors on Spyre device and with potential dtype conversion

    Example:
        >>> x = torch.randn(10, 20)  # CPU tensor, any dtype
        >>> x_spyre = convert_for_spyre(x)
        >>> # x_spyre is now on Spyre device in float16

    Note:
        - Used by SpyreRMSNorm.forward_native and other Spyre custom ops
        - Output must be transferred back to CPU after computation
    """

    dtype = torch.float16
    if "dtype" in kwargs:
        dtype = kwargs["dtype"]
        del kwargs["dtype"]

    device = "cpu"
    if "device" in kwargs:
        device = kwargs["device"]
        del kwargs["device"]

    if len(kwargs) > 0:
        raise ValueError(
            f"`convert_for_spyre` only supports `dtype` and `device` kwargs, but these are provided in addition: {kwargs.keys()}"
        )

    def _convert(arg):
        return (
            arg.to(device=torch.device(device)).to(dtype=dtype)
            if isinstance(arg, torch.Tensor)
            else arg
        )

    return pytree.tree_map(_convert, args)[0]

"""Utility functions for Spyre custom operations.

This module provides helper functions for preparing tensors and data structures
for execution on IBM's Spyre device, primarily handling device transfer and
dtype conversion.
"""

import torch
import torch.utils._pytree as pytree
from vllm.logger import init_logger

logger = init_logger(__name__)


def prepare_inputs_on_spyre(*args):
    """Transfer tensors from CPU to Spyre device with dtype conversion.

    Central utility for preparing inputs before Spyre kernel execution.
    Recursively processes nested data structures (lists, tuples, dicts) using
    PyTorch's pytree utilities to convert all tensors to Spyre device format.

    Conversion Process:
        1. Dtype conversion: Any dtype -> float16 (Spyre requirement)
        2. Device transfer: CPU/CUDA -> Spyre device
        3. Non-tensor passthrough: Other types remain unchanged

    Args:
        *args: Variable number of arguments containing tensors or nested structures
               (lists, tuples, dicts) with tensors

    Returns:
        Converted structure with all tensors on Spyre device in float16 format.
        Returns first element of pytree.tree_map result (unwraps single-arg case).

    Example:
        >>> x = torch.randn(10, 20)  # CPU tensor, any dtype
        >>> x_spyre = prepare_inputs_on_spyre([x])[0]
        >>> # x_spyre is now on Spyre device in float16

    Note:
        - Always converts to float16 regardless of input dtype
        - Used by SpyreRMSNorm.forward_native and other Spyre custom ops
        - Output must be transferred back to CPU after computation
    """

    def _convert_to_spyre(arg):
        return (
            arg.to(dtype=torch.float16).to(device=torch.device("spyre"))
            if isinstance(arg, torch.Tensor)
            else arg
        )

    return pytree.tree_map(_convert_to_spyre, args)[0]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific SiluAndMul implementation using out-of-tree (OOT) registration.

This module provides a custom SiluAndMul (SwiGLU) activation layer for
IBM's Spyre device, replacing the upstream vLLM implementation from
vllm/model_executor/layers/activation.py when instantiated.

Architecture:
    - OOT Registration: @SiluAndMul.register_oot() replaces upstream at instantiation
    - forward_oot(): Entry point for OOT dispatch, fully transparent to the outer
      torch.compile graph (no opaque custom-op boundary).
    - Halves use .contiguous() to ensure zero storage offset before device transfer.
    - convert() utility avoids redundant data movement when tensor is already on
      the correct device/dtype (optimization from vllm-spyre#863).

Spyre Device Constraints:
    - Splitting (aten.slice.Tensor) inside compiled Spyre graphs is unsupported —
      non-zero storage offsets are rejected by the Flex backend.
    - The .contiguous() call ensures zero storage offset before transfer to Spyre.

Output Shape Note:
    input shape: [..., 2*d] -> output shape: [..., d]

References:
    - Upstream SiluAndMul: vllm/model_executor/layers/activation.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul

from .utils import convert

logger = init_logger(__name__)


@SiluAndMul.register_oot(name="SiluAndMul")
class SpyreSiluAndMul(SiluAndMul):
    """Out-of-tree (OOT) SiluAndMul implementation for IBM's Spyre device.

    Computes: x -> silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2

    Fully transparent to the outer torch.compile graph — no opaque custom-op
    boundary. Uses .contiguous() to ensure zero storage offset before device
    transfer, as Spyre's Flex backend rejects non-zero offsets
    (aten.slice.Tensor unsupported in compiled Spyre graphs).
    """

    _dynamic_arg_dims = {"x": []}

    def __init__(self, *args, **kwargs):
        """Initialize SpyreSiluAndMul layer.

        Sets up the target device (Spyre) and dtype (float16) for computation.
        The simplified implementation computes directly in forward_oot() without
        requiring layer registry or custom op registration.
        """
        super().__init__(*args, **kwargs)
        logger.debug("Building custom SiluAndMul")
        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        """Spyre-optimized SiLU and multiply activation (SwiGLU).

        Computes silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2.
        The input tensor is split into two halves, with .contiguous() ensuring
        zero storage offset (Spyre's Flex backend rejects non-zero offsets).

        The convert() utility handles device/dtype transfers efficiently,
        skipping data movement when the tensor is already on the correct
        device/dtype (vllm-spyre#863 optimization).

        Args:
            x: Input tensor of shape [..., 2*d] containing concatenated gate halves.

        Returns:
            Activated output tensor of shape [..., d] on the original device
            with the original dtype.
        """
        x_dtype = x.dtype
        x_device = x.device
        d = x.shape[-1] // 2
        # Call .contiguous() to ensure zero storage offset (Spyre's Flex backend
        # rejects non-zero offsets). convert() then transfers to Spyre device/dtype,
        # skipping transfer if already correct (vllm-spyre#863 optimization).
        x1 = convert(x[..., :d].contiguous(), self._target_device, self._target_dtype)
        x2 = convert(x[..., d:].contiguous(), self._target_device, self._target_dtype)
        return convert(F.silu(x1) * x2, x_device, x_dtype)


def register():
    """No-op: the custom-op barrier has been removed.

    Retained so register_all() in custom_ops/__init__.py needs no changes.
    """
    logger.debug("SpyreSiluAndMul: no custom op to register (barrier removed)")

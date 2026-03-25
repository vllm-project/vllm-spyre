# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre IR provider for rms_norm.

Registers a Spyre-optimized implementation of ir.ops.rms_norm via the
vLLM IR provider system. The provider contains only the mathematical
computation — no device transfers, no padding, no compilation management.

The implementation function (spyre_rms_norm_impl) is also exposed for
direct use with torch.compile in Phase 1 (eager mode with device sandwich),
where individual Spyre ops may not all work in eager mode.

Spyre workarounds applied:
    - Transpose trick: mean(dim=-1) not supported on Spyre, so we
      transpose(-1, -2) and use mean(dim=-2) instead.
    - Epsilon as tensor: scalar broadcast is limited on Spyre, so
      epsilon is expanded via torch.full().
    - No dtype promotion: torch-spyre does not yet support .to(float32),
      so computation stays in the input dtype (typically float16).
"""

import torch
from torch import Tensor

from vllm import ir


def _supports_spyre_args(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> bool:
    """Dynamic arg check: only handle tensors already on Spyre."""
    return x.device.type == "spyre"


def spyre_rms_norm_impl(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    """Spyre-optimized rms_norm implementation.

    Matches the ir.ops.rms_norm signature exactly (enforced by IrOpImpl).
    Assumes tensors are already on Spyre device in the correct dtype.

    This function is:
    - Registered as the "spyre" IR provider (for future Inductor lowering)
    - Used directly with torch.compile in Phase 1 (eager mode)
    """
    # Transpose: move hidden dim from last to second-to-last position
    # so that mean() operates on a supported dimension.
    x = x.transpose(-1, -2).contiguous()

    # Epsilon as full tensor (Spyre scalar broadcast limitation)
    eps_tensor = torch.full(
        x.shape, epsilon, dtype=x.dtype, device=x.device
    )

    # Variance computation (optionally on a prefix of hidden dim)
    if variance_size is None:
        x_var = x
    else:
        # After transpose(-1, -2), the hidden dimension is now at position -2.
        # Slice the first `variance_size` elements along that dimension.
        x_var = x[..., :variance_size, :]

    variance = x_var.pow(2).mean(dim=-2, keepdim=True)
    x = x * torch.rsqrt(variance + eps_tensor)

    # Transpose back to original layout
    x = x.transpose(-1, -2).contiguous()

    if weight is not None:
        x = x * weight
    return x


# Register with vLLM IR system
ir.ops.rms_norm.register_impl(
    "spyre",
    supports_args=_supports_spyre_args,
    supported=True,
)(spyre_rms_norm_impl)

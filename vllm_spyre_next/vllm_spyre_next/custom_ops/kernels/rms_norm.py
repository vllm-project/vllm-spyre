"""Spyre IR provider for rms_norm.

Spyre workarounds vs upstream native implementation:
    - Transpose trick: mean(dim=-1) not supported, use transpose + mean(dim=-2)
    - Epsilon as tensor: scalar broadcast limited, expand via torch.full()
    - No dtype promotion: torch-spyre limitation, stays in input dtype
"""

import torch
from torch import Tensor

from vllm import ir

_supports_spyre = lambda x, weight, epsilon, variance_size=None: (
    x.device.type == "spyre"
)


def spyre_rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    # x = x.transpose(-1, -2).contiguous()

    eps_tensor = torch.full(x.shape, epsilon, dtype=x.dtype, device=x.device)

    x_var = x if variance_size is None else x[..., :variance_size, :]
    # variance = x_var.pow(2).mean(dim=-2, keepdim=True)
    variance = x_var.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps_tensor)

    # x = x.transpose(-1, -2).contiguous()

    if weight is not None:
        x = x * weight
    return x


# Register with vLLM IR system (returns IrOpImpl, not the function)
ir.ops.rms_norm.register_impl(
    "spyre", supports_args=_supports_spyre, supported=True
)(spyre_rms_norm)

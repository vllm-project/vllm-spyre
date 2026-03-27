"""Spyre IR provider for rms_norm.

Spyre workarounds vs upstream native implementation:
    - Transpose trick: mean(dim=-1) not supported, use transpose + mean(dim=-2)
    - Epsilon as tensor: scalar broadcast limited, expand via torch.full()
    - No dtype promotion: torch-spyre limitation, stays in input dtype
"""

import torch
from torch import Tensor

from vllm import ir

_supports_spyre = lambda x, weight, epsilon, variance_size=None: (x.device.type == "spyre")


def spyre_rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
) -> Tensor:
    eps_tensor = torch.full(x.shape, epsilon, dtype=x.dtype, device=x.device)

    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps_tensor)

    if weight is not None:
        x = x * weight
    return x


# Register with vLLM IR system (returns IrOpImpl, not the function)
ir.ops.rms_norm.register_impl("spyre", supports_args=_supports_spyre, supported=True)(
    spyre_rms_norm
)

"""Spyre IR provider for rms_norm."""

import torch
from torch import Tensor

from vllm import ir

_supports_spyre = lambda x, weight, epsilon, variance_size=None: (x.device.type == "spyre")
"""Spyre provider handles tensors already on Spyre device"""


# Register with vLLM IR system (returns IrOpImpl, not the function)
@ir.ops.rms_norm.register_impl("spyre", supports_args=_supports_spyre, supported=True)
def spyre_rms_norm(
    x: Tensor,
    weight: Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> Tensor:
    """Spyre IR provider for rms_norm.

    Spyre-specific implementation details:
    - Epsilon as tensor: scalar broadcast limited, expand via torch.full()
    - No dtype promotion: torch-spyre limitation, stays in input dtype
    """
    eps_tensor = torch.full(x.shape, epsilon, dtype=x.dtype, device=x.device)

    # x_var = x if variance_size is None else x[..., :variance_size]
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps_tensor)

    if weight is not None:
        x = x * weight
    return x

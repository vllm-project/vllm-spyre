# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT RMSNorm — device sandwich around the vLLM IR rms_norm op.

Tensors arrive on CPU. This module:
1. Keeps a custom op boundary (forward_oot → torch.ops.vllm.spyre_rmsnorm) so
   model-level torch.compile does not trace into Spyre device transfers.
2. In _forward_spyre_impl(), converts tensors to Spyre and calls
   ir.ops.rms_norm() via direct dispatch. The IR system routes to the
   "spyre" provider (kernels/rms_norm.py) based on priority config.
"""

import torch

from functools import lru_cache

from vllm import ir
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)

_SPYRE_MIN_BATCH_SIZE = 64


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """OOT shim: device sandwich around ir.ops.rms_norm for Spyre."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        self._layer_name = register_layer(self, "spyre_rmsnorm")

        logger.warning_once(
            "SpyreRMSNorm: no dtype promotion (torch-spyre limitation), "
            "expect numerical differences to upstream vLLM."
        )

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Custom op boundary — opaque to model-level torch.compile."""
        output = torch.empty_like(x)
        torch.ops.vllm.spyre_rmsnorm(x, output, self._layer_name, residual)
        if residual is not None:
            return output, residual
        return output

    @staticmethod
    def forward_spyre(
        x: torch.Tensor,
        variance_epsilon: float,
        hidden_size: int,
        weight: torch.Tensor | None = None,
        residual: torch.Tensor | None = None,
        variance_size_override: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Spyre-optimized RMS norm using transpose-based computation (active implementation).

        Based on upstream vLLM's forward_static (vllm/model_executor/layers/layernorm.py)
        but adapted for Spyre device with transpose operations and torch.ops.spyre.full().
        Compiled separately via torch.compile in __init__.

        Key differences from upstream:
            - Uses transpose(-1, -2) for computation efficiency on Spyre
            - Creates epsilon tensor via torch.ops.spyre.full() instead of scalar
            - No dtype promotion support (torch-spyre limitation)
        """
        if residual is not None:
            x = x + residual
            residual = x

        if x.shape[-1] != hidden_size:
            raise ValueError(f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}")

        variance_epsilon = torch.full(
            x.shape, variance_epsilon, dtype=torch.float16, device=x.device
        )

        variance = x.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + variance_epsilon)

        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        else:
            return x, residual

    def _forward_spyre_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Device sandwich: CPU → Spyre → ir.ops.rms_norm (direct dispatch) → CPU."""
        # Handle residual on CPU (no need to transfer residual to Spyre)
        if residual is not None:
            x = x + residual
            residual = x

        x_dtype = x.dtype
        x_device = x.device
        orig_batch_size = x.shape[0]

        # Pad to Spyre minimum batch size
        if orig_batch_size < _SPYRE_MIN_BATCH_SIZE:
            pad = _SPYRE_MIN_BATCH_SIZE - orig_batch_size
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))

        # Transfer to Spyre, call IR op (direct dispatch → spyre provider), back
        result = ir.ops.rms_norm(
            convert(x, self._target_device, self._target_dtype),
            convert(self.weight.data, self._target_device, self._target_dtype)
            if self.has_weight
            else None,
            convert(residual, self._target_device, self._target_dtype),
        )
        result = convert(result, x_device, x_dtype)[:orig_batch_size, :]

        if residual is None:
            return result
        return result, residual


def _op_func(
    x: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    residual: torch.Tensor | None = None,
) -> None:
    """Custom op implementation — runs outside torch.compile graph."""
    layer = get_layer(layer_name)
    result = layer._forward_spyre_impl(x, residual)

    if residual is not None:
        output_data, residual_data = result
        output.copy_(output_data)
        residual.copy_(residual_data)
    else:
        output.copy_(result)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rmsnorm custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_rmsnorm",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreRMSNorm")

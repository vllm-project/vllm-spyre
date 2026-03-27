# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT RMSNorm — device sandwich around the Spyre IR provider.

Tensors arrive on CPU. This module:
1. Keeps a custom op boundary (forward_oot → torch.ops.vllm.spyre_rmsnorm) so
   model-level torch.compile does not trace into Spyre device transfers.
2. In _forward_spyre_impl(), converts tensors to Spyre and calls the compiled
   Spyre IR provider (kernels/rms_norm.py) which is also registered as the
   "spyre" implementation of ir.ops.rms_norm.

TODO: Once torch-spyre supports all ops in eager mode, remove maybe_compile
and call ir.ops.rms_norm() directly with direct_dispatch. This would let
the IR dispatch chain select the implementation at runtime, making the
explicit import of the provider function unnecessary.
"""

import torch

from functools import lru_cache

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)

_SPYRE_MIN_BATCH_SIZE = 64


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """OOT shim: device sandwich around the Spyre IR provider for rms_norm."""

    # Tensor args of the compiled function (spyre_rms_norm) — all static shapes
    _dynamic_arg_dims = {"x": [], "weight": []}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16

        # Compile the Spyre IR provider function. This is the same function
        # registered as the "spyre" impl of ir.ops.rms_norm.
        # TODO: Remove maybe_compile once torch-spyre supports all individual
        # ops eagerly; then ir.ops.rms_norm() with direct_dispatch suffices.
        from .kernels.rms_norm import spyre_rms_norm

        self._fwd = self.maybe_compile(spyre_rms_norm)

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

    def _forward_spyre_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Device sandwich: CPU → Spyre → compiled IR provider → CPU."""
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

        # Transfer to Spyre, call compiled IR provider, transfer back
        result = self._fwd(
            convert(x, self._target_device, self._target_dtype),
            convert(self.weight.data, self._target_device, self._target_dtype)
            if self.has_weight
            else None,
            self.variance_epsilon,
            self.variance_size_override,
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

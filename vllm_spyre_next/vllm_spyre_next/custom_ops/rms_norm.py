# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre OOT shim for RMSNorm (Phase 1 — device sandwich).

Tensors arrive on CPU. This module provides an OOT RMSNorm class that:
1. Transfers tensors to Spyre (float16)
2. Calls a pre-compiled version of the Spyre rms_norm kernel
3. Transfers results back to CPU in the original dtype

The Spyre-optimized math lives in kernels/rms_norm.py and is registered as
an IR provider. In Phase 1 (eager mode), we torch.compile it and call it
directly because Spyre's eager dispatch may not support all individual ops.
In Phase 2 (all tensors on Spyre + Inductor), the IR lowering pass will
trace the provider function and inline it into the compiled model graph.

Spyre Device Constraints:
    - Minimum batch size: 64 (padded automatically)
    - Device dtype: float16
    - No dtype promotion (torch-spyre limitation)
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

from .utils import convert

logger = init_logger(__name__)

_SPYRE_MIN_BATCH_SIZE = 64


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """Thin OOT shim: handles device sandwich, calls compiled Spyre kernel.

    The dispatch chain (with forward() NOT overridden) is:
        CustomOp.forward() → forward_oot() → forward_native() [this override]
    """

    _target_device = torch.device("spyre")
    _target_dtype = torch.float16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("SpyreRMSNorm: OOT shim active (Phase 1 — device sandwich)")

        # Pre-compile the Spyre kernel. The same function is registered as
        # the "spyre" IR provider in kernels/rms_norm.py.
        from .kernels.rms_norm import spyre_rms_norm_impl
        self._fwd = self.maybe_compile(spyre_rms_norm_impl)

        logger.warning(
            "SpyreRMSNorm: no dtype promotion is performed, "
            "expect numerical differences to upstream vLLM."
        )

    def forward_native(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Device-sandwich shim: CPU → Spyre → compiled kernel → CPU.

        For the non-residual path, transfers tensors to Spyre and calls
        the pre-compiled Spyre rms_norm kernel directly.
        For the residual path, delegates to upstream's forward_static().
        """
        if residual is not None:
            # Residual path: not yet IR-ified (waiting for IR 2/N).
            # Upstream's forward_static uses standard PyTorch ops.
            return self.forward_static(
                x,
                self.variance_epsilon,
                self.hidden_size,
                x.dtype,
                self.weight.data if self.has_weight else None,
                residual,
                self.variance_size_override,
            )

        # --- Non-residual path: device sandwich + compiled kernel ---
        x_dtype = x.dtype
        x_device = x.device
        orig_batch_size = x.shape[0]

        # Pad to minimum batch size (Spyre hardware constraint)
        if x.shape[0] < _SPYRE_MIN_BATCH_SIZE:
            pad_amount = _SPYRE_MIN_BATCH_SIZE - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, pad_amount))

        # Transfer to Spyre
        x_spyre = convert(x, self._target_device, self._target_dtype)
        weight_spyre = (
            convert(self.weight.data, self._target_device, self._target_dtype)
            if self.has_weight
            else None
        )

        # Call pre-compiled kernel (same function as IR provider)
        result = self._fwd(
            x_spyre,
            weight_spyre,
            self.variance_epsilon,
            self.variance_size_override,
        )

        # Transfer back to CPU, restore dtype, trim padding
        result = convert(result, device=x_device, dtype=x_dtype)
        return result[:orig_batch_size, :]

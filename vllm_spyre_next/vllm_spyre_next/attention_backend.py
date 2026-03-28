# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre attention backend — wraps CPU attention with device transfers.

The Spyre device cannot yet run attention kernels natively. This backend
reuses the CPU attention implementation (CPUAttentionBackend) and adds a
thin wrapper that moves q/k/v from Spyre to CPU before the computation
and copies the output back to Spyre afterward.

KV cache stays on CPU (allocated there by SpyreModelRunner), so no
transfer is needed for cached keys/values.

Remove this file once Spyre supports attention natively.
"""

from dataclasses import replace

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer

from vllm_spyre_next.custom_ops.utils import convert
from vllm.v1.attention.backends.cpu_attn import (
    CPUAttentionBackend,
    CPUAttentionBackendImpl,
    CPUAttentionMetadata,
)

logger = init_logger(__name__)


class SpyreCPUAttentionBackend(CPUAttentionBackend):
    """Attention backend for Spyre that delegates to CPU attention.

    Inherits everything from CPUAttentionBackend but overrides the
    impl class to add Spyre<->CPU device transfers.
    """

    @staticmethod
    def get_name() -> str:
        # Return "CPU_ATTN" so AttentionBackendEnum lookup succeeds.
        # The actual impl class is determined by get_impl_cls(), not the name.
        return "CPU_ATTN"

    @staticmethod
    def get_impl_cls() -> type["SpyreCPUAttentionBackendImpl"]:
        return SpyreCPUAttentionBackendImpl


class SpyreCPUAttentionBackendImpl(CPUAttentionBackendImpl):
    """CPU attention impl with Spyre<->CPU transfers.

    Intercepts forward() to:
    1. Move q/k/v from Spyre to CPU
    2. Run CPU attention (KV cache update + attention computation)
    3. Copy output back to Spyre
    """

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPUAttentionMetadata | None,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # For warmup (attn_metadata is None), just return output as-is
        if attn_metadata is None:
            return output

        # Move metadata tensors to CPU if on Spyre. The CPU attention
        # kernel expects all metadata (query_start_loc, seq_lens,
        # block_table, slot_mapping) on CPU.
        attn_metadata = self._metadata_to_cpu(attn_metadata)

        # Move q/k/v from Spyre to CPU (synchronous — Spyre doesn't
        # support non_blocking). KV cache is already on CPU.
        cpu_query = self._to_cpu(query)
        cpu_key = self._to_cpu(key)
        cpu_value = self._to_cpu(value)

        # Create CPU output buffer for the CPU attention kernel
        spyre_output = output
        cpu_output = (
            torch.empty_like(output, device="cpu")
            if output is not None and output.device.type != "cpu"
            else output
        )

        # Run CPU attention (includes KV cache update)
        result = super().forward(
            layer,
            cpu_query,
            cpu_key,
            cpu_value,
            kv_cache,
            attn_metadata,
            output=cpu_output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

        # Copy output back to Spyre
        if spyre_output is not None and spyre_output.device.type != "cpu":
            spyre_output.copy_(cpu_output)
            return spyre_output

        return result

    @staticmethod
    def _to_cpu(t: torch.Tensor | None) -> torch.Tensor | None:
        """Safely move a tensor to CPU.

        Delegates to convert() which handles torch-spyre's D2H copy
        crash on sliced views by copying the root tensor and
        reconstructing the view on CPU.
        """
        return convert(t, device="cpu")

    def _metadata_to_cpu(
        self, meta: CPUAttentionMetadata
    ) -> CPUAttentionMetadata:
        """Move metadata tensor fields to CPU if on Spyre.

        The CPU attention kernel expects all tensor metadata on CPU.
        Uses dataclasses.replace to create a new metadata with CPU tensors,
        preserving all non-tensor fields.
        """
        # Fast path: if query_start_loc is already on CPU, assume all are
        if meta.query_start_loc.device.type == "cpu":
            return meta

        updates: dict = {
            "query_start_loc": self._to_cpu(meta.query_start_loc),
            "seq_lens": self._to_cpu(meta.seq_lens),
            "block_table": self._to_cpu(meta.block_table),
            "slot_mapping": self._to_cpu(meta.slot_mapping),
            "scheduler_metadata": self._to_cpu(meta.scheduler_metadata),
        }

        # Handle optional SDPA fields
        if meta.sdpa_start_loc is not None:
            updates["sdpa_start_loc"] = self._to_cpu(meta.sdpa_start_loc)
        if meta.sdpa_attn_masks is not None:
            updates["sdpa_attn_masks"] = [
                self._to_cpu(m) for m in meta.sdpa_attn_masks
            ]

        return replace(meta, **updates)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure PyTorch implementation of PagedAttention.

This backend aims to implement PagedAttention using only PyTorch native operations,
such as matmul, softmax, etc. It supports vLLM's KV cache.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class SpyreAttentionPagedMetadata:
    """Metadata for PyTorch native attention computation."""

    # Batch information
    num_actual_tokens: int
    num_seqs: int
    max_query_len: int
    max_seq_len: int

    # Sequence lengths
    seq_lens: torch.Tensor  # [num_seqs]
    query_start_loc: torch.Tensor  # [num_seqs + 1]

    # Block table for paged KV cache
    block_table: torch.Tensor  # [num_seqs, max_num_blocks_per_seq]
    block_size: int

    # Slot mapping for KV cache updates
    slot_mapping: torch.Tensor  # [num_actual_tokens]

    # Whether causal masking is needed (True when max_query_len > 1)
    causal_mask: torch.Tensor | None = None

    # For grouped-query attention
    num_kv_heads: int = 0
    num_heads: int = 0


class SpyreAttentionPagedMetadataBuilder(AttentionMetadataBuilder[SpyreAttentionPagedMetadata]):
    """Builds attention metadata from batch information."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        self.num_heads = model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreAttentionPagedMetadata:
        """Build attention metadata from common metadata."""

        # Extract information from common metadata
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_seqs = common_attn_metadata.num_reqs
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len

        seq_lens = common_attn_metadata.seq_lens
        query_start_loc = common_attn_metadata.query_start_loc
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        # Create causal mask if needed
        causal_mask = None
        if common_attn_metadata.causal and max_query_len > 1:
            causal_mask = self._create_causal_mask(max_query_len, max_seq_len, self.device)

        return SpyreAttentionPagedMetadata(
            num_actual_tokens=num_actual_tokens,
            num_seqs=num_seqs,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            query_start_loc=query_start_loc,
            block_table=block_table,
            block_size=self.block_size,
            slot_mapping=slot_mapping,
            causal_mask=causal_mask,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
        )

    def _create_causal_mask(
        self,
        query_len: int,
        kv_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create causal attention mask."""
        # Create indices
        query_idx = torch.arange(query_len, device=device).unsqueeze(1)
        kv_idx = torch.arange(kv_len, device=device).unsqueeze(0)

        # Causal mask: query position can only attend to kv positions <= query position
        mask = query_idx < kv_idx

        return mask


class SpyreAttentionPagedBackend(AttentionBackend):
    """Pure PyTorch implementation of PagedAttention."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # Support any block size (no kernel-specific constraints)
        return [MultipleOf(1)]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SpyreAttentionPagedImpl"]:
        return SpyreAttentionPagedImpl

    @staticmethod
    def get_builder_cls() -> type["SpyreAttentionPagedMetadataBuilder"]:
        return SpyreAttentionPagedMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """KV cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]"""
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # Support any head size
        return True

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes


class SpyreAttentionPagedImpl(AttentionImpl[SpyreAttentionPagedMetadata]):
    """PyTorch native implementation of attention with paged KV cache."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        use_sdpa: bool = False,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        working_precision=None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type

        if working_precision is not None:
            self.working_precision = working_precision
        else:
            self.working_precision = torch.bfloat16

        # When True, use torch.nn.functional.scaled_dot_product_attention
        # Otherwise, use implementation with native PyTorch ops 
        self.use_sdpa = use_sdpa

        # Simplified implementation: don't support these features initially
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi slopes not supported yet")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window not supported yet")
        if logits_soft_cap is not None:
            raise NotImplementedError("Logits soft cap not supported yet")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,  # [num_tokens, num_heads, head_size]
        key: torch.Tensor,    # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,  # [2, num_blocks, block_size, num_kv_heads, head_size]
        attn_metadata: SpyreAttentionPagedMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute attention output using PyTorch native operations."""

        assert output is not None, "Output tensor must be provided"

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Step 1: Update KV cache
        self._write_to_kv_cache(
            key[:num_actual_tokens],
            value[:num_actual_tokens],
            kv_cache,
            attn_metadata.slot_mapping,
            attn_metadata.block_size,
        )

        # Step 2: Gather compact KV cache
        # compact_k/v: [num_seqs, max_seq_len, num_kv_heads, head_size]
        compact_k, compact_v = self._gather_compact_kv_cache(
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.block_size,
        )

        # Step 3: Reshape query to per-sequence format
        # query_per_seq: [num_seqs, max_query_len, num_heads, head_size]
        query_per_seq = self._reshape_query_to_sequences(
            query[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
            attn_metadata.max_query_len,
        )

        # Step 4: Build per-sequence attention mask
        # mask: [num_seqs, 1, max_query_len, max_seq_len]  (True = masked out)
        mask = self._build_attention_mask(
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            attn_metadata.causal_mask,
            compact_k.device,
        )

        # Step 5: Compute batched per-sequence attention
        # attn_output: [num_seqs, max_query_len, num_heads, head_size]
        attn_output = self._compute_attention(query_per_seq, compact_k, compact_v, mask)

        # Step 6: Extract only the actual query tokens (strip padding)
        # [num_actual_tokens, num_heads, head_size]
        attn_output_flat = self._extract_relevant_output(attn_output, attn_metadata.query_start_loc)

        output[:num_actual_tokens].copy_(attn_output_flat)
        return output

    def _write_to_kv_cache(
        self,
        key: torch.Tensor,    # [num_tokens, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,  # [num_tokens]
        block_size: int,
    ) -> None:
        """Write keys and values to paged KV cache using vectorized scatter."""
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        kv_cache[0][block_indices, block_offsets] = key
        kv_cache[1][block_indices, block_offsets] = value

    def _gather_compact_kv_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather only the relevant KV cache entries into compact tensors.

        Args:
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            block_table: [num_seqs, max_num_blocks_per_seq]
            seq_lens: [num_seqs]
            block_size: int

        Returns:
            compact_k: [num_seqs, max_seq_len, num_kv_heads, head_size]
            compact_v: [num_seqs, max_seq_len, num_kv_heads, head_size]
        """
        num_seqs = block_table.shape[0]
        max_blocks_per_seq = block_table.shape[1]
        max_seq_len = seq_lens.max()
        device = kv_cache.device

        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = kv_cache[1]

        # [num_seqs, max_seq_len]
        position_indices = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(num_seqs, -1)

        block_indices = position_indices // block_size
        offset_in_block = position_indices % block_size

        # Clamp to valid range; out-of-range positions are zeroed by valid_mask
        block_indices_clamped = torch.clamp(block_indices, 0, max_blocks_per_seq - 1)
        physical_blocks = block_table.gather(1, block_indices_clamped)

        # Zero out physical blocks for padding positions to avoid stale reads
        valid_mask = position_indices < seq_lens.unsqueeze(1)  # [num_seqs, max_seq_len]
        physical_blocks = physical_blocks * valid_mask

        # Gather: [num_seqs * max_seq_len, num_kv_heads, head_size]
        flat_blocks = physical_blocks.reshape(-1)
        flat_offsets = offset_in_block.reshape(-1)
        gathered_k = key_cache[flat_blocks, flat_offsets]
        gathered_v = value_cache[flat_blocks, flat_offsets]

        # Reshape to [num_seqs, max_seq_len, num_kv_heads, head_size]
        shape = (num_seqs, max_seq_len, key_cache.shape[2], key_cache.shape[3])
        compact_k = gathered_k.reshape(shape)
        compact_v = gathered_v.reshape(shape)

        # No need to zero padding positions: the attention mask already sets
        # scores for those positions to -inf, so their values are never used.

        return compact_k, compact_v

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,       # [num_seqs]
        query_start_loc: torch.Tensor, # [num_seqs + 1]
        causal_mask: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a per-sequence attention mask.

        Returns:
            mask: [num_seqs, 1, max_query_len, max_seq_len]
                  True = masked out (don't attend), False = attend
        """
        num_seqs = seq_lens.shape[0]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]  # [num_seqs]
        max_query_len = query_lens.max()
        max_seq_len = seq_lens.max()

        # Positions along query and KV dimensions
        q_pos = torch.arange(max_query_len, device=device)   # [max_query_len]
        kv_pos = torch.arange(max_seq_len, device=device)    # [max_seq_len]

        # Validity: which (seq, q, kv) positions are real (not padding)?
        # [num_seqs, max_query_len]
        q_valid = q_pos.unsqueeze(0) < query_lens.unsqueeze(1)
        # [num_seqs, max_seq_len]
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)

        # [num_seqs, max_query_len, max_seq_len]
        attend = q_valid.unsqueeze(2) & kv_valid.unsqueeze(1)

        if causal_mask is not None:
            # context_len[s] = seq_len[s] - query_len[s]
            # query token q_i (0-indexed) can attend to KV positions 0 .. context_len + q_i
            context_lens = seq_lens - query_lens  # [num_seqs]
            # [num_seqs, max_query_len, 1]
            causal_limit = (context_lens.unsqueeze(1) + q_pos.unsqueeze(0)).unsqueeze(2)
            # [num_seqs, 1, max_seq_len]
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)
            causal_ok = kv_pos_exp <= causal_limit  # [num_seqs, max_query_len, max_seq_len]
            attend = attend & causal_ok

        # [num_seqs, 1, max_query_len, max_seq_len]  True = masked out
        return ~attend.unsqueeze(1)

    def _reshape_query_to_sequences(
        self,
        query: torch.Tensor,           # [num_actual_tokens, num_heads, head_size]
        query_start_loc: torch.Tensor, # [num_seqs + 1]
        num_seqs: int,
        max_query_len: int,
    ) -> torch.Tensor:
        """
        Reshape flat query tokens into a padded per-sequence tensor.

        Returns:
            [num_seqs, max_query_len, num_heads, head_size]
        """
        num_heads = query.shape[1]
        head_size = query.shape[2]
        device = query.device

        query_lens = query_start_loc[1:] - query_start_loc[:-1]  # [num_seqs]

        # [num_seqs, max_query_len]
        positions = torch.arange(max_query_len, device=device).unsqueeze(0).expand(num_seqs, -1)
        global_indices = query_start_loc[:-1].unsqueeze(1) + positions

        # Clamp so gather doesn't go OOB; invalid positions are masked in attention
        global_indices_clamped = torch.clamp(global_indices, 0, query.shape[0] - 1)

        # [num_seqs, max_query_len, num_heads, head_size]
        query_per_seq = query[global_indices_clamped]

        # Zero out padding positions
        valid_mask = positions < query_lens.unsqueeze(1)
        query_per_seq = query_per_seq * valid_mask.unsqueeze(-1).unsqueeze(-1)

        return query_per_seq

    def _compute_attention(
        self,
        query: torch.Tensor,  # [num_seqs, max_query_len, num_heads, head_size]
        key: torch.Tensor,    # [num_seqs, max_seq_len, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_seqs, max_seq_len, num_kv_heads, head_size]
        mask: torch.Tensor,   # [num_seqs, 1, max_query_len, max_seq_len]  True=masked
    ) -> torch.Tensor:
        """
        Compute batched per-sequence attention with GQA via reshape+broadcast.

        Avoids repeat_interleave to expand KV heads (profiling showed it costs
        ~17 ms/layer vs ~0.18 ms for the actual matmuls on 30 seqs × 300 tok).
        Instead, Q is reshaped to expose the GQA grouping so that K/V can be
        broadcast over the query-group dimension without materialising the
        expanded tensors:

            Q  [B, Hkv, Gq, q_len, D]  ×  K  [B, Hkv,  1, kv_len, D]^T
            -> scores [B, Hkv, Gq, q_len, kv_len]

        All operations are pure torch.matmul / torch.softmax / masked_fill.

        # Alternative (requires PyTorch ≥ 2.5):
        # out = torch.nn.functional.scaled_dot_product_attention(
        #     query.transpose(1, 2),   # [B, H,   q_len,  D]
        #     key.transpose(1, 2),     # [B, Hkv, kv_len, D]
        #     value.transpose(1, 2),
        #     attn_mask=~mask,         # bool: True = attend
        #     scale=self.scale,
        #     enable_gqa=True,
        # ).transpose(1, 2)

        Returns:
            [num_seqs, max_query_len, num_heads, head_size]
        """
        B      = query.shape[0]
        q_len  = query.shape[1]
        kv_len = key.shape[1]
        Hkv    = self.num_kv_heads
        Gq     = self.num_queries_per_kv  # Q heads per KV head
        D      = self.head_size

        if self.use_sdpa:
            # torch.nn.functional.scaled_dot_product_attention path.
            # Requires PyTorch ≥ 2.5 for enable_gqa support.
            # Q:   [B, q_len,  H,   D] → [B, H,   q_len,  D]
            # K/V: [B, kv_len, Hkv, D] → [B, Hkv, kv_len, D]
            # attn_mask: [B, 1, q_len, kv_len]  True=attend (invert our mask)
            out = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),    # [B, H,   q_len,  D]
                key.transpose(1, 2),      # [B, Hkv, kv_len, D]
                value.transpose(1, 2),    # [B, Hkv, kv_len, D]
                attn_mask=~mask,          # bool: True = attend
                scale=self.scale,
                enable_gqa=True,
            )
            # [B, H, q_len, D] → [B, q_len, H, D]
            return out.transpose(1, 2)

        # --- Manual matmul/softmax path ---

        # Q: [B, H, q_len, D] → [B, Hkv, Gq, q_len, D]
        q = query.transpose(1, 2).reshape(B, Hkv, Gq, q_len, D)

        # K/V: [B, Hkv, kv_len, D] → unsqueeze Gq dim for broadcast
        # Shape: [B, Hkv, 1, kv_len, D]  (never materialised as Gq copies)
        k = key.transpose(1, 2).unsqueeze(2)    # [B, Hkv, 1, kv_len, D]
        v = value.transpose(1, 2).unsqueeze(2)  # [B, Hkv, 1, kv_len, D]

        # Scores: [B, Hkv, Gq, q_len, kv_len]
        scores = torch.matmul(
            q.to(self.working_precision),
            k.to(self.working_precision).transpose(-2, -1),
        ) * self.scale

        # mask [B, 1, q_len, kv_len] → unsqueeze to [B, 1, 1, q_len, kv_len]
        # for broadcast over Hkv and Gq
        scores = scores.masked_fill(mask.unsqueeze(1), -float("inf"))

        weights = torch.softmax(scores, dim=-1)

        # Padding query rows (q >= query_len for each seq) get all-masked scores → softmax NaN.
        # On some BLAS/CPU backends, NaN in padding rows can contaminate valid rows in
        # the following matmul.  Zero them out before the matmul; the padding output is
        # discarded by _extract_relevant_output anyway.
        weights = weights.nan_to_num(nan=0.0)

        # Output: [B, Hkv, Gq, q_len, D] → [B, H, q_len, D]
        out = torch.matmul(weights, v.to(self.working_precision))
        out = out.to(query.dtype).reshape(B, Hkv * Gq, q_len, D)

        # [B, q_len, H, D]
        return out.transpose(1, 2)

    def _extract_relevant_output(
        self,
        attn_output: torch.Tensor,     # [num_seqs, max_query_len, num_heads, head_size]
        query_start_loc: torch.Tensor, # [num_seqs + 1]
    ) -> torch.Tensor:
        """
        Extract actual query tokens from padded per-sequence output.

        Returns:
            [num_actual_tokens, num_heads, head_size]
        """
        max_query_len = attn_output.shape[1]
        device = attn_output.device

        query_lens = query_start_loc[1:] - query_start_loc[:-1]  # [num_seqs]

        # Boolean index into [num_seqs, max_query_len]
        positions = torch.arange(max_query_len, device=device).unsqueeze(0)
        valid = positions < query_lens.unsqueeze(1)  # [num_seqs, max_query_len]

        # Boolean indexing flattens the first two dims and keeps the rest
        return attn_output[valid]  # [num_actual_tokens, num_heads, head_size]

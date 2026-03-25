# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contiguous KV-cache implementation of AttentionBackend using torch-spyre.

This backend aims to implement attention using only PyTorch native operations,
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
    apply_causal_mask: bool = False

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
            apply_causal_mask=common_attn_metadata.causal and max_query_len > 1,
            num_kv_heads=self.num_kv_heads,
            num_heads=self.num_heads,
        )


class SpyreAttentionPagedBackend(AttentionBackend):
    """Pure PyTorch implementation of PagedAttention."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "float16",
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

    # KV length alignment: KV tensors are padded to the next multiple of this value.
    # Because torch.compile treats shapes as static constants, every distinct kv_len
    # triggers a full recompile. Aligning to 256 buckets sequence lengths into tiers
    # (256, 512, 768, ...) so only the first request at each tier pays compilation cost,
    # rather than recompiling on every decode step.
    KV_LENGTH_ALIGNMENT = 256

    # Query chunk size for padding - ensures consistent tensor sizes for Spyre compilation
    QUERY_CHUNK_SIZE = 32

    @staticmethod
    def _attn_transposed(qt, k, vt, sm_scale, mask_values):
        """Transposed attention for Spyre: handles all heads at once.

        Args:
            qt: Query transposed [head_size, num_heads * query_len_padded]
            k: Key [num_heads * kv_len, head_size]
            vt: Value transposed [head_size, num_heads * kv_len]
            sm_scale: Scale factor (1D tensor) [num_heads * query_len_padded]
            mask_values: Mask values tensor [num_heads * kv_len, num_heads * query_len_padded]
                        Pre-computed on CPU: 0.0 for valid, -65504.0 for masked/padded
        """
        kq = k @ qt  # [num_heads * kv_len, num_heads * query_len_padded]
        kq = kq * sm_scale

        # Add pre-computed mask values
        # Valid positions have 0.0, masked/padded positions have -65504.0
        kq = kq + mask_values

        p = kq.softmax(dim=0)
        return vt @ p  # [head_size, num_heads * query_len_padded]

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        working_precision=None,
        use_sdpa: bool = False,
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

        # Compile the attention function once for reuse
        self.attn_transposed_compiled = torch.compile(self._attn_transposed, dynamic=False)

        # When True, use torch.nn.functional.scaled_dot_product_attention
        # Otherwise, use Spyre implementation with compiled kernel
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
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
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
            attn_metadata.max_seq_len,
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
            attn_metadata.apply_causal_mask,
            compact_k.device,
            attn_metadata.max_seq_len,
            attn_metadata.max_query_len,
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
        key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
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
        max_seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather only the relevant KV cache entries into compact tensors with alignment.

        Args:
            kv_cache: [2, num_blocks, block_size, num_kv_heads, head_size]
            block_table: [num_seqs, max_num_blocks_per_seq]
            seq_lens: [num_seqs]
            block_size: int
            max_seq_len: pre-computed max of seq_lens (avoids a device sync)

        Returns:
            compact_k: [num_seqs, aligned_max_seq_len, num_kv_heads, head_size]
            compact_v: [num_seqs, aligned_max_seq_len, num_kv_heads, head_size]

        Note: aligned_max_seq_len is max_seq_len rounded up to KV_LENGTH_ALIGNMENT
        """
        num_seqs = block_table.shape[0]
        max_blocks_per_seq = block_table.shape[1]

        # Align max_seq_len to KV_LENGTH_ALIGNMENT
        aligned_max_seq_len = (
            (max_seq_len + self.KV_LENGTH_ALIGNMENT - 1)
            // self.KV_LENGTH_ALIGNMENT
            * self.KV_LENGTH_ALIGNMENT
        )

        device = kv_cache.device

        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = kv_cache[1]
        num_kv_heads = key_cache.shape[2]
        head_size = key_cache.shape[3]

        # [num_seqs, max_seq_len] - only gather up to actual max_seq_len
        position_indices = (
            torch.arange(max_seq_len, device=device).unsqueeze(0).expand(num_seqs, -1)
        )

        block_indices = position_indices // block_size
        offset_in_block = position_indices % block_size

        # Clamp to valid range
        block_indices_clamped = torch.clamp(block_indices, 0, max_blocks_per_seq - 1)
        physical_blocks = block_table.gather(1, block_indices_clamped)

        # Zero out physical blocks for padding positions
        valid_mask = position_indices < seq_lens.unsqueeze(1)  # [num_seqs, max_seq_len]
        physical_blocks = physical_blocks * valid_mask

        # Gather: [num_seqs * max_seq_len, num_kv_heads, head_size]
        flat_blocks = physical_blocks.reshape(-1)
        flat_offsets = offset_in_block.reshape(-1)
        gathered_k = key_cache[flat_blocks, flat_offsets]
        gathered_v = value_cache[flat_blocks, flat_offsets]

        # Reshape to [num_seqs, max_seq_len, num_kv_heads, head_size]
        gathered_k = gathered_k.reshape(num_seqs, max_seq_len, num_kv_heads, head_size)
        gathered_v = gathered_v.reshape(num_seqs, max_seq_len, num_kv_heads, head_size)

        # Pad to aligned length if needed
        if aligned_max_seq_len > max_seq_len:
            padding_size = aligned_max_seq_len - max_seq_len
            gathered_k = torch.nn.functional.pad(
                gathered_k,
                (0, 0, 0, 0, 0, padding_size),  # pad seq_len dimension
                mode="constant",
                value=0.0,
            )
            gathered_v = torch.nn.functional.pad(
                gathered_v, (0, 0, 0, 0, 0, padding_size), mode="constant", value=0.0
            )

        return gathered_k, gathered_v

    def _build_attention_mask(
        self,
        seq_lens: torch.Tensor,  # [num_seqs]
        query_start_loc: torch.Tensor,  # [num_seqs + 1]
        apply_causal_mask: bool,
        device: torch.device,
        max_seq_len: int,
        max_query_len: int,
    ) -> torch.Tensor:
        """
        Build a per-sequence attention mask with aligned KV length.

        Args:
            max_seq_len: pre-computed max of seq_lens (avoids a device sync)
            max_query_len: pre-computed max of query_lens (avoids a device sync)

        Returns:
            mask: [num_seqs, 1, max_query_len, aligned_max_seq_len]
                  True = masked out (don't attend), False = attend
        """
        query_lens = query_start_loc[1:] - query_start_loc[:-1]  # [num_seqs]

        # Align max_seq_len to KV_LENGTH_ALIGNMENT
        aligned_max_seq_len = (
            (max_seq_len + self.KV_LENGTH_ALIGNMENT - 1)
            // self.KV_LENGTH_ALIGNMENT
            * self.KV_LENGTH_ALIGNMENT
        )

        # Positions along query and KV dimensions
        q_pos = torch.arange(max_query_len, device=device)  # [max_query_len]
        kv_pos = torch.arange(aligned_max_seq_len, device=device)  # [aligned_max_seq_len]

        # Validity: which (seq, q, kv) positions are real (not padding)?
        # [num_seqs, max_query_len]
        q_valid = q_pos.unsqueeze(0) < query_lens.unsqueeze(1)
        # [num_seqs, aligned_max_seq_len] - only positions < seq_len are valid
        kv_valid = kv_pos.unsqueeze(0) < seq_lens.unsqueeze(1)

        # [num_seqs, max_query_len, aligned_max_seq_len]
        attend = q_valid.unsqueeze(2) & kv_valid.unsqueeze(1)

        if apply_causal_mask:
            # context_len[s] = seq_len[s] - query_len[s]
            # query token q_i (0-indexed) can attend to KV positions 0 .. context_len + q_i
            context_lens = seq_lens - query_lens  # [num_seqs]
            # [num_seqs, max_query_len, 1]
            causal_limit = (context_lens.unsqueeze(1) + q_pos.unsqueeze(0)).unsqueeze(2)
            # [num_seqs, 1, aligned_max_seq_len]
            kv_pos_exp = kv_pos.unsqueeze(0).unsqueeze(0)
            causal_ok = kv_pos_exp <= causal_limit  # [num_seqs, max_query_len, aligned_max_seq_len]
            attend = attend & causal_ok

        # [num_seqs, 1, max_query_len, aligned_max_seq_len]  True = masked out
        return ~attend.unsqueeze(1)

    def _reshape_query_to_sequences(
        self,
        query: torch.Tensor,  # [num_actual_tokens, num_heads, head_size]
        query_start_loc: torch.Tensor,  # [num_seqs + 1]
        num_seqs: int,
        max_query_len: int,
    ) -> torch.Tensor:
        """
        Reshape flat query tokens into a padded per-sequence tensor.

        Returns:
            [num_seqs, max_query_len, num_heads, head_size]
        """
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
        key: torch.Tensor,  # [num_seqs, max_seq_len, num_kv_heads, head_size]
        value: torch.Tensor,  # [num_seqs, max_seq_len, num_kv_heads, head_size]
        mask: torch.Tensor,  # [num_seqs, 1, max_query_len, max_seq_len]  True=masked
    ) -> torch.Tensor:
        """
        Compute attention using Spyre compiled operations.

        Processes each sequence separately with query chunking for Spyre compatibility.

        Returns:
            [num_seqs, max_query_len, num_heads, head_size]
        """
        num_seqs = query.shape[0]

        # As fallback, use SDPA implementation
        if self.use_sdpa:
            return self._compute_attention_sdpa(query, key, value, mask)

        # Allocate output tensor for all sequences
        output_all_seqs = torch.zeros_like(query)

        # Process each sequence separately
        for seq_idx in range(num_seqs):
            # Extract single sequence
            query_seq = query[seq_idx : seq_idx + 1]  # [1, max_query_len, num_heads, head_size]
            key_seq = key[seq_idx : seq_idx + 1]  # [1, max_seq_len, num_kv_heads, head_size]
            value_seq = value[seq_idx : seq_idx + 1]  # [1, max_seq_len, num_kv_heads, head_size]
            mask_seq = (
                mask[seq_idx : seq_idx + 1] if mask is not None else None
            )  # [1, 1, max_query_len, max_seq_len]

            # Compute attention for this sequence
            output_seq = self._compute_attention_single_seq(query_seq, key_seq, value_seq, mask_seq)

            # Store result
            output_all_seqs[seq_idx] = output_seq.squeeze(0)

        return output_all_seqs

    def _compute_attention_sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """SDPA fallback implementation."""
        out = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            attn_mask=~mask,
            scale=self.scale,
            enable_gqa=True,
        )
        return out.transpose(1, 2)

    def _compute_attention_single_seq(
        self,
        query: torch.Tensor,  # [1, max_query_len, num_heads, head_size]
        key: torch.Tensor,  # [1, max_seq_len, num_kv_heads, head_size]
        value: torch.Tensor,  # [1, max_seq_len, num_kv_heads, head_size]
        mask: torch.Tensor | None,  # [1, 1, max_query_len, max_seq_len]
    ) -> torch.Tensor:
        """Compute attention for a single sequence using Spyre.

        Processes queries in fixed-size chunks of QUERY_CHUNK_SIZE tokens.
        """
        batch_size, query_len, num_heads, head_size = query.shape
        _, kv_len, num_kv_heads, _ = key.shape

        assert batch_size == 1, f"Expected batch_size=1, got {batch_size}"

        # Handle grouped-query attention by repeating KV heads
        if self.num_queries_per_kv > 1:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=2)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Squeeze batch dimension
        query_squeezed = query.squeeze(0)  # [query_len, num_heads, head_size]
        key_squeezed = key.squeeze(0)  # [kv_len, num_heads, head_size]
        value_squeezed = value.squeeze(0)  # [kv_len, num_heads, head_size]

        # Calculate number of chunks needed
        actual_query_len = query_squeezed.shape[0]
        num_chunks = (actual_query_len + self.QUERY_CHUNK_SIZE - 1) // self.QUERY_CHUNK_SIZE

        # Allocate output tensor
        output_full = torch.zeros(
            actual_query_len,
            num_heads,
            head_size,
            dtype=query_squeezed.dtype,
            device=query_squeezed.device,
        )

        # Process each chunk
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.QUERY_CHUNK_SIZE
            chunk_end = min(chunk_start + self.QUERY_CHUNK_SIZE, actual_query_len)
            chunk_len = chunk_end - chunk_start

            # Extract query chunk
            query_chunk = query_squeezed[chunk_start:chunk_end]

            # Pad chunk if needed
            if chunk_len < self.QUERY_CHUNK_SIZE:
                padding_size = self.QUERY_CHUNK_SIZE - chunk_len
                query_chunk_padded = torch.nn.functional.pad(
                    query_chunk, (0, 0, 0, 0, 0, padding_size), mode="constant", value=0.0
                )
            else:
                query_chunk_padded = query_chunk

            # Extract corresponding mask for this chunk
            if mask is not None:
                mask_chunk = mask[:, :, chunk_start:chunk_end, :]  # [1, 1, chunk_len, kv_len]
            else:
                mask_chunk = None

            # Compute attention for this chunk
            chunk_output = self._compute_attention_chunk(
                query_chunk_padded,
                key_squeezed,
                value_squeezed,
                mask_chunk,
                chunk_len,
                num_heads,
                head_size,
                kv_len,
                chunk_idx,
            )

            # Store chunk output (only valid positions)
            output_full[chunk_start:chunk_end] = chunk_output[:chunk_len]

        return output_full.unsqueeze(0)  # [1, query_len, num_heads, head_size]

    def _compute_attention_chunk(
        self,
        query_chunk_padded: torch.Tensor,  # [QUERY_CHUNK_SIZE, num_heads, head_size]
        key_squeezed: torch.Tensor,  # [kv_len, num_heads, head_size]
        value_squeezed: torch.Tensor,  # [kv_len, num_heads, head_size]
        mask_chunk: torch.Tensor | None,  # [1, 1, chunk_len, kv_len]
        chunk_len: int,
        num_heads: int,
        head_size: int,
        kv_len: int,
        chunk_idx: int,
    ) -> torch.Tensor:
        """Compute attention for a single query chunk on Spyre.

        Returns:
            [QUERY_CHUNK_SIZE, num_heads, head_size] - attention output (padded)
        """
        try:
            spyre_device = torch.device("spyre")
            padded_query_len = self.QUERY_CHUNK_SIZE

            # Validate dimensions for Spyre
            if head_size <= 0 or num_heads <= 0 or kv_len <= 0 or padded_query_len <= 0:
                raise ValueError(
                    f"Invalid dimensions: head_size={head_size}, num_heads={num_heads}, "
                    f"kv_len={kv_len}, padded_query_len={padded_query_len}"
                )

            # Reshape query to flatten heads into query dimension
            query_reordered = query_chunk_padded.transpose(
                0, 1
            ).contiguous()  # [num_heads, QUERY_CHUNK_SIZE, head_size]
            query_flat = query_reordered.reshape(num_heads * padded_query_len, head_size)

            # Key and value: also flatten across heads
            key_reordered = key_squeezed.transpose(
                0, 1
            ).contiguous()  # [num_heads, kv_len, head_size]
            value_reordered = value_squeezed.transpose(
                0, 1
            ).contiguous()  # [num_heads, kv_len, head_size]

            key_flat = key_reordered.reshape(num_heads * kv_len, head_size)
            value_flat = value_reordered.reshape(num_heads * kv_len, head_size)

            # Transpose for attention computation
            qt = query_flat.T.contiguous()  # [head_size, num_heads * QUERY_CHUNK_SIZE]
            vt = value_flat.T.contiguous()  # [head_size, num_heads * kv_len]
            k = key_flat  # [num_heads * kv_len, head_size]

            # Apply stickification for Spyre
            # Query-like tensors (transposed format): double transpose-contiguous
            qt_stickified = qt.transpose(0, 1).contiguous().transpose(0, 1).contiguous()
            vt_stickified = vt.transpose(0, 1).contiguous().transpose(0, 1).contiguous()

            # Key-like tensors (standard format): single contiguous
            k_stickified = k.contiguous()

            # Create sm_scale as a 1D tensor repeated for each head and query position
            sm_scale_scalar = torch.tensor(self.scale, dtype=query_chunk_padded.dtype, device="cpu")
            sm_scale_1d = sm_scale_scalar.repeat(
                num_heads * padded_query_len
            )  # [num_heads * QUERY_CHUNK_SIZE]

            # Prepare mask values
            if mask_chunk is not None:
                mask_all_heads = mask_chunk[0, 0]  # [chunk_len, kv_len]

                # Pad if needed
                if chunk_len < self.QUERY_CHUNK_SIZE:
                    padding_size = self.QUERY_CHUNK_SIZE - chunk_len
                    mask_padding = torch.ones(
                        (padding_size, kv_len), dtype=torch.bool, device="cpu"
                    )
                    mask_all_heads = torch.cat([mask_all_heads, mask_padding], dim=0)

                # Build block-diagonal boolean mask.
                # ~block_diag(~per_head_block): diagonal = per_head_block, off-diagonal = True.
                # All heads share the same causal/padding mask, so the block is identical for each.
                head_mask_t = mask_all_heads.T  # [kv_len, QUERY_CHUNK_SIZE], True = masked
                mask_bool = ~torch.block_diag(*([~head_mask_t] * num_heads))

                # Convert: True (masked) -> -65504.0, False (valid) -> 0.0
                mask_values_cpu = torch.where(
                    mask_bool,
                    torch.tensor(-65504.0, dtype=query_chunk_padded.dtype, device="cpu"),
                    torch.tensor(0.0, dtype=query_chunk_padded.dtype, device="cpu"),
                )
            else:
                # No causal/padding mask: all within-head positions are valid.
                # ~block_diag(ones): diagonal = False (valid), off-diagonal = True (masked).
                ones_block = torch.ones(kv_len, padded_query_len, dtype=torch.bool, device="cpu")
                mask_bool = ~torch.block_diag(*([ones_block] * num_heads))
                mask_values_cpu = torch.where(
                    mask_bool,
                    torch.tensor(-65504.0, dtype=query_chunk_padded.dtype, device="cpu"),
                    torch.tensor(0.0, dtype=query_chunk_padded.dtype, device="cpu"),
                )

            # Ensure mask values are contiguous before transfer to Spyre
            mask_values_stickified = mask_values_cpu.contiguous()

            # Convert bfloat16 to float16 if needed (Spyre doesn't support bfloat16)
            if qt_stickified.dtype == torch.bfloat16:
                qt_stickified = qt_stickified.to(torch.float16)
                k_stickified = k_stickified.to(torch.float16)
                vt_stickified = vt_stickified.to(torch.float16)
                sm_scale_1d = sm_scale_1d.to(torch.float16)
                mask_values_stickified = mask_values_stickified.to(torch.float16)

            # Transfer to Spyre device
            qt_spyre = qt_stickified.to(spyre_device)
            k_spyre = k_stickified.to(spyre_device)
            vt_spyre = vt_stickified.to(spyre_device)
            sm_scale_spyre = sm_scale_1d.to(spyre_device)
            mask_values_spyre = mask_values_stickified.to(spyre_device)

            # Compute attention on Spyre (compiled)
            output_spyre_t = self.attn_transposed_compiled(
                qt_spyre, k_spyre, vt_spyre, sm_scale_spyre, mask_values_spyre
            )

            # Transfer back to CPU and convert back to original dtype if needed
            output_flat = output_spyre_t.to(
                "cpu"
            ).contiguous()  # [head_size, num_heads * QUERY_CHUNK_SIZE]

            # Convert back to original dtype (bfloat16) if we converted from it
            if query_chunk_padded.dtype == torch.bfloat16 and output_flat.dtype == torch.float16:
                output_flat = output_flat.to(torch.bfloat16)

            # Transpose and reshape
            output_transposed = output_flat.T  # [num_heads * QUERY_CHUNK_SIZE, head_size]
            output_reshaped = output_transposed.reshape(num_heads, padded_query_len, head_size)

            # Transpose to match expected format
            output_final = output_reshaped.transpose(
                0, 1
            ).contiguous()  # [QUERY_CHUNK_SIZE, num_heads, head_size]

            return output_final

        except Exception as e:
            raise RuntimeError(f"Error during Spyre computation for chunk {chunk_idx}: {e}") from e

    def _extract_relevant_output(
        self,
        attn_output: torch.Tensor,  # [num_seqs, max_query_len, num_heads, head_size]
        query_start_loc: torch.Tensor,  # [num_seqs + 1]
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

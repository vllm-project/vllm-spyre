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

    # Attention mask (optional)
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
            # Profiling run
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

        # Step 2: Gather compact KV cache (only relevant entries)
        compact_k, compact_v = self._gather_compact_kv_cache(
            kv_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            attn_metadata.block_size,
        )

        # Step 3: Create attention mask for compact KV cache
        max_seq_len = attn_metadata.seq_lens.max()
        compact_mask = self._create_compact_mask(
            attn_metadata.seq_lens,
            attn_metadata.query_start_loc,
            attn_metadata.slot_mapping,
            attn_metadata.causal_mask,
            attn_metadata.block_table,
            attn_metadata.block_size,
            query.shape[1],
            max_seq_len,
            kv_cache.device,
        )

        # Step 4: Prepare query tensor
        query_per_seq = self._reshape_query_to_sequences(
            query[:num_actual_tokens],
            attn_metadata.query_start_loc,
            attn_metadata.num_seqs,
            attn_metadata.max_query_len,
        )

        # Step 5: Compute attention with compact KV cache
        attn_output = self._compute_attention(
            query_per_seq.reshape(1, -1, query.shape[1], query.shape[2]),
            compact_k.reshape(1, -1, compact_k.shape[2], compact_k.shape[3]),
            compact_v.reshape(1, -1, compact_v.shape[2], compact_v.shape[3]),
            attn_metadata,
            block_mask=compact_mask,
        )

        # Step 6: Extract only relevant tokens
        attn_output = self._extract_relevant_output(
            attn_output,
            attn_metadata.query_start_loc,
        )

        # Copy to output tensor
        output[:num_actual_tokens].copy_(attn_output)

        return output

    def _write_to_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_size: int,
    ) -> None:
        """Write keys and values to paged KV cache."""

        num_tokens = key.shape[0]

        # Convert slot indices to block indices and offsets
        block_indices = slot_mapping // block_size
        block_offsets = slot_mapping % block_size

        # Get key and value caches
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]

        # Write keys and values using advanced indexing
        for i in range(num_tokens):
            # block_idx = block_indices[i].item()
            block_idx = block_indices[i]
            # offset = block_offsets[i].item()
            offset = block_offsets[i]
            key_cache[block_idx, offset] = key[i]
            value_cache[block_idx, offset] = value[i]

    def _gather_compact_kv_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather only the relevant KV cache entries into compact tensors.

        This method creates contiguous tensors containing only the KV entries
        that are actually used by the sequences in the batch, significantly
        reducing memory footprint and computation compared to using the full cache.

        Vectorized implementation without .item() calls for torch.compile compatibility.

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
        max_seq_len = seq_lens.max()
        num_kv_heads = kv_cache.shape[3]
        head_size = kv_cache.shape[4]
        device = kv_cache.device
        dtype = kv_cache.dtype

        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = kv_cache[1]

        # Allocate compact tensors
        compact_k = torch.zeros(
            num_seqs, max_seq_len, num_kv_heads, head_size, dtype=dtype, device=device
        )
        compact_v = torch.zeros(
            num_seqs, max_seq_len, num_kv_heads, head_size, dtype=dtype, device=device
        )

        # Vectorized gathering using advanced indexing
        # Create position indices for all sequences
        # [num_seqs, max_seq_len]
        position_indices = (
            torch.arange(max_seq_len, device=device).unsqueeze(0).expand(num_seqs, -1)
        )

        # Calculate which block each position belongs to
        # [num_seqs, max_seq_len]
        block_indices = position_indices // block_size
        offset_in_block = position_indices % block_size

        # Get physical block numbers from block table
        # [num_seqs, max_seq_len]
        max_blocks_per_seq = block_table.shape[1]
        block_indices_clamped = torch.clamp(block_indices, 0, max_blocks_per_seq - 1)
        physical_blocks = block_table.gather(1, block_indices_clamped)

        # Create mask for valid positions (within sequence length)
        # [num_seqs, max_seq_len]
        valid_mask = position_indices < seq_lens.unsqueeze(1)

        # Gather from cache using advanced indexing
        # We need to handle this carefully to avoid out-of-bounds access
        # For invalid positions, we'll use block 0 (which should be zeros or unused)
        physical_blocks_safe = torch.where(
            valid_mask, physical_blocks, torch.zeros_like(physical_blocks)
        )
        offset_in_block_safe = torch.where(
            valid_mask, offset_in_block, torch.zeros_like(offset_in_block)
        )

        # Reshape for gathering: [num_seqs * max_seq_len]
        flat_physical_blocks = physical_blocks_safe.reshape(-1)
        flat_offsets = offset_in_block_safe.reshape(-1)

        # Gather keys and values
        # key_cache shape: [num_blocks, block_size, num_kv_heads, head_size]
        gathered_k = key_cache[
            flat_physical_blocks, flat_offsets
        ]  # [num_seqs * max_seq_len, num_kv_heads, head_size]
        gathered_v = value_cache[flat_physical_blocks, flat_offsets]

        # Reshape back to [num_seqs, max_seq_len, num_kv_heads, head_size]
        compact_k = gathered_k.reshape(num_seqs, max_seq_len, num_kv_heads, head_size)
        compact_v = gathered_v.reshape(num_seqs, max_seq_len, num_kv_heads, head_size)

        # Zero out invalid positions (optional, but cleaner)
        valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(
            -1
        )  # [num_seqs, max_seq_len, 1, 1]
        compact_k = compact_k * valid_mask_expanded
        compact_v = compact_v * valid_mask_expanded

        return compact_k, compact_v

    def _create_compact_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        slot_mapping: torch.Tensor,
        causal_mask: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        num_heads: int,
        max_seq_len: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create attention mask for compact KV cache.

        Vectorized implementation without .item() calls for torch.compile compatibility.

        Args:
            seq_lens: [num_seqs] - actual sequence lengths
            query_start_loc: [num_seqs + 1] - cumulative query positions
            slot_mapping: [num_actual_tokens] - maps tokens to cache slots
            causal_mask: bool tensor or None
            block_table: [num_seqs, max_num_blocks_per_seq]
            block_size: int
            num_heads: int
            max_seq_len: tensor - maximum sequence length in batch
            device: torch device

        Returns:
            mask: [1, num_heads, num_seqs * max_query_len, num_seqs * max_seq_len]
                  True = masked out, False = attend
        """
        num_seqs = seq_lens.shape[0]

        # Calculate query lengths for each sequence
        # [num_seqs]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        max_query_len = query_lens.max()

        # Initialize mask (True = masked out)
        # [num_seqs * max_query_len, num_seqs * max_seq_len]
        compact_mask = torch.ones(
            num_seqs * max_query_len,
            num_seqs * max_seq_len,
            dtype=torch.bool,
            device=device,
        )

        # Create position indices
        # [num_seqs, max_query_len]
        query_positions = (
            torch.arange(max_query_len, device=device).unsqueeze(0).expand(num_seqs, -1)
        )
        # [num_seqs, max_seq_len]
        kv_positions = torch.arange(max_seq_len, device=device).unsqueeze(0).expand(num_seqs, -1)

        # Create validity masks
        # [num_seqs, max_query_len]
        query_valid = query_positions < query_lens.unsqueeze(1)
        # [num_seqs, max_seq_len]
        kv_valid = kv_positions < seq_lens.unsqueeze(1)

        # Create global indices for mask
        # [num_seqs, max_query_len]
        query_global_idx = (
            torch.arange(num_seqs, device=device).unsqueeze(1) * max_query_len + query_positions
        )
        # [num_seqs, max_seq_len]
        kv_global_idx = (
            torch.arange(num_seqs, device=device).unsqueeze(1) * max_seq_len + kv_positions
        )

        # Unmask valid positions
        # [num_seqs, max_query_len, max_seq_len]
        valid_mask_3d = query_valid.unsqueeze(2) & kv_valid.unsqueeze(1)

        # Flatten to 2D indices
        query_flat = query_global_idx.unsqueeze(2).expand(-1, -1, max_seq_len)[valid_mask_3d]
        kv_flat = kv_global_idx.unsqueeze(1).expand(-1, max_query_len, -1)[valid_mask_3d]

        compact_mask[query_flat, kv_flat] = False

        # Apply causal masking if needed
        if causal_mask is not None:
            # For causal masking, we need to mask based on token positions
            # In compact representation, position i can only attend
            # to positions <= i within the same sequence

            # # Create causal mask for each sequence
            # # [max_query_len, max_seq_len]
            # causal_pattern = (
            #     query_positions[0:1, :, None]
            #     + (seq_lens[0:1, None, None] - query_lens[0:1, None, None])
            #     < kv_positions[0:1, None, :]
            # )

            # Apply to all sequences
            for seq_idx in range(num_seqs):
                q_start = seq_idx * max_query_len
                # q_end = q_start + query_lens[seq_idx]
                kv_start = seq_idx * max_seq_len
                # kv_end = kv_start + seq_lens[seq_idx]

                # Create causal mask for this sequence
                seq_query_len = query_lens[seq_idx]
                seq_seq_len = seq_lens[seq_idx]
                context_len = seq_seq_len - seq_query_len

                # Query position i (relative) can attend to KV positions 0 to context_len + i
                q_pos = torch.arange(seq_query_len, device=device)
                kv_pos = torch.arange(seq_seq_len, device=device)

                # Causal mask: query at position i can attend to kv
                # at position j if j <= context_len + i
                causal_seq_mask = kv_pos[None, :] > (context_len + q_pos[:, None])

                # Apply to global mask
                compact_mask[
                    q_start : q_start + seq_query_len, kv_start : kv_start + seq_seq_len
                ] |= causal_seq_mask

        # Expand for batch and heads
        compact_mask = compact_mask.unsqueeze(0).unsqueeze(0)
        compact_mask = compact_mask.expand(-1, num_heads, -1, -1)

        return compact_mask

    def _extract_relevant_output(
        self,
        attn_output: torch.Tensor,
        query_start_loc: torch.Tensor,
    ):
        num_tokens_per_sequence = [
            query_start_loc[i + 1] - query_start_loc[i] for i in range(len(query_start_loc) - 1)
        ]
        max_num_tokens_per_sequence = torch.max(torch.stack(num_tokens_per_sequence))

        token_indices = []
        for seq_idx, seq_n_tokens in enumerate(num_tokens_per_sequence):
            token_indices.extend(
                range(
                    seq_idx * max_num_tokens_per_sequence,
                    seq_idx * max_num_tokens_per_sequence + seq_n_tokens,
                )
            )

        return attn_output[token_indices]

    def _reshape_query_to_sequences(
        self,
        query: torch.Tensor,
        query_start_loc: torch.Tensor,
        num_seqs: int,
        max_query_len: int,
    ) -> torch.Tensor:
        """
        Reshape query from flat tokens to per-sequence format.

        Vectorized implementation without .item() calls for torch.compile compatibility.
        """

        num_heads = query.shape[1]
        head_size = query.shape[2]
        device = query.device

        # Initialize output
        query_per_seq = torch.zeros(
            num_seqs,
            max_query_len,
            num_heads,
            head_size,
            dtype=query.dtype,
            device=device,
        )

        # Calculate query lengths for each sequence
        # [num_seqs]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        # Create indices for gathering
        # [num_seqs, max_query_len]
        position_indices = (
            torch.arange(max_query_len, device=device).unsqueeze(0).expand(num_seqs, -1)
        )

        # Calculate global indices in the flat query tensor
        # [num_seqs, max_query_len]
        global_indices = query_start_loc[:-1].unsqueeze(1) + position_indices

        # Create mask for valid positions
        # [num_seqs, max_query_len]
        valid_mask = position_indices < query_lens.unsqueeze(1)

        # Clamp indices to avoid out-of-bounds (invalid positions will be masked anyway)
        global_indices_clamped = torch.clamp(global_indices, 0, query.shape[0] - 1)

        # Gather queries
        # [num_seqs, max_query_len, num_heads, head_size]
        query_per_seq = query[global_indices_clamped]

        # Zero out invalid positions
        valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(
            -1
        )  # [num_seqs, max_query_len, 1, 1]
        query_per_seq = query_per_seq * valid_mask_expanded

        return query_per_seq

    def _compute_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SpyreAttentionPagedMetadata,
        block_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute attention using PyTorch operations."""

        # Handle grouped-query attention
        if self.num_queries_per_kv > 1:
            # Repeat KV heads to match query heads
            key = key.repeat_interleave(self.num_queries_per_kv, dim=2)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Bq, Bkv = query.size(0), key.size(0)
        # if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        #     raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

        # key = key.expand((Bq, *key.size()[1:]))
        # value = value.expand((Bq, *value.size()[1:]))

        # Transpose for matmul
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Compute Q @ K^T
        attn_scores = torch.matmul(
            query.to(self.working_precision), key.to(self.working_precision).transpose(-2, -1)
        )

        # Scale
        attn_scores = (attn_scores * self.scale).to(self.working_precision)

        if block_mask is not None:
            attn_scores = attn_scores.masked_fill(block_mask, -float("inf"))

        # Softmax
        attn_weights = torch._safe_softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights.to(query.dtype), value.to(query.dtype))

        # Transpose back
        attn_output = attn_output.transpose(1, 2).squeeze(0)

        return attn_output

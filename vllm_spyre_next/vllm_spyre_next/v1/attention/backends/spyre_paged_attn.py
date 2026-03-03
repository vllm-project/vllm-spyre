# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure PyTorch implementation of PagedAttention.

This backend aims to implement PagedAttention using only PyTorch native operations, such as matmul, softmax, etc.
It supports vLLM's KV cache.
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
        max_seq_len = attn_metadata.seq_lens.max().item()
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
        max_seq_len = seq_lens.max().item()
        num_kv_heads = kv_cache.shape[3]
        head_size = kv_cache.shape[4]
        device = kv_cache.device
        dtype = kv_cache.dtype
        
        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = kv_cache[1]
        
        # Allocate compact tensors
        compact_k = torch.zeros(
            num_seqs, max_seq_len, num_kv_heads, head_size,
            dtype=dtype, device=device
        )
        compact_v = torch.zeros(
            num_seqs, max_seq_len, num_kv_heads, head_size,
            dtype=dtype, device=device
        )
        
        # Gather KV entries for each sequence
        for seq_idx in range(num_seqs):
            seq_len = seq_lens[seq_idx].item()
            num_blocks_needed = (seq_len + block_size - 1) // block_size
            
            for block_idx in range(num_blocks_needed):
                # Get physical block index from block table
                physical_block = block_table[seq_idx, block_idx].item()
                
                # Calculate how many tokens to copy from this block
                start_pos = block_idx * block_size
                end_pos = min(start_pos + block_size, seq_len)
                tokens_in_block = end_pos - start_pos
                
                # Copy tokens from physical block to compact tensor
                compact_k[seq_idx, start_pos:end_pos] = key_cache[
                    physical_block, :tokens_in_block
                ]
                compact_v[seq_idx, start_pos:end_pos] = value_cache[
                    physical_block, :tokens_in_block
                ]
        
        return compact_k, compact_v

    def _gather_block_mask(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        causal_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        block_size: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Gather keys or values from paged KV cache."""

        num_seqs = block_table.shape[0]
        num_blocks, block_size, num_kv_heads, head_size = kv_cache.shape[1:]

        num_tokens_per_sequence = [
            query_start_loc[i + 1] - query_start_loc[i] for i in range(len(query_start_loc) - 1)
        ]
        max_num_tokens_per_sequence = torch.max(torch.stack(num_tokens_per_sequence))

        # Initialize output tensor
        block_mask = torch.ones(
            num_seqs * max_num_tokens_per_sequence,
            num_blocks * block_size,
            dtype=torch.bool,
            device=kv_cache.device,
        )

        # Gather tokens for each sequence
        for seq_idx in range(num_seqs):
            seq_remainder = seq_lens[seq_idx] % block_size
            for bl_ind, bl in enumerate(block_table[seq_idx]):
                seq_len_curr_block = min(seq_lens[seq_idx] - bl_ind * block_size, block_size)
                max_bl_ind = (seq_lens[seq_idx] // block_size) + (
                    1 if seq_lens[seq_idx] % block_size else 0
                )
                if bl_ind < max_bl_ind:
                    block_mask[
                        seq_idx * max_num_tokens_per_sequence : (seq_idx + 1)
                        * max_num_tokens_per_sequence,
                        bl * block_size : bl * block_size + seq_len_curr_block,
                    ] = 0

            if causal_mask is not None:
                for token_nr in range(num_tokens_per_sequence[seq_idx] - 1):
                    slot_for_write = slot_mapping[query_start_loc[seq_idx] + 1 + token_nr :]
                    block_mask[seq_idx * max_num_tokens_per_sequence + token_nr, slot_for_write] = 1

        block_mask = block_mask.unsqueeze(0).unsqueeze(0)
        block_mask = block_mask.expand(-1, num_heads, -1, -1)

        return block_mask
    
    def _create_compact_mask(
        self,
        seq_lens: torch.Tensor,
        query_start_loc: torch.Tensor,
        slot_mapping: torch.Tensor,
        causal_mask: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        num_heads: int,
        max_seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create attention mask for compact KV cache.
        
        This mask works with the compact KV representation where tokens are
        arranged sequentially per sequence, rather than scattered across blocks.
        
        Args:
            seq_lens: [num_seqs] - actual sequence lengths
            query_start_loc: [num_seqs + 1] - cumulative query positions
            slot_mapping: [num_actual_tokens] - maps tokens to cache slots
            causal_mask: bool tensor or None
            block_table: [num_seqs, max_num_blocks_per_seq]
            block_size: int
            num_heads: int
            max_seq_len: int - maximum sequence length in batch
            device: torch device
            
        Returns:
            mask: [1, num_heads, num_seqs * max_query_len, num_seqs * max_seq_len]
                  True = masked out, False = attend
        """
        num_seqs = seq_lens.shape[0]
        
        num_tokens_per_sequence = [
            query_start_loc[i + 1] - query_start_loc[i] for i in range(len(query_start_loc) - 1)
        ]
        max_num_tokens_per_sequence = torch.max(torch.stack(num_tokens_per_sequence))
        
        # Initialize mask (True = masked out)
        compact_mask = torch.ones(
            num_seqs * max_num_tokens_per_sequence,
            num_seqs * max_seq_len,
            dtype=torch.bool,
            device=device,
        )
        
        # Mark valid KV positions for each sequence
        for seq_idx in range(num_seqs):
            seq_len = seq_lens[seq_idx].item()
            query_len = num_tokens_per_sequence[seq_idx].item()
            
            # Unmask valid KV positions (all positions up to seq_len)
            compact_mask[
                seq_idx * max_num_tokens_per_sequence : seq_idx * max_num_tokens_per_sequence + query_len,
                seq_idx * max_seq_len : seq_idx * max_seq_len + seq_len,
            ] = 0
            
            # Apply causal masking if needed
            if causal_mask is not None:
                # For each query token, mask out future KV tokens
                # We need to map from compact KV positions back to slot indices
                # to determine which tokens are "future" relative to each query
                
                # Build mapping from compact position to slot index for this sequence
                compact_to_slot = torch.zeros(seq_len, dtype=torch.long, device=device)
                for block_idx in range((seq_len + block_size - 1) // block_size):
                    physical_block = block_table[seq_idx, block_idx].item()
                    start_pos = block_idx * block_size
                    end_pos = min(start_pos + block_size, seq_len)
                    tokens_in_block = end_pos - start_pos
                    
                    # Map compact positions to slot indices
                    for i in range(tokens_in_block):
                        compact_pos = start_pos + i
                        slot_idx = physical_block * block_size + i
                        compact_to_slot[compact_pos] = slot_idx
                
                # For each query token, mask future tokens based on slot_mapping
                for token_nr in range(query_len - 1):
                    query_global_idx = query_start_loc[seq_idx].item() + 1 + token_nr
                    # Get slots that will be written by future tokens
                    future_slots = slot_mapping[query_global_idx:]
                    
                    # Find which compact positions correspond to these future slots
                    for kv_pos in range(seq_len):
                        if compact_to_slot[kv_pos] in future_slots:
                            compact_mask[
                                seq_idx * max_num_tokens_per_sequence + token_nr,
                                seq_idx * max_seq_len + kv_pos
                            ] = 1
        
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
        """Reshape query from flat tokens to per-sequence format."""

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

        # Fill in queries for each sequence
        for seq_idx in range(num_seqs):
            start = query_start_loc[seq_idx].item()
            end = query_start_loc[seq_idx + 1].item()
            seq_len = end - start

            query_per_seq[seq_idx, :seq_len] = query[start:end]

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

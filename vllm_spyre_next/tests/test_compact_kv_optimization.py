# SPDX-License-Identifier: Apache-2.0
"""Test to verify correctness of compact KV cache optimization.

This test compares the old (full cache) implementation with the new
(compact cache) implementation to ensure they produce identical results.
"""

import torch
import pytest


def create_test_metadata(
    num_seqs: int,
    seq_lens: list[int],
    query_lens: list[int],
    block_size: int,
    num_kv_heads: int,
    num_heads: int,
    device: torch.device,
):
    """Create test metadata similar to vLLM's CommonAttentionMetadata."""
    from vllm_spyre_next.v1.attention.backends.spyre_paged_attn import (
        SpyreAttentionPagedMetadata,
    )

    # Create query start locations
    query_start_loc = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
    query_start_loc[1:] = torch.tensor(query_lens, dtype=torch.int32, device=device).cumsum(0)

    # Create sequence lengths
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    max_seq_len = max(seq_lens)
    max_query_len = max(query_lens)

    # Create block table
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_table = torch.zeros(num_seqs, max_blocks_per_seq, dtype=torch.int32, device=device)

    # Assign blocks sequentially for each sequence
    current_block = 1  # Start from block 1 (block 0 is null)
    for seq_idx in range(num_seqs):
        num_blocks_needed = (seq_lens[seq_idx] + block_size - 1) // block_size
        block_table[seq_idx, :num_blocks_needed] = torch.arange(
            current_block, current_block + num_blocks_needed, dtype=torch.int32, device=device
        )
        current_block += num_blocks_needed

    # Create slot mapping
    num_actual_tokens = sum(query_lens)
    slot_mapping = torch.zeros(num_actual_tokens, dtype=torch.int64, device=device)

    # Fill slot mapping based on block table
    token_idx = 0
    for seq_idx in range(num_seqs):
        context_len = seq_lens[seq_idx] - query_lens[seq_idx]
        for q_idx in range(query_lens[seq_idx]):
            token_pos = context_len + q_idx
            block_idx = token_pos // block_size
            offset_in_block = token_pos % block_size
            physical_block = block_table[seq_idx, block_idx].item()
            slot_mapping[token_idx] = physical_block * block_size + offset_in_block
            token_idx += 1

    # Create causal mask
    causal_mask = torch.tril(
        torch.ones(max_query_len, max_seq_len, dtype=torch.bool, device=device)
    )

    return SpyreAttentionPagedMetadata(
        num_actual_tokens=num_actual_tokens,
        num_seqs=num_seqs,
        max_query_len=max_query_len,
        max_seq_len=max_seq_len,
        seq_lens=seq_lens_tensor,
        query_start_loc=query_start_loc,
        block_table=block_table,
        block_size=block_size,
        slot_mapping=slot_mapping,
        causal_mask=causal_mask,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
    )


def create_and_populate_kv_cache(
    k_contexts: list[torch.Tensor],
    v_contexts: list[torch.Tensor],
    block_table: torch.Tensor,
    seq_lens: list[int],
    block_size: int,
    num_blocks: int,
    num_kv_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Create and populate KV cache with context data."""
    kv_cache = torch.zeros(
        2, num_blocks, block_size, num_kv_heads, head_size, dtype=dtype, device=device
    )

    for seq_idx in range(len(k_contexts)):
        k_context = k_contexts[seq_idx]
        v_context = v_contexts[seq_idx]
        context_len = k_context.shape[0]

        # Write context tokens to cache
        for token_idx in range(context_len):
            block_idx_in_seq = token_idx // block_size
            offset_in_block = token_idx % block_size
            physical_block = block_table[seq_idx, block_idx_in_seq].item()

            kv_cache[0, physical_block, offset_in_block] = k_context[token_idx]
            kv_cache[1, physical_block, offset_in_block] = v_context[token_idx]

    return kv_cache


@pytest.mark.parametrize(
    "batch_config",
    [
        # (num_seqs, seq_lens, query_lens)
        (2, [32, 40], [8, 8]),  # Small prefill
        (4, [32, 40, 48, 56], [1, 1, 5, 5]),  # Mixed
        (1, [128], [1]),  # Single decode
        (2, [64, 128], [16, 32]),  # Medium prefill
    ],
)
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("num_kv_heads", [8, 16])
@pytest.mark.parametrize("head_size", [64, 128])
def test_compact_kv_correctness(batch_config, block_size, num_kv_heads, head_size):
    """Test that compact KV implementation produces same results as full cache."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    num_seqs, seq_lens, query_lens = batch_config
    num_heads = num_kv_heads  # For simplicity, use MHA (can test GQA separately)

    # Calculate total blocks needed
    max_seq_len = max(seq_lens)
    total_blocks_needed = sum((s + block_size - 1) // block_size for s in seq_lens)
    num_blocks = total_blocks_needed + 10  # Add some extra blocks

    # Create metadata
    metadata = create_test_metadata(
        num_seqs, seq_lens, query_lens, block_size, num_kv_heads, num_heads, device
    )

    # Generate random Q, K, V tensors
    num_actual_tokens = sum(query_lens)
    query = torch.randn(num_actual_tokens, num_heads, head_size, dtype=dtype, device=device)
    key = torch.randn(num_actual_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(num_actual_tokens, num_kv_heads, head_size, dtype=dtype, device=device)

    # Generate context K, V for each sequence
    k_contexts = []
    v_contexts = []
    for seq_idx in range(num_seqs):
        context_len = seq_lens[seq_idx] - query_lens[seq_idx]
        k_ctx = torch.randn(context_len, num_kv_heads, head_size, dtype=dtype, device=device)
        v_ctx = torch.randn(context_len, num_kv_heads, head_size, dtype=dtype, device=device)
        k_contexts.append(k_ctx)
        v_contexts.append(v_ctx)

    # Create and populate KV cache
    kv_cache = create_and_populate_kv_cache(
        k_contexts,
        v_contexts,
        metadata.block_table,
        seq_lens,
        block_size,
        num_blocks,
        num_kv_heads,
        head_size,
        dtype,
        device,
    )

    # Import the implementation
    from vllm_spyre_next.v1.attention.backends.spyre_paged_attn import (
        SpyreAttentionPagedImpl,
    )

    # Create attention implementation
    scale = 1.0 / (head_size**0.5)
    impl = SpyreAttentionPagedImpl(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )

    # Test the old implementation (using full cache)
    # We'll call the internal methods directly to compare
    
    # First, write new K/V to cache
    impl._write_to_kv_cache(
        key, value, kv_cache, metadata.slot_mapping, metadata.block_size
    )

    # OLD METHOD: Use full cache with block mask
    key_cache_old = kv_cache[0]
    value_cache_old = kv_cache[1]

    block_mask_old = impl._gather_block_mask(
        kv_cache,
        metadata.block_table,
        metadata.seq_lens,
        metadata.causal_mask,
        metadata.slot_mapping,
        metadata.query_start_loc,
        metadata.block_size,
        num_heads,
    )

    query_per_seq = impl._reshape_query_to_sequences(
        query,
        metadata.query_start_loc,
        metadata.num_seqs,
        metadata.max_query_len,
    )

    # Compute attention with old method (full cache)
    output_old = impl._compute_attention(
        query_per_seq.reshape(1, -1, num_heads, head_size),
        key_cache_old.reshape(1, -1, num_kv_heads, head_size),
        value_cache_old.reshape(1, -1, num_kv_heads, head_size),
        metadata,
        block_mask=block_mask_old,
    )

    output_old = impl._extract_relevant_output(output_old, metadata.query_start_loc)

    # NEW METHOD: Use compact cache
    compact_k, compact_v = impl._gather_compact_kv_cache(
        kv_cache, metadata.block_table, metadata.seq_lens, metadata.block_size
    )

    max_seq_len = metadata.seq_lens.max().item()
    compact_mask = impl._create_compact_mask(
        metadata.seq_lens,
        metadata.query_start_loc,
        metadata.slot_mapping,
        metadata.causal_mask,
        metadata.block_table,
        metadata.block_size,
        num_heads,
        max_seq_len,
        device,
    )

    # Compute attention with new method (compact cache)
    output_new = impl._compute_attention(
        query_per_seq.reshape(1, -1, num_heads, head_size),
        compact_k.reshape(1, -1, num_kv_heads, head_size),
        compact_v.reshape(1, -1, num_kv_heads, head_size),
        metadata,
        block_mask=compact_mask,
    )

    output_new = impl._extract_relevant_output(output_new, metadata.query_start_loc)

    # Compare outputs
    torch.testing.assert_close(
        output_new,
        output_old,
        rtol=1e-3,
        atol=1e-3,
        msg=f"Outputs differ for config: {batch_config}, block_size={block_size}",
    )

    print(
        f"✓ Test passed for config: seqs={num_seqs}, seq_lens={seq_lens}, "
        f"query_lens={query_lens}, block_size={block_size}, "
        f"num_kv_heads={num_kv_heads}, head_size={head_size}"
    )


if __name__ == "__main__":
    # Run a simple test
    test_compact_kv_correctness(
        batch_config=(2, [32, 40], [8, 8]),
        block_size=16,
        num_kv_heads=8,
        head_size=64,
    )
    print("\n✓ All tests passed!")

# Made with Bob

import torch
from dataclasses import dataclass


@dataclass
class PyTorchNativeAttentionMetadata:
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


def _compute_attention_per_head_loop_core(qt_reshaped, k_reshaped, vt_reshaped, sm_scale, num_heads):
    """Core computation for per-head attention (to be compiled).

    This version processes one head at a time to avoid Spyre compilation issues.
    Note: Returns result in [D, Q] format to avoid transpose issues.

    Args:
        qt_reshaped: [D, Q] - already on target device (single head)
        k_reshaped: [L, D] - already on target device (single head)
        vt_reshaped: [D, L] - already on target device (single head)
        sm_scale: scalar tensor
        num_heads: int (should be 1 for single-head processing)

    Returns:
        attn_output_t: [D, Q] for single head (will be transposed outside)
    """
    # For single head processing: qt_reshaped is [D, Q], k_reshaped is [L, D], vt_reshaped is [D, L]
    # kq = k @ qt: [L, Q]
    kq = k_reshaped @ qt_reshaped

    # Scale: [L, Q]
    kq = kq * sm_scale

    # Softmax along L dimension (dim=0): [L, Q]
    p = kq.softmax(dim=0)

    # attn_output_t = vt @ p: [D, Q]
    # Return in [D, Q] format - transpose will be done outside compiled function
    attn_output_t = vt_reshaped @ p

    return attn_output_t


def _compute_attention_per_head_loop(qt_reshaped, k_reshaped, vt_reshaped, sm_scale, num_heads, device=None):
    """Wrapper function that processes each head separately to avoid Spyre compilation issues.

    Args:
        qt_reshaped: [D, Q, H]
        k_reshaped: [L, H, D]
        vt_reshaped: [D, L, H]
        sm_scale: scalar (float or tensor)
        num_heads: int
        device: target device for computation (e.g., 'spyre')

    Returns:
        attn_output: [Q, H, D]
    """
    # Store original device
    original_device = qt_reshaped.device

    # Convert sm_scale to tensor if it's a scalar (required for Spyre compilation)
    if not isinstance(sm_scale, torch.Tensor):
        sm_scale = torch.tensor(sm_scale, dtype=qt_reshaped.dtype, device=qt_reshaped.device)

    # Get dimensions
    D = qt_reshaped.shape[0]  # head_size
    Q = qt_reshaped.shape[1]  # total_tokens

    # Pre-allocate output tensor on original device
    attn_output = torch.empty(Q, num_heads, D, dtype=qt_reshaped.dtype, device=original_device)

    # Process each head separately
    for h in range(num_heads):
        # Extract tensors for this head
        qt_h = qt_reshaped[:, :, h].contiguous()  # [D, Q]
        k_h = k_reshaped[:, h, :].contiguous()    # [L, D]
        vt_h = vt_reshaped[:, :, h].contiguous()  # [D, L]

        # Move tensors to target device if specified
        if device is not None:
            qt_h = qt_h.to(device)
            k_h = k_h.to(device)
            vt_h = vt_h.to(device)
            sm_scale_device = sm_scale.to(device)
        else:
            sm_scale_device = sm_scale

        # Call the compiled core function for this head
        # Returns [D, Q] format to avoid transpose issues in compiled code
        attn_output_t = _compute_attention_per_head_loop_compiled(
            qt_h, k_h, vt_h, sm_scale_device, 1  # num_heads=1 for single head
        )

        # Move result back to original device first (result is [D, Q] and contiguous)
        if device is not None and device != original_device:
            attn_output_t = attn_output_t.to(original_device)

        # Transpose on CPU to get [Q, D] - this is safe and produces contiguous result
        attn_output_h = attn_output_t.transpose(0, 1).contiguous()

        # Store in output tensor
        attn_output[:, h, :] = attn_output_h

    return attn_output


# Apply torch.compile to the core computation function (without device transfers)
# Use "eager" backend on Windows to avoid C++ compiler requirement
import sys
if sys.platform == "win32":
    # On Windows, use eager backend to avoid C++ compiler issues
    _compute_attention_per_head_loop_compiled = torch.compile(
        _compute_attention_per_head_loop_core,
        backend="eager"
    )
else:
    # On other platforms, use inductor for better performance
    _compute_attention_per_head_loop_compiled = torch.compile(
        _compute_attention_per_head_loop_core,
        backend="inductor"
    )


class AttentionDebugger:
    """Debug class to test attention implementations."""

    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, working_precision=None, spyre_device=None):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.spyre_device = spyre_device  # Device for Spyre computation

        if working_precision is not None:
            self.working_precision = working_precision
        else:
            self.working_precision = torch.bfloat16

    def _compute_attention1(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: PyTorchNativeAttentionMetadata,
        block_mask: torch.Tensor = None
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
        attn_scores = torch.matmul(query.to(self.working_precision), key.to(self.working_precision).transpose(-2, -1))

        # Scale
        attn_scores = (attn_scores * self.scale).to(self.working_precision)

        if block_mask is not None:
            attn_scores = attn_scores.masked_fill(block_mask, -float('inf'))

        # Softmax
        attn_weights = torch._safe_softmax(attn_scores, dim=-1)

        # Compute attention output
        attn_output = torch.matmul(attn_weights.to(query.dtype), value.to(query.dtype))

        # Transpose back
        attn_output = attn_output.transpose(1, 2).squeeze(0)

        return attn_output



    def _compute_attention2(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: PyTorchNativeAttentionMetadata,
        block_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute attention using PyTorch operations with transposed computation.

        Input shapes:
            query: [1, total_tokens, num_heads, head_size]
            key: [1, total_kv_tokens, num_kv_heads, head_size]
            value: [1, total_kv_tokens, num_kv_heads, head_size]

        Converts to attn_transposed format:
            qt: [head_size, num_heads * total_tokens] = [D, H*Q]
            k: [total_kv_tokens * num_kv_heads, head_size] = [L*H_kv, D]
            vt: [head_size, total_kv_tokens * num_kv_heads] = [D, L*H_kv]
            sm_scale: scalar
        """
        # Handle grouped-query attention by repeating KV heads
        if self.num_queries_per_kv > 1:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=2)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=2)

        # Remove batch dimension (batch=1)
        # query: [total_tokens, num_heads, head_size]
        # key: [total_kv_tokens, num_heads, head_size]
        # value: [total_kv_tokens, num_heads, head_size]
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)

        # Get dimensions
        total_tokens = query.shape[0]
        num_heads = query.shape[1]
        head_size = query.shape[2]
        total_kv_tokens = key.shape[0]

        # Reshape query to [total_tokens * num_heads, head_size] then transpose
        # qt: [head_size, total_tokens * num_heads] = [D, H*Q]
        query_flat = query.reshape(total_tokens * num_heads, head_size)
        qt = query_flat.transpose(0, 1).contiguous()

        # Reshape key to [total_kv_tokens * num_heads, head_size]
        # k: [total_kv_tokens * num_heads, head_size] = [L*H, D]
        k = key.reshape(total_kv_tokens * num_heads, head_size)

        # Reshape value to [total_kv_tokens * num_heads, head_size] then transpose
        # vt: [head_size, total_kv_tokens * num_heads] = [D, L*H]
        value_flat = value.reshape(total_kv_tokens * num_heads, head_size)
        vt = value_flat.transpose(0, 1).contiguous()

        # Use scalar scale
        sm_scale = self.scale

        # Inline attn_transposed computation
        # qt: [D, H*Q], k: [L*H, D], vt: [D, L*H], sm_scale: scalar

        # We need to compute attention per head, so we need to reshape k and vt
        # to separate heads and compute attention for each query-head pair

        # Reshape k: [L*H, D] -> [L, H, D]
        k_reshaped = k.reshape(total_kv_tokens, num_heads, head_size)

        # Reshape vt: [D, L*H] -> [D, L, H]
        vt_reshaped = vt.reshape(head_size, total_kv_tokens, num_heads)

        # Reshape qt: [D, H*Q] -> [D, Q, H]
        qt_reshaped = qt.reshape(head_size, total_tokens, num_heads)

        # Use wrapper function for per-head attention computation on Spyre device
        # The wrapper handles device transfers and calls the compiled core function
        attn_output = _compute_attention_per_head_loop(
            qt_reshaped, k_reshaped, vt_reshaped, sm_scale, num_heads, device=self.spyre_device
        )

        # Reshape to [total_tokens, num_heads, head_size]
        attn_output = attn_output.reshape(total_tokens, num_heads, head_size)

        return attn_output


def generate_test_inputs(
    num_heads: int = 8,
    num_kv_heads: int = 8,
    head_size: int = 64,
    total_tokens: int = 32,
    total_kv_tokens: int = 128,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device('cpu')
):
    """Generate test inputs for attention functions.

    Args:
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        head_size: Dimension of each head
        total_tokens: Number of query tokens
        total_kv_tokens: Number of key/value tokens
        dtype: Data type for tensors
        device: Device to create tensors on

    Returns:
        Tuple of (query, key, value, attn_metadata, block_mask)
    """
    # Create query, key, value tensors
    # Shape: [1, total_tokens, num_heads, head_size]
    query = torch.randn(1, total_tokens, num_heads, head_size, dtype=dtype, device=device)

    # Shape: [1, total_kv_tokens, num_kv_heads, head_size]
    key = torch.randn(1, total_kv_tokens, num_kv_heads, head_size, dtype=dtype, device=device)
    value = torch.randn(1, total_kv_tokens, num_kv_heads, head_size, dtype=dtype, device=device)

    # Create minimal attention metadata
    num_seqs = 2
    seq_lens = torch.tensor([64, 64], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 16, 32], dtype=torch.int32, device=device)
    block_table = torch.zeros(num_seqs, 10, dtype=torch.int32, device=device)
    slot_mapping = torch.arange(total_tokens, dtype=torch.int32, device=device)

    attn_metadata = PyTorchNativeAttentionMetadata(
        num_actual_tokens=total_tokens,
        num_seqs=num_seqs,
        max_query_len=16,
        max_seq_len=64,
        seq_lens=seq_lens,
        query_start_loc=query_start_loc,
        block_table=block_table,
        block_size=16,
        slot_mapping=slot_mapping,
        causal_mask=None,
        num_kv_heads=num_kv_heads,
        num_heads=num_heads,
    )

    # Optional block mask (None for now)
    block_mask = None

    return query, key, value, attn_metadata, block_mask


def compare_attention_outputs(
    output1: torch.Tensor,
    output2: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3
) -> dict:
    """Compare outputs from two attention implementations.

    Args:
        output1: Output from first attention function
        output2: Output from second attention function
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        Dictionary with comparison results
    """
    # Compute differences
    abs_diff = torch.abs(output1 - output2)
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    # Check if outputs are close
    are_close = torch.allclose(output1, output2, rtol=rtol, atol=atol)

    # Compute relative error
    rel_error = abs_diff / (torch.abs(output1) + 1e-8)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    return {
        'are_close': are_close,
        'max_abs_diff': max_abs_diff,
        'mean_abs_diff': mean_abs_diff,
        'max_rel_error': max_rel_error,
        'mean_rel_error': mean_rel_error,
    }


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Configuration
    num_heads = 8
    num_kv_heads = 8
    head_size = 64
    total_tokens = 32
    total_kv_tokens = 128
    scale = 1.0 / (head_size ** 0.5)

    print("=" * 80)
    print("Attention Implementation Comparison Test")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  num_heads: {num_heads}")
    print(f"  num_kv_heads: {num_kv_heads}")
    print(f"  head_size: {head_size}")
    print(f"  total_tokens: {total_tokens}")
    print(f"  total_kv_tokens: {total_kv_tokens}")
    print(f"  scale: {scale:.6f}")
    print()

    # Create debugger instance
    # To use Spyre device, set spyre_device='spyre' (or appropriate device string)
    # For now, using None (CPU) as default - change to 'spyre' when Spyre is available
    use_spyre = True  # Set to True to enable Spyre device
    spyre_device = 'spyre' if use_spyre else None

    debugger = AttentionDebugger(
        num_heads=num_heads,
        head_size=head_size,
        scale=scale,
        num_kv_heads=num_kv_heads,
        working_precision=torch.float32,
        spyre_device=spyre_device
    )

    if use_spyre:
        print(f"Using Spyre device: {spyre_device}")
    else:
        print("Using CPU device (set use_spyre=True to enable Spyre)")
    print()

    # Generate test inputs
    print("Generating test inputs...")
    query, key, value, attn_metadata, block_mask = generate_test_inputs(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        total_tokens=total_tokens,
        total_kv_tokens=total_kv_tokens,
        dtype=torch.float16,
        device=torch.device('cpu')
    )

    print(f"Input shapes:")
    print(f"  query: {query.shape}")
    print(f"  key: {key.shape}")
    print(f"  value: {value.shape}")
    print()

    # Compute attention using both methods
    print("Computing attention with method 1...")
    output1 = debugger._compute_attention1(
        query.clone(),
        key.clone(),
        value.clone(),
        attn_metadata,
        block_mask
    )
    print(f"Output1 shape: {output1.shape}")

    print("\nComputing attention with method 2...")
    output2 = debugger._compute_attention2(
        query.clone(),
        key.clone(),
        value.clone(),
        attn_metadata,
        block_mask
    )
    print(f"Output2 shape: {output2.shape}")

    # Compare outputs
    print("\n" + "=" * 80)
    print("Comparison Results")
    print("=" * 80)

    # Check for NaN/Inf in outputs
    print(f"\nOutput1 checks:")
    print(f"  Shape: {output1.shape}")
    print(f"  Has NaN: {torch.isnan(output1).any().item()}")
    print(f"  Has Inf: {torch.isinf(output1).any().item()}")
    print(f"  Min: {output1.min().item():.6f}, Max: {output1.max().item():.6f}, Mean: {output1.mean().item():.6f}")
    print(f"  Num zeros: {(output1 == 0).sum().item()}")

    print(f"\nOutput2 checks:")
    print(f"  Shape: {output2.shape}")
    print(f"  Has NaN: {torch.isnan(output2).any().item()}")
    print(f"  Has Inf: {torch.isinf(output2).any().item()}")
    print(f"  Min: {output2.min().item():.6f}, Max: {output2.max().item():.6f}, Mean: {output2.mean().item():.6f}")
    print(f"  Num zeros: {(output2 == 0).sum().item()}")

    comparison = compare_attention_outputs(output1, output2, rtol=1e-2, atol=1e-2)

    print(f"\nComparison metrics:")
    print(f"  Outputs are close: {comparison['are_close']}")
    print(f"  Max absolute difference: {comparison['max_abs_diff']:.6e}")
    print(f"  Mean absolute difference: {comparison['mean_abs_diff']:.6e}")
    print(f"  Max relative error: {comparison['max_rel_error']:.6e}")
    print(f"  Mean relative error: {comparison['mean_rel_error']:.6e}")

    if comparison['are_close']:
        print("\n[PASS] TEST PASSED: Both implementations produce similar results!")
    else:
        print("\n[FAIL] TEST FAILED: Implementations produce different results!")
        print("\nSample outputs (first 10 elements):")
        print(f"  Output1: {output1.flatten()[:10]}")
        print(f"  Output2: {output2.flatten()[:10]}")
        print("\nSample outputs (elements 100-110):")
        print(f"  Output1: {output1.flatten()[100:110]}")
        print(f"  Output2: {output2.flatten()[100:110]}")

        # Check per-head statistics
        print("\nPer-head statistics:")
        for h in range(min(num_heads, 3)):  # Show first 3 heads
            out1_h = output1[:, h, :]
            out2_h = output2[:, h, :]
            diff_h = torch.abs(out1_h - out2_h)
            print(f"  Head {h}:")
            print(f"    Output1 mean: {out1_h.mean().item():.6f}, std: {out1_h.std().item():.6f}")
            print(f"    Output2 mean: {out2_h.mean().item():.6f}, std: {out2_h.std().item():.6f}")
            print(f"    Max diff: {diff_h.max().item():.6f}, Mean diff: {diff_h.mean().item():.6f}")
            print(f"    Output2 zeros: {(out2_h == 0).sum().item()}/{out2_h.numel()}")

    print("=" * 80)
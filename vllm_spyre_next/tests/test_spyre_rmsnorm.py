"""Tests for SpyreRMSNorm custom operation.

This test suite validates the SpyreRMSNorm implementation against the reference
forward_static implementation, inspired by vLLM's test_layernorm.py.
"""

import pytest
import torch

from vllm_spyre_next.custom_ops.rms_norm import SpyreRMSNorm


# Test parameters inspired by vLLM's test_layernorm.py
NUM_TOKENS = [7, 83, 2048]
HIDDEN_SIZES = [64, 128, 4096, 5120, 8192]
ADD_RESIDUAL = [False, True]
DTYPES = [torch.bfloat16, torch.float16]
SEEDS = [0]


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.mark.spyre
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_spyre_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
) -> None:
    """Test SpyreRMSNorm against reference implementation.

    This test validates that SpyreRMSNorm produces results consistent with
    the reference forward_static implementation, accounting for:
    - Spyre device constraints (min batch size 64)
    - Dtype conversions (float16 on device, bfloat16 output)
    - Numerical precision differences

    Args:
        num_tokens: Number of tokens in the batch
        hidden_size: Hidden dimension size
        add_residual: Whether to test with residual connection
        dtype: Data type for tensors
        seed: Random seed for reproducibility
    """
    set_random_seed(seed)
    
    # Initialize SpyreRMSNorm layer
    layer = SpyreRMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Create input tensors
    scale = 1 / (2 * hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype).to("spyre")
    x = x * scale if add_residual else x
    residual = torch.randn_like(x) * scale if add_residual else None

    # Execute reference implementation (forward_static)
    # NOTE: Reference should be executed first as custom kernel may be in-place
    ref_out = SpyreRMSNorm.forward_static(
        x.clone(),
        variance_epsilon=layer.variance_epsilon,
        hidden_size=layer.hidden_size,
        weight=layer.weight,
        residual=residual.clone() if residual is not None else None,
        variance_size_override=layer.variance_size_override,
    )

    # Execute SpyreRMSNorm implementation
    out = layer.forward_native(x, residual)

    # Validate results
    # NOTE: RMSNorm operators typically have larger numerical errors due to
    # reductions. SpyreRMSNorm has additional errors from:
    # - Device transfers (CPU -> Spyre -> CPU)
    # - Dtype conversions (bfloat16 -> float16 -> bfloat16)
    # - Batch padding (for batches < 64)
    # Therefore, we use relaxed tolerances
    if add_residual:
        torch.testing.assert_close(out[0], ref_out[0], atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(out[1], ref_out[1], atol=1e-1, rtol=1e-1)
    else:
        torch.testing.assert_close(out, ref_out, atol=1e-1, rtol=1e-1)


@pytest.mark.spyre
@pytest.mark.parametrize("batch_size", [1, 32, 63, 64, 128])
@pytest.mark.parametrize("hidden_size", [128, 4096])
def test_spyre_rms_norm_batch_padding(
    default_vllm_config,
    batch_size: int,
    hidden_size: int,
) -> None:
    """Test SpyreRMSNorm batch padding behavior.

    Validates that SpyreRMSNorm correctly handles batches smaller than the
    minimum Spyre batch size (64) by padding and trimming appropriately.

    Args:
        batch_size: Number of tokens in the batch
        hidden_size: Hidden dimension size
    """
    torch.set_default_device("cpu")

    # Initialize layer
    layer = SpyreRMSNorm(hidden_size).to(dtype=torch.bfloat16)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Create input
    x = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)

    # Execute
    out = layer.forward_native(x, residual=None)

    # Validate output shape matches input shape (padding should be trimmed)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"

    # Validate output is on CPU and in correct dtype
    assert out.device.type == "cpu", f"Expected CPU device, got {out.device}"
    assert out.dtype == torch.bfloat16, f"Expected bfloat16, got {out.dtype}"


@pytest.mark.spyre
def test_spyre_rms_norm_strided_input(default_vllm_config):
    """Test SpyreRMSNorm with strided (non-contiguous) input.

    Validates that SpyreRMSNorm handles non-contiguous tensors correctly,
    similar to the upstream vLLM test.
    """
    torch.set_default_device("cpu")

    hidden_size = 128
    num_tokens = 64
    last_dim = 2 * hidden_size

    # Initialize layer
    layer = SpyreRMSNorm(hidden_size).to(dtype=torch.bfloat16)
    layer.weight.data.normal_(mean=1.0, std=0.1)

    # Create strided input (non-contiguous)
    x = torch.randn(num_tokens, last_dim, dtype=torch.bfloat16)
    x = x[..., :hidden_size]
    assert not x.is_contiguous(), "Input should be non-contiguous for this test"

    # Execute - should handle non-contiguous input
    out = layer.forward_native(x, residual=None)

    # Validate output shape
    assert out.shape == (num_tokens, hidden_size)


@pytest.mark.spyre
def test_spyre_rms_norm_variance_size_override_not_implemented(default_vllm_config):
    """Test that variance_size_override raises NotImplementedError.

    SpyreRMSNorm currently does not support variance_size_override parameter.
    """
    torch.set_default_device("cpu")

    hidden_size = 128
    num_tokens = 64

    # Initialize layer with variance_size_override
    layer = SpyreRMSNorm(hidden_size, variance_epsilon=1e-5).to(dtype=torch.bfloat16)
    layer.variance_size_override = 64  # Set override

    x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)

    # Should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="variance_size_override not yet implemented"):
        layer.forward_native(x, residual=None)


@pytest.mark.spyre
def test_spyre_rms_norm_custom_op_registration(default_vllm_config):
    """Test that the custom op is properly registered.

    Validates that torch.ops.vllm.spyre_rmsnorm is available and callable.
    """
    # Check that custom op exists
    assert hasattr(torch.ops.vllm, "spyre_rmsnorm"), "Custom op not registered"

    # Validate it's callable
    assert callable(torch.ops.vllm.spyre_rmsnorm), "Custom op not callable"

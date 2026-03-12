"""Tests for SpyreRMSNorm custom operation.

This test suite validates the SpyreRMSNorm implementation against the reference
forward_static implementation, inspired by vLLM's test_layernorm.py.
"""

import pytest
import torch

from vllm.model_executor.layers.layernorm import RMSNorm
from vllm_spyre_next.custom_ops.rms_norm import SpyreRMSNorm


# Test parameters inspired by vLLM's test_layernorm.py
# NUM_TOKENS = [7, 83, 2048]
# HIDDEN_SIZES = [64, 128, 4096, 5120, 8192]
# ADD_RESIDUAL = [False, True]
# DTYPES = [torch.bfloat16, torch.float16]
# SEEDS = [0]
NUM_TOKENS = [1024]
HIDDEN_SIZES = [4096]
ADD_RESIDUAL = [False, True]
DTYPES = [torch.bfloat16]
SEEDS = [0]


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)


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
    the reference vllm RMSNorm implementation running on cpu.

    Args:
        num_tokens: Number of tokens in the batch
        hidden_size: Hidden dimension size
        add_residual: Whether to test with residual connection
        dtype: Data type for tensors
        seed: Random seed for reproducibility
    """
    torch.set_default_device("cpu")
    set_random_seed(seed)

    # Initialize SpyreRMSNorm layer
    layer = SpyreRMSNorm(hidden_size)

    # Create input tensors
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    scale = 1 / (2 * hidden_size)
    x = x * scale if add_residual else x
    residual = torch.randn_like(x) * scale if add_residual else None

    # Execute reference implementation on cpu
    ref_out = RMSNorm.forward_static(
        x=x,
        variance_epsilon=1e-6,
        hidden_size=hidden_size,
        orig_dtype=dtype,
        weight=layer.weight.clone(),
        residual=residual,
        variance_size_override=None,
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
def test_spyre_rms_norm_custom_op_registration(default_vllm_config):
    """Test that the custom op is properly registered."""
    rms_layer = RMSNorm(1024)
    assert isinstance(rms_layer, SpyreRMSNorm)

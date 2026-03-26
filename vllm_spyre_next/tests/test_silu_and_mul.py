"""
Test SpyreSiluAndMul custom op correctness against a reference implementation.
"""

import pytest
import torch
import torch.nn.functional as F


def reference_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Golden reference: standard SiluAndMul (SwiGLU) in PyTorch.

    Computes: silu(x[..., :d]) * x[..., d:] where d = x.shape[-1] // 2
    """
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return F.silu(x1) * x2


@pytest.mark.spyre
@pytest.mark.siluandmul
@pytest.mark.parametrize("num_tokens", [1, 7, 63, 64, 65, 1024])
@pytest.mark.parametrize("d", [2, 63, 64, 65, 1024, 13824])
def test_spyre_siluandmul_matches_reference(default_vllm_config, num_tokens, d):
    """SpyreSiluAndMul output matches golden reference.

    Tests both paths:
    - forward(): custom op dispatch (no-compile path via torch.ops.vllm.spyre_siluandmul)
    - forward_native(): direct Spyre device execution
    """
    from vllm_spyre_next.custom_ops.silu_and_mul import SpyreSiluAndMul

    torch.manual_seed(42)

    # Input shape is [num_tokens, 2*d], output shape is [num_tokens, d]
    x = torch.randn(num_tokens, 2 * d)
    layer = SpyreSiluAndMul()

    expected = reference_silu_and_mul(x)
    actual = layer.forward_native(x)

    torch.testing.assert_close(actual, expected, atol=1e-1, rtol=1e-1)


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 256, dtype=torch.float32)  # 2*d=256, so d=128


def mock_forward_native(x):
    """Mock: return ones with output shape (halves last dim)."""
    d = x.shape[-1] // 2
    return torch.ones(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)


@pytest.mark.spyre
@pytest.mark.siluandmul
def test_siluandmul_oot_dispatch(default_vllm_config, monkeypatch, dummy_tensor):
    """Verify SiluAndMul OOT registration: class swap and forward_oot routing."""
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm_spyre_next.custom_ops.silu_and_mul import SpyreSiluAndMul

    layer = SiluAndMul()

    # OOT class swap: SiluAndMul.__new__ should produce SpyreSiluAndMul
    assert isinstance(layer, SpyreSiluAndMul)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot

    # Mock forward_native (called by forward_oot) with a known transform
    monkeypatch.setattr(layer, "forward_native", mock_forward_native)
    out = layer.forward_oot(dummy_tensor)

    # Expected: ones with shape [4, 128] (halved last dim)
    expected = torch.ones(4, 128, dtype=dummy_tensor.dtype)
    assert torch.allclose(out, expected)

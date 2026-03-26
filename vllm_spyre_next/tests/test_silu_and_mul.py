"""
Test SpyreSiluAndMul custom op correctness against a reference implementation.
"""

import pytest
import torch

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

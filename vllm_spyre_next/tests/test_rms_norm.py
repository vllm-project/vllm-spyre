"""
Test SpyreRMSNorm custom op correctness against a reference implementation.
"""

import pytest
import torch

def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    """Golden reference: standard RMSNorm in PyTorch."""
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight.float()
    return x_normed

@pytest.fixture()
def default_vllm_config(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.platforms import PlatformEnum, current_platform
    from vllm.plugins import load_general_plugins

    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)
    load_general_plugins()
    from vllm_spyre_next.custom_ops import register_all
    register_all()

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    with set_current_vllm_config(config):
        yield

@pytest.mark.spyre
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("hidden_size", [128, 512])
def test_spyre_rmsnorm_matches_reference(default_vllm_config, batch_size, hidden_size):
    """SpyreRMSNorm forward_native output matches golden reference."""
    from vllm_spyre_next.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=torch.float16)
    layer = SpyreRMSNorm(hidden_size, eps=eps)

    expected = reference_rms_norm(x, layer.weight.data, eps)
    actual = layer.forward(x)

    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.spyre
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("hidden_size", [63, 64, 65, 127, 128, 129, 256, 512])
def test_spyre_rmsnorm_on_device(default_vllm_config, batch_size, hidden_size):
    """SpyreRMSNorm full forward pass on Spyre hardware."""
    from vllm_spyre_next.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=torch.float16)
    layer = SpyreRMSNorm(hidden_size, eps=eps)

    expected = reference_rms_norm(x, layer.weight.data, eps)
    actual = layer.forward_native(x)


    torch.testing.assert_close(actual.float(), expected.float(), atol=1e-2, rtol=1e-2)

@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 128, dtype=torch.float32)


def mock_forward_native_no_residual(x, residual=None):
    """Mock: return x + 1 (no residual path)."""
    return x + 1


def mock_forward_native_with_residual(x, residual=None):
    """Mock: return (2 * x, 2 * residual) (residual path)."""
    return 2 * x, 2 * residual


@pytest.mark.cpu
@pytest.mark.parametrize("residual", [None, torch.randn(4, 128, dtype=torch.float32)])
def test_rmsnorm_oot_dispatch(default_vllm_config, monkeypatch, dummy_tensor, residual):
    """Verify RMSNorm OOT registration: class swap and forward_oot routing."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm_spyre_next.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    # OOT class swap: RMSNorm.__new__ should produce SpyreRMSNorm
    assert isinstance(layer, SpyreRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot

    # Mock forward_native (called by forward_oot) with a known transform
    if residual is not None:
        monkeypatch.setattr(layer, "forward_native", mock_forward_native_with_residual)
        out_x, out_residual = layer.forward_oot(dummy_tensor, residual)

        assert torch.allclose(out_x, 2 * dummy_tensor)
        assert torch.allclose(out_residual, 2 * residual)
    else:
        monkeypatch.setattr(layer, "forward_native", mock_forward_native_no_residual)
        out_x = layer.forward_oot(dummy_tensor, residual)

        assert torch.allclose(out_x, dummy_tensor + 1)

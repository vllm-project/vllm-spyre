"""Shared fixtures for vllm_spyre_next tests."""

import pytest


@pytest.fixture()
def default_vllm_config(monkeypatch):
    """Set a default VllmConfig for tests that directly test CustomOps.

    Mirrors the fixture from vLLM's tests/conftest.py — required because
    CustomOp.__init__ calls get_current_vllm_config() to resolve dispatch.

    Also ensures the platform is detected as OOT (matching
    TorchSpyrePlatform._enum = PlatformEnum.OOT) so that
    dispatch_forward selects forward_oot.
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.platforms import PlatformEnum, current_platform

    # Ensure platform dispatches as OOT, matching TorchSpyrePlatform
    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    with set_current_vllm_config(config):
        yield

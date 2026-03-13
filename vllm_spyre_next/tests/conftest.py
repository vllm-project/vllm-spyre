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
    from vllm.plugins import load_general_plugins
    from vllm.forward_context import ForwardContext, override_forward_context

    # Ensure platform dispatches as OOT, matching TorchSpyrePlatform
    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)
    load_general_plugins()
    
    # Explicitly register custom ops
    from vllm_spyre_next.custom_ops import register_all
    register_all()

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    with set_current_vllm_config(config):
        # Set up forward context so custom ops can find layers in no_compile_layers
        forward_ctx = ForwardContext(
            no_compile_layers=config.compilation_config.static_forward_context,
            attn_metadata={},
            slot_mapping={},
            virtual_engine=0,
        )
        with override_forward_context(forward_ctx):
            yield
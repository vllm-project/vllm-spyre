"""Shared fixtures for vllm_spyre_next tests."""

import pytest


@pytest.fixture()
def default_vllm_config():
    """Set a default VllmConfig for tests that directly test CustomOps.

    Mirrors the fixture from vLLM's tests/conftest.py — required because
    CustomOp.__init__ calls get_current_vllm_config() to resolve dispatch.
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        yield

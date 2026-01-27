"""Tests for ModelConfigurator - configuration application logic."""

import logging
import os
import pytest
from unittest.mock import Mock, patch

from vllm_spyre import envs as envs_spyre
from vllm_spyre.config.model_config import DeviceConfig, ModelConfig
from vllm_spyre.config.configurators.model_configurator import (
    ConfigurationSummary,
    ModelConfigurator,
)
from vllm_spyre.config.configurators import model_configurator

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def log_capture(caplog):
    """Fixture to setup log capture for model_configurator logger."""
    caplog.set_level(logging.DEBUG)
    model_configurator.logger.setLevel(logging.DEBUG)
    if caplog.handler not in model_configurator.logger.handlers:
        model_configurator.logger.addHandler(caplog.handler)
    return caplog


@pytest.fixture
def model_config():
    """Fixture providing a basic ModelConfig."""
    return ModelConfig(
        name="test-model",
        architecture=Mock(model_name="test-model", model_type="granite"),
    )


@pytest.fixture
def vllm_config():
    """Fixture providing a mock VllmConfig."""
    config = Mock()
    config.parallel_config = Mock(world_size=4)
    config.cache_config = Mock(num_gpu_blocks_override=None)
    config.scheduler_config = Mock(max_num_batched_tokens=None)
    config.model_config = Mock(max_model_len=8192)
    return config


class TestEnvironmentVariables:
    """Tests for environment variable configuration."""

    def test_set_env_var_when_not_set(self, monkeypatch, model_config):
        """Test setting environment variable when not already set."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        result = configurator.set_env_var("TEST_VAR", "123")

        assert result is True
        assert os.getenv("TEST_VAR") == "123"

    def test_skip_when_already_set_to_same_value(self, monkeypatch, model_config):
        """Test that setting is skipped when already set to same value."""
        monkeypatch.setenv("TEST_VAR", "123")

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        result = configurator.set_env_var("TEST_VAR", "123")

        assert result is False
        assert os.getenv("TEST_VAR") == "123"

    def test_log_warning_when_different_value_no_override(
        self, monkeypatch, model_config, log_capture
    ):
        """Test warning logged when already set to different value (no override)."""
        monkeypatch.setenv("TEST_VAR", "456")
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "0")

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        result = configurator.set_env_var("TEST_VAR", "123")

        assert result is False
        assert os.getenv("TEST_VAR") == "456"  # Not overridden
        assert "not overriding" in log_capture.text.lower()

    def test_raise_error_when_require_known_config_and_conflict(self, monkeypatch, model_config):
        """Test RuntimeError when VLLM_SPYRE_REQUIRE_KNOWN_CONFIG=1 and conflict."""
        monkeypatch.setenv("TEST_VAR", "456")
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "1")

        # Clear the env cache to pick up the new value
        envs_spyre.clear_env_cache()

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        with pytest.raises(RuntimeError, match="VLLM_SPYRE_REQUIRE_KNOWN_CONFIG"):
            configurator.set_env_var("TEST_VAR", "123")

    def test_set_multiple_env_vars_from_device_config(self, monkeypatch, model_config, vllm_config):
        """Test setting multiple environment variables from device_config."""
        monkeypatch.delenv("VAR1", raising=False)
        monkeypatch.delenv("VAR2", raising=False)
        monkeypatch.delenv("VAR3", raising=False)

        device_config = DeviceConfig(
            tp_size=4,
            env_vars={"VAR1": "value1", "VAR2": "value2", "VAR3": "value3"},
        )
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert os.getenv("VAR1") == "value1"
        assert os.getenv("VAR2") == "value2"
        assert os.getenv("VAR3") == "value3"
        assert summary.env_vars == {"VAR1": "value1", "VAR2": "value2", "VAR3": "value3"}

    def test_configuration_summary_tracks_env_vars(self, monkeypatch, model_config, vllm_config):
        """Test that ConfigurationSummary tracks environment variables."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert summary.model_name == "test-model"
        assert summary.tp_size == 4
        assert summary.env_vars == {"TEST_VAR": "123"}


class TestGPUBlocksOverride:
    """Tests for GPU blocks override configuration."""

    def test_apply_integer_override_when_not_set(self, model_config, vllm_config):
        """Test applying integer override when cache_config.num_gpu_blocks_override is None."""
        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=1000)
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.cache_config.num_gpu_blocks_override == 1000
        assert summary.num_blocks == 1000

    @pytest.mark.parametrize(
        "sendnn_configured,sendnn_version,expected_blocks",
        [
            (True, (1, 0, 2), 800),  # Old version uses torch_sendnn_lt_1_0_3
            (True, (1, 0, 3), 1000),  # New version uses default
            (True, (1, 1, 0), 1000),  # Newer version uses default
            (False, None, 1000),  # Not configured uses default
        ],
        ids=["old_sendnn", "sendnn_1.0.3", "newer_sendnn", "not_configured"],
    )
    @patch("vllm_spyre.platform.SpyrePlatform")
    def test_version_aware_gpu_blocks_override(
        self,
        mock_platform,
        model_config,
        vllm_config,
        sendnn_configured,
        sendnn_version,
        expected_blocks,
    ):
        """Test version-aware GPU blocks override selection."""
        mock_platform.sendnn_configured.return_value = sendnn_configured
        mock_platform.sendnn_version.return_value = sendnn_version

        device_config = DeviceConfig(
            tp_size=4,
            num_gpu_blocks_override={
                "torch_sendnn_lt_1_0_3": 800,
                "default": 1000,
            },
        )
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.cache_config.num_gpu_blocks_override == expected_blocks
        assert summary.num_blocks == expected_blocks

    def test_log_warning_when_user_set_different_value(
        self, monkeypatch, model_config, vllm_config, log_capture
    ):
        """Test warning when user already set --num-gpu-blocks-override to different value."""
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "0")
        envs_spyre.clear_env_cache()

        vllm_config.cache_config.num_gpu_blocks_override = 500  # User set this

        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=1000)
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.cache_config.num_gpu_blocks_override == 500  # Not changed
        assert summary.num_blocks == 1000  # But summary shows what we wanted
        assert "not using model default" in log_capture.text.lower()

    def test_raise_error_when_require_known_config_and_user_override_conflicts(
        self, monkeypatch, model_config, vllm_config
    ):
        """Test RuntimeError when VLLM_SPYRE_REQUIRE_KNOWN_CONFIG=1 and user override conflicts."""
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "1")
        envs_spyre.clear_env_cache()

        vllm_config.cache_config.num_gpu_blocks_override = 500  # User set this

        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=1000)
        configurator = ModelConfigurator(model_config, device_config)

        with pytest.raises(RuntimeError, match="VLLM_SPYRE_REQUIRE_KNOWN_CONFIG"):
            configurator.configure(vllm_config)

    def test_skip_when_num_gpu_blocks_override_is_none(self, model_config, vllm_config):
        """Test that GPU blocks override is skipped when device_config value is None."""
        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=None)
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.cache_config.num_gpu_blocks_override is None
        assert summary.num_blocks is None

    @patch("vllm_spyre.platform.SpyrePlatform")
    def test_gpu_blocks_dict_with_no_matching_key(self, mock_platform, model_config, vllm_config):
        """Test that None is returned when dict has no matching key."""
        mock_platform.sendnn_configured.return_value = False

        device_config = DeviceConfig(
            tp_size=4,
            num_gpu_blocks_override={"some_other_key": 1000},  # No 'default' key
        )
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert summary.num_blocks is None
        assert vllm_config.cache_config.num_gpu_blocks_override is None

    def test_gpu_blocks_when_user_value_matches(self, model_config, vllm_config):
        """Test that override value is returned when user already set to same value."""
        vllm_config.cache_config.num_gpu_blocks_override = 1000  # User set to same value

        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=1000)
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert summary.num_blocks == 1000
        assert vllm_config.cache_config.num_gpu_blocks_override == 1000


class TestChunkedPrefill:
    """Tests for chunked prefill configuration."""

    def test_apply_max_num_batched_tokens_when_enabled(
        self, monkeypatch, model_config, vllm_config
    ):
        """Test applying max_num_batched_tokens when VLLM_SPYRE_USE_CHUNKED_PREFILL=1."""
        monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "1")
        monkeypatch.delenv("VLLM_DT_CHUNK_LEN", raising=False)
        envs_spyre.clear_env_cache()

        device_config = DeviceConfig(
            tp_size=4,
            chunked_prefill_config={"max_num_batched_tokens": 2048},
        )
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.scheduler_config.max_num_batched_tokens == 2048
        assert summary.chunk_size == 2048

    @pytest.mark.parametrize(
        "use_cp,chunk_len_set,cp_config,expected_tokens",
        [
            ("0", False, {"max_num_batched_tokens": 2048}, None),  # CP disabled
            ("1", True, {"max_num_batched_tokens": 2048}, None),  # VLLM_DT_CHUNK_LEN set
            ("1", False, None, None),  # Config is None
            ("1", False, {"max_num_batched_tokens": 2048}, 2048),  # Should apply
            ("1", False, {"other_key": "value"}, None),  # Config missing max_num_batched_tokens
            ("1", False, {"max_num_batched_tokens": 0}, None),  # max_num_batched_tokens is 0
        ],
        ids=["cp_disabled", "chunk_len_set", "config_none", "applies", "missing_key", "zero_value"],
    )
    def test_chunked_prefill_conditions(
        self,
        monkeypatch,
        model_config,
        vllm_config,
        use_cp,
        chunk_len_set,
        cp_config,
        expected_tokens,
    ):
        """Test various chunked prefill skip conditions and application."""
        monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", use_cp)
        if chunk_len_set:
            monkeypatch.setenv("VLLM_DT_CHUNK_LEN", "1024")
        else:
            monkeypatch.delenv("VLLM_DT_CHUNK_LEN", raising=False)
        envs_spyre.clear_env_cache()

        device_config = DeviceConfig(tp_size=4, chunked_prefill_config=cp_config)
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.scheduler_config.max_num_batched_tokens == expected_tokens
        assert summary.chunk_size == expected_tokens


class TestNoDeviceConfig:
    """Tests for behavior when device_config is None."""

    def test_return_empty_summary_when_no_device_config(self, model_config, vllm_config):
        """Test returning empty ConfigurationSummary when device_config is None."""
        configurator = ModelConfigurator(model_config, device_config=None)

        summary = configurator.configure(vllm_config)

        assert isinstance(summary, ConfigurationSummary)
        assert summary.model_name == "test-model"
        assert summary.tp_size == 4
        assert summary.env_vars == {}
        assert summary.num_blocks is None
        assert summary.chunk_size is None

    def test_log_debug_message_when_no_device_config(self, model_config, vllm_config, log_capture):
        """Test that debug message is logged when no device config."""
        configurator = ModelConfigurator(model_config, device_config=None)

        configurator.configure(vllm_config)

        assert "no device configuration" in log_capture.text.lower()


# Made with Bob

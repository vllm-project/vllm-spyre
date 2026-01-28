"""Tests for ModelConfigurator - configuration application logic."""

import logging
import os
import pytest
from unittest.mock import Mock, patch

from vllm_spyre import envs as envs_spyre
from vllm_spyre.config.model_config import DeviceConfig, ModelConfig
from vllm_spyre.config.configurators.model_configurator import (
    ConfigValue,
    ConfigurationSummary,
    ModelConfigurator,
)

pytestmark = pytest.mark.skip_global_cleanup


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

        assert isinstance(result, ConfigValue)
        assert result.expected == "123"
        assert result.actual == "123"
        assert result.was_overridden() is False
        assert os.getenv("TEST_VAR") == "123"

    def test_skip_when_already_set_to_same_value(self, monkeypatch, model_config):
        """Test that setting is skipped when already set to same value."""
        monkeypatch.setenv("TEST_VAR", "123")

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        result = configurator.set_env_var("TEST_VAR", "123")

        assert isinstance(result, ConfigValue)
        assert result.expected == "123"
        assert result.actual == "123"
        assert result.was_overridden() is False
        assert os.getenv("TEST_VAR") == "123"

    def test_log_warning_when_different_value_no_override(
        self, monkeypatch, model_config, caplog_vllm_spyre
    ):
        """Test warning logged when already set to different value (no override)."""
        monkeypatch.setenv("TEST_VAR", "456")
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "0")

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        result = configurator.set_env_var("TEST_VAR", "123")

        assert isinstance(result, ConfigValue)
        assert result.expected == "123"
        assert result.actual == "456"
        assert result.was_overridden() is True
        assert os.getenv("TEST_VAR") == "456"  # Not overridden
        assert "not using model default" in caplog_vllm_spyre.text.lower()

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
        assert "VAR1" in summary.env_vars
        assert "VAR2" in summary.env_vars
        assert "VAR3" in summary.env_vars
        assert summary.env_vars["VAR1"] == "value1"
        assert summary.env_vars["VAR2"] == "value2"
        assert summary.env_vars["VAR3"] == "value3"

    def test_configuration_summary_tracks_env_vars(self, monkeypatch, model_config, vllm_config):
        """Test that ConfigurationSummary tracks environment variables."""
        monkeypatch.delenv("TEST_VAR", raising=False)

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": "123"})
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert summary.model_name == "test-model"
        assert summary.tp_size == 4
        assert "TEST_VAR" in summary.env_vars
        assert summary.env_vars["TEST_VAR"] == "123"


class TestGPUBlocksOverride:
    """Tests for GPU blocks override configuration."""

    def test_apply_integer_override_when_not_set(self, model_config, vllm_config):
        """Test applying integer override when cache_config.num_gpu_blocks_override is None."""
        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=1000)
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.cache_config.num_gpu_blocks_override == 1000
        assert summary.num_blocks.actual == 1000
        assert summary.num_blocks.was_overridden() is False

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
        assert summary.num_blocks.actual == expected_blocks
        assert summary.num_blocks.was_overridden() is False

    def test_log_warning_when_user_set_different_value(
        self, monkeypatch, model_config, vllm_config, caplog_vllm_spyre
    ):
        """Test warning when user already set --num-gpu-blocks-override to different value."""
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "0")
        envs_spyre.clear_env_cache()

        vllm_config.cache_config.num_gpu_blocks_override = 500  # User set this

        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=1000)
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        assert vllm_config.cache_config.num_gpu_blocks_override == 500  # Not changed
        assert summary.num_blocks.expected == 1000  # Summary shows what we wanted
        assert summary.num_blocks.actual == 500  # And what was actually set
        assert summary.num_blocks.was_overridden() is True
        assert "not using model default" in caplog_vllm_spyre.text.lower()

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

        assert summary.num_blocks.actual == 1000
        assert summary.num_blocks.was_overridden() is False
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
        assert summary.chunk_size.actual == 2048
        assert summary.chunk_size.was_overridden() is False

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

        if expected_tokens is None:
            assert vllm_config.scheduler_config.max_num_batched_tokens == expected_tokens
            # When chunk_len_set=True, we now return a ConfigValue tracking the override
            if chunk_len_set and cp_config and cp_config.get("max_num_batched_tokens"):
                assert summary.chunk_size is not None
                assert summary.chunk_size.expected == cp_config["max_num_batched_tokens"]
                assert summary.chunk_size.actual == 1024
                assert summary.chunk_size.was_overridden() is True
            else:
                assert summary.chunk_size is None
        else:
            assert vllm_config.scheduler_config.max_num_batched_tokens == expected_tokens
            assert summary.chunk_size.actual == expected_tokens
            assert summary.chunk_size.was_overridden() is False


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

    def test_log_debug_message_when_no_device_config(
        self, model_config, vllm_config, caplog_vllm_spyre
    ):
        """Test that debug message is logged when no device config."""
        configurator = ModelConfigurator(model_config, device_config=None)

        configurator.configure(vllm_config)

        assert "no device configuration" in caplog_vllm_spyre.text.lower()


class TestConfigValue:
    """Tests for ConfigValue dataclass."""

    @pytest.mark.parametrize(
        "expected,actual,should_be_overridden",
        [
            ("100", "200", True),  # Different strings
            ("100", "100", False),  # Same strings
            (1000, 500, True),  # Different ints
            (1000, 1000, False),  # Same ints
            (None, None, False),  # Both None
            ("100", None, True),  # Expected set, actual None
        ],
        ids=["diff_str", "same_str", "diff_int", "same_int", "both_none", "expected_only"],
    )
    def test_was_overridden(self, expected, actual, should_be_overridden):
        """Test was_overridden method with various value combinations."""
        config_value = ConfigValue(expected=expected, actual=actual)
        assert config_value.was_overridden() is should_be_overridden


class TestOverrideTracking:
    """Tests for override tracking in configuration methods."""

    @pytest.mark.parametrize(
        "existing_value,expected_value,should_override",
        [
            (None, "123", False),  # Not set, should apply
            ("123", "123", False),  # Same value, no override
            ("456", "123", True),  # Different value, is overridden
        ],
        ids=["not_set", "same_value", "overridden"],
    )
    def test_set_env_var_override_tracking(
        self, monkeypatch, model_config, existing_value, expected_value, should_override
    ):
        """Test that set_env_var correctly tracks overrides."""
        if existing_value is None:
            monkeypatch.delenv("TEST_VAR", raising=False)
        else:
            monkeypatch.setenv("TEST_VAR", existing_value)
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "0")
        envs_spyre.clear_env_cache()

        device_config = DeviceConfig(tp_size=4, env_vars={"TEST_VAR": expected_value})
        configurator = ModelConfigurator(model_config, device_config)

        result = configurator.set_env_var("TEST_VAR", expected_value)

        assert isinstance(result, ConfigValue)
        assert result.expected == expected_value
        assert result.actual == (existing_value if existing_value else expected_value)
        assert result.was_overridden() is should_override

    @pytest.mark.parametrize(
        "user_value,expected_value",
        [
            (None, 1000),  # Not set by user
            (1000, 1000),  # User set to same value
            (500, 1000),  # User set to different value
        ],
        ids=["not_set", "same_value", "overridden"],
    )
    def test_configure_gpu_blocks_override_tracking(
        self, monkeypatch, model_config, vllm_config, user_value, expected_value
    ):
        """Test that GPU blocks override correctly tracks overrides."""
        monkeypatch.setenv("VLLM_SPYRE_REQUIRE_KNOWN_CONFIG", "0")
        envs_spyre.clear_env_cache()

        vllm_config.cache_config.num_gpu_blocks_override = user_value

        device_config = DeviceConfig(tp_size=4, num_gpu_blocks_override=expected_value)
        configurator = ModelConfigurator(model_config, device_config)

        # Use public configure() method instead of private _configure_gpu_blocks()
        summary = configurator.configure(vllm_config)

        assert summary.num_blocks is not None
        assert isinstance(summary.num_blocks, ConfigValue)
        assert summary.num_blocks.expected == expected_value
        assert summary.num_blocks.actual == (user_value if user_value else expected_value)
        assert summary.num_blocks.was_overridden() is (
            user_value is not None and user_value != expected_value
        )

    @pytest.mark.parametrize(
        "chunk_len_env,expected_tokens",
        [
            (None, 2048),  # Not set by user
            ("2048", 2048),  # User set to same value
            ("1024", 2048),  # User set to different value
        ],
        ids=["not_set", "same_value", "overridden"],
    )
    def test_configure_chunked_prefill_override_tracking(
        self, monkeypatch, model_config, vllm_config, chunk_len_env, expected_tokens
    ):
        """Test that chunked prefill correctly tracks overrides."""
        monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "1")
        if chunk_len_env is None:
            monkeypatch.delenv("VLLM_DT_CHUNK_LEN", raising=False)
        else:
            monkeypatch.setenv("VLLM_DT_CHUNK_LEN", chunk_len_env)
        envs_spyre.clear_env_cache()

        device_config = DeviceConfig(
            tp_size=4,
            chunked_prefill_config={"max_num_batched_tokens": expected_tokens},
        )
        configurator = ModelConfigurator(model_config, device_config)

        # Use public configure() method instead of private _configure_chunked_prefill()
        summary = configurator.configure(vllm_config)

        assert summary.chunk_size is not None
        assert isinstance(summary.chunk_size, ConfigValue)
        assert summary.chunk_size.expected == expected_tokens
        assert summary.chunk_size.actual == int(chunk_len_env) if chunk_len_env else expected_tokens
        assert summary.chunk_size.was_overridden() is (
            chunk_len_env is not None and int(chunk_len_env) != expected_tokens
        )

    def test_configuration_summary_contains_config_values(
        self, monkeypatch, model_config, vllm_config
    ):
        """Test that ConfigurationSummary contains ConfigValue objects."""
        monkeypatch.delenv("TEST_VAR", raising=False)
        monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "1")
        monkeypatch.delenv("VLLM_DT_CHUNK_LEN", raising=False)
        envs_spyre.clear_env_cache()

        device_config = DeviceConfig(
            tp_size=4,
            env_vars={"TEST_VAR": "123"},
            num_gpu_blocks_override=1000,
            chunked_prefill_config={"max_num_batched_tokens": 2048},
        )
        configurator = ModelConfigurator(model_config, device_config)

        summary = configurator.configure(vllm_config)

        # Check env_vars contains ConfigValue
        assert "TEST_VAR" in summary.env_vars
        assert isinstance(summary.env_vars["TEST_VAR"], ConfigValue)
        assert summary.env_vars["TEST_VAR"].expected == "123"
        assert summary.env_vars["TEST_VAR"].actual == "123"

        # Check num_blocks is ConfigValue
        assert isinstance(summary.num_blocks, ConfigValue)
        assert summary.num_blocks.expected == 1000
        assert summary.num_blocks.actual == 1000

        # Check chunk_size is ConfigValue
        assert isinstance(summary.chunk_size, ConfigValue)
        assert summary.chunk_size.expected == 2048
        assert summary.chunk_size.actual == 2048


class TestConfigurationSummaryLogging:
    """Tests for ConfigurationSummary.format_log_message() method"""

    def test_log_format_with_no_overrides(self, caplog_vllm_spyre):
        """Test log format when all values are applied as expected."""
        # Create a summary with no overrides
        config_summary = ConfigurationSummary(
            model_name="test-model",
            tp_size=4,
            env_vars={
                "VLLM_DT_MAX_BATCH_TKV_LIMIT": ConfigValue(expected="131072", actual="131072"),
                "FLEX_HDMA_P2PSIZE": ConfigValue(expected="268435456", actual="268435456"),
            },
            num_blocks=ConfigValue(expected=8192, actual=8192),
            chunk_size=ConfigValue(expected=1024, actual=1024),
        )

        # Actually print the log for human inspection as well
        logging.getLogger().info(config_summary.format_log_message())

        # Validate the log output
        log_text = caplog_vllm_spyre.text
        assert "Applied registry configuration for 'test-model' (TP=4):" in log_text
        assert "Environment variables:" in log_text
        assert "VLLM_DT_MAX_BATCH_TKV_LIMIT=131072 ✓" in log_text
        assert "FLEX_HDMA_P2PSIZE=268435456 ✓" in log_text
        assert "num_gpu_blocks_override=8192 ✓" in log_text
        assert "max_num_batched_tokens=1024 ✓" in log_text
        # Should not contain any override warnings
        assert "⚠" not in log_text
        assert "expected:" not in log_text

    def test_log_format_with_overrides(self, caplog_vllm_spyre):
        """Test log format when values are overridden by user."""
        # Create a summary with overrides
        config_summary = ConfigurationSummary(
            model_name="granite-3.3-8b-instruct",
            tp_size=4,
            env_vars={
                "VLLM_DT_MAX_BATCH_TKV_LIMIT": ConfigValue(expected="131072", actual="131072"),
                "FLEX_HDMA_P2PSIZE": ConfigValue(expected="268435456", actual="134217728"),
            },
            num_blocks=ConfigValue(expected=8192, actual=8192),
            chunk_size=ConfigValue(expected=1024, actual=2048),
        )

        # Actually print the log for human inspection as well
        logging.getLogger().info(config_summary.format_log_message())

        # Validate the log output
        log_text = caplog_vllm_spyre.text
        assert "Applied registry configuration for 'granite-3.3-8b-instruct' (TP=4):" in log_text
        assert "Environment variables:" in log_text

        # Check non-overridden value
        assert "VLLM_DT_MAX_BATCH_TKV_LIMIT=131072 ✓" in log_text

        # Check overridden env var
        assert "FLEX_HDMA_P2PSIZE=134217728 ⚠" in log_text
        assert "expected: 268435456" in log_text

        # Check non-overridden num_blocks
        assert "num_gpu_blocks_override=8192 ✓" in log_text

        # Check overridden chunk_size
        assert "max_num_batched_tokens=2048 ⚠" in log_text
        assert "expected: 1024" in log_text

    def test_log_format_with_mixed_overrides(self, caplog_vllm_spyre):
        """Test log format with a mix of overridden and non-overridden values."""
        # Create a summary with mixed overrides
        config_summary = ConfigurationSummary(
            model_name="test-model",
            tp_size=2,
            env_vars={
                "VAR1": ConfigValue(expected="100", actual="100"),
                "VAR2": ConfigValue(expected="200", actual="300"),
                "VAR3": ConfigValue(expected="400", actual="400"),
            },
            num_blocks=ConfigValue(expected=1000, actual=500),
            chunk_size=None,  # Not configured
        )

        # Log using caplog's logger
        logging.getLogger().info(config_summary.format_log_message())

        # Validate the log output
        log_text = caplog_vllm_spyre.text

        # Check header
        assert "Applied registry configuration for 'test-model' (TP=2):" in log_text

        # Check env vars
        assert "VAR1=100 ✓" in log_text
        assert "VAR2=300 ⚠" in log_text
        assert "expected: 200" in log_text
        assert "VAR3=400 ✓" in log_text

        # Check num_blocks override
        assert "num_gpu_blocks_override=500 ⚠" in log_text
        assert "expected: 1000" in log_text

        # chunk_size should not appear since it's None
        assert "max_num_batched_tokens" not in log_text

    def test_log_format_with_empty_config(self, caplog_vllm_spyre):
        """Test log format when no device-specific configs are present."""
        # Create an empty summary
        config_summary = ConfigurationSummary(
            model_name="test-model",
            tp_size=1,
            env_vars={},
            num_blocks=None,
            chunk_size=None,
        )

        # Actually print the log for human inspection as well
        logging.getLogger().info(config_summary.format_log_message())

        # Validate the log output
        log_text = caplog_vllm_spyre.text
        assert "Applied registry configuration for 'test-model' (TP=1):" in log_text
        assert "no device-specific configs" in log_text


# Made with Bob

"""Integration tests for the configuration system - end-to-end scenarios."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from vllm_spyre.config.configurators.model_configurator import ModelConfigurator
from .conftest import FIXTURES_PATH

pytestmark = pytest.mark.skip_global_cleanup


class TestRegistryLoading:
    """Tests for registry initialization and model loading."""

    def test_load_real_config_file(self, registry):
        """Test loading the actual model_configs.yaml file."""
        models = registry.list_models()

        # Should have at least 9 models from the config file
        # (may have more if additional models are added)
        assert len(models) >= 9

        # Verify specific models are present
        expected_models = [
            "ibm-granite/granite-3.3-8b-instruct",
            "ibm-granite/granite-3.3-8b-instruct-FP8",
            "ibm-granite/granite-4-8b-dense",
            "ibm-granite/granite-embedding-125m-english",
        ]
        for model in expected_models:
            assert model in models, f"Expected model {model} missing"


class TestModelMatching:
    """Tests for model detection and matching."""

    def test_match_granite_3_3_cb_config(self, registry, granite_3_3_hf_config, create_vllm_config):
        """Test matching granite-3.3-8b-instruct with CB config and getting configurator."""
        vllm_config = create_vllm_config(
            hf_config=granite_3_3_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)

        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-3.3-8b-instruct"
        assert configurator.device_config is not None
        assert "VLLM_DT_MAX_BATCH_TKV_LIMIT" in configurator.device_config.env_vars

    def test_match_granite_3_3_static_config(
        self, registry, granite_3_3_hf_config, create_vllm_config
    ):
        """Test matching granite-3.3-8b-instruct with static batching config."""
        vllm_config = create_vllm_config(
            hf_config=granite_3_3_hf_config,
            world_size=4,
        )

        warmup_shapes = [(6144, 2048, 1)]
        configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)

        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-3.3-8b-instruct"
        assert (
            configurator.device_config is None
        )  # Static batching configs don't have device_config

    def test_match_granite_4_cb_config(self, registry, granite_4_hf_config, create_vllm_config):
        """Test matching granite-4-8b-dense with CB config."""
        vllm_config = create_vllm_config(
            hf_config=granite_4_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)

        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-4-8b-dense"
        assert configurator.device_config is not None

    def test_embedding_models_have_no_device_configs(
        self, registry, embedding_hf_config, create_vllm_config
    ):
        """Test that embedding models don't have device_configs."""
        vllm_config = create_vllm_config(
            hf_config=embedding_hf_config,
            world_size=1,
            max_num_seqs=None,  # Static batching
        )

        warmup_shapes = [(512, 0, 64)]
        configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)

        assert configurator is not None
        assert configurator.model_config.name == "ibm-granite/granite-embedding-125m-english"
        assert configurator.device_config is None

    def test_generation_models_with_tp4_have_device_configs(
        self, registry, granite_3_3_hf_config, create_vllm_config
    ):
        """Test that generation models with TP=4 have device_configs."""
        vllm_config = create_vllm_config(
            hf_config=granite_3_3_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)

        assert configurator is not None
        assert configurator.device_config is not None
        assert configurator.device_config.env_vars is not None
        assert configurator.device_config.num_gpu_blocks_override is not None


class TestConfigurationApplication:
    """Tests for applying configuration settings."""

    @patch.dict(os.environ, {}, clear=True)
    def test_apply_configuration_sets_env_vars(
        self, registry, granite_3_3_hf_config, create_vllm_config
    ):
        """Test that applying configuration sets env vars correctly."""
        vllm_config = create_vllm_config(
            hf_config=granite_3_3_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is not None

        with patch("vllm_spyre.platform.SpyrePlatform") as mock_platform:
            mock_platform.sendnn_version.return_value = (1, 0, 3)
            mock_platform.sendnn_configured.return_value = True
            summary = configurator.configure(vllm_config)

        assert "VLLM_DT_MAX_BATCH_TKV_LIMIT" in summary.env_vars
        assert summary.env_vars["VLLM_DT_MAX_BATCH_TKV_LIMIT"] == "131072"
        assert "FLEX_HDMA_P2PSIZE" in summary.env_vars
        assert summary.env_vars["FLEX_HDMA_P2PSIZE"] == "268435456"

    @patch.dict(os.environ, {}, clear=True)
    def test_apply_configuration_sets_gpu_blocks(
        self, registry, granite_3_3_hf_config, create_vllm_config
    ):
        """Test that applying configuration sets GPU blocks correctly."""
        vllm_config = create_vllm_config(
            hf_config=granite_3_3_hf_config, world_size=4, max_model_len=32768, max_num_seqs=32
        )

        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is not None

        with patch("vllm_spyre.platform.SpyrePlatform") as mock_platform:
            mock_platform.sendnn_version.return_value = (1, 0, 3)
            mock_platform.sendnn_configured.return_value = True
            summary = configurator.configure(vllm_config)

        assert summary.num_blocks == 8192
        assert vllm_config.cache_config.num_gpu_blocks_override == 8192


class TestUnregisteredModels:
    """Tests for unregistered model handling."""

    def test_unregistered_model_returns_none(self, registry):
        """Test that unregistered model returns None from registry (no error)."""
        # Create mock vllm_config with unknown model
        vllm_config = Mock()
        hf_config = Mock(model_type="unknown_model_type")
        vllm_config.model_config = Mock(hf_config=hf_config, max_model_len=8192)
        vllm_config.parallel_config = Mock(world_size=1)
        vllm_config.scheduler_config = Mock(max_num_seqs=4)

        # Should return None, not raise error
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

    def test_micro_model_not_in_registry(self, registry):
        """Test that micro model (not in registry) returns None but doesn't error."""
        # Load micro model config
        fixture_path = FIXTURES_PATH / "ibm-ai-platform/micro-g3.3-8b-instruct-1b/config.json"
        with open(fixture_path) as f:
            config_dict = json.load(f)

        hf_config = Mock()
        for key, value in config_dict.items():
            setattr(hf_config, key, value)

        # Create mock vllm_config
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=hf_config, max_model_len=8192)
        vllm_config.parallel_config = Mock(world_size=1)
        vllm_config.scheduler_config = Mock(max_num_seqs=4)

        # Should return None (micro model not in registry)
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

    def test_log_message_when_model_not_found(self, registry, caplog_vllm_spyre):
        """Test that appropriate message is logged when model not found."""
        # Create mock vllm_config with unknown model
        vllm_config = Mock()
        hf_config = Mock(model_type="unknown_model")
        vllm_config.model_config = Mock(
            hf_config=hf_config, max_model_len=8192, model="unknown-model"
        )
        vllm_config.parallel_config = Mock(world_size=1)
        vllm_config.scheduler_config = Mock(max_num_seqs=4)

        # Try to get configurator
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None

        # Check that debug message was logged
        assert any(
            "No matching model configuration found" in record.message
            for record in caplog_vllm_spyre.records
        )


class TestPartialOverrides:
    """Tests for partial user override behavior."""

    @patch.dict(os.environ, {"VLLM_DT_MAX_BATCH_TKV_LIMIT": "999999"}, clear=True)
    def test_user_can_override_some_device_configs(
        self, registry, granite_3_3_hf_config, caplog_vllm_spyre
    ):
        """Test that user can override some device configs (set env var before configurator)."""
        # Create mock vllm_config
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=granite_3_3_hf_config, max_model_len=32768)
        vllm_config.parallel_config = Mock(world_size=4)
        vllm_config.scheduler_config = Mock(max_num_seqs=32, max_num_batched_tokens=None)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Get configurator and apply
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is not None

        with patch("vllm_spyre.platform.SpyrePlatform") as mock_platform:
            mock_platform.sendnn_version.return_value = (1, 0, 3)
            mock_platform.sendnn_configured.return_value = True
            _summary = configurator.configure(vllm_config)

        # User's value should be preserved
        assert os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] == "999999"

        # Should log warning about conflict
        assert any(
            "was set to" in record.message and "VLLM_DT_MAX_BATCH_TKV_LIMIT" in record.message
            for record in caplog_vllm_spyre.records
        )

    @patch.dict(os.environ, {"FLEX_HDMA_P2PSIZE": "268435456"}, clear=True)
    def test_configurator_respects_existing_env_vars(self, registry, granite_3_3_hf_config):
        """Test that configurator respects existing env vars when value matches."""
        # Create mock vllm_config
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=granite_3_3_hf_config, max_model_len=32768)
        vllm_config.parallel_config = Mock(world_size=4)
        vllm_config.scheduler_config = Mock(max_num_seqs=32, max_num_batched_tokens=None)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Get configurator and apply
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is not None

        with patch("vllm_spyre.platform.SpyrePlatform") as mock_platform:
            mock_platform.sendnn_version.return_value = (1, 0, 3)
            mock_platform.sendnn_configured.return_value = True
            _summary = configurator.configure(vllm_config)

        # Value should remain unchanged (matching value doesn't trigger warning)
        assert os.environ["FLEX_HDMA_P2PSIZE"] == "268435456"

    @patch.dict(os.environ, {}, clear=True)
    def test_user_gpu_blocks_override_logs_warning(
        self, registry, granite_3_3_hf_config, caplog_vllm_spyre
    ):
        """Test that user can set --num-gpu-blocks-override and configurator logs warning."""
        # Create mock vllm_config with user override
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=granite_3_3_hf_config, max_model_len=32768)
        vllm_config.parallel_config = Mock(world_size=4)
        vllm_config.scheduler_config = Mock(max_num_seqs=32, max_num_batched_tokens=None)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=5000)  # User set this

        # Get configurator and apply
        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is not None

        with patch("vllm_spyre.platform.SpyrePlatform") as mock_platform:
            mock_platform.sendnn_version.return_value = (1, 0, 3)
            mock_platform.sendnn_configured.return_value = True
            _summary = configurator.configure(vllm_config)

        # User's value should be preserved
        assert vllm_config.cache_config.num_gpu_blocks_override == 5000

        # Should log warning about conflict
        assert any(
            "num_gpu_blocks_override was set to 5000" in record.message
            for record in caplog_vllm_spyre.records
        )


class TestGraniteVersionAwareOverrides:
    """Tests for version-aware GPU blocks overrides for Granite models."""

    @pytest.mark.cpu
    @pytest.mark.parametrize(
        "model_name, sendnn_configured, sendnn_version, expected_blocks",
        [
            ("granite-3.3-8b-instruct", True, (0, 0, 0), 8192),
            ("granite-3.3-8b-instruct", True, (1, 0, 2), 2080),
            ("granite-3.3-8b-instruct", True, (1, 1, 0), 8192),
            ("granite-3.3-8b-instruct", False, (1, 0, 2), 8192),
            ("granite-4-8b-dense", True, (1, 1, 0), 8192),
        ],
        ids=lambda vals: f"{vals}",
    )
    def test_granite_version_aware_overrides(
        self, registry, model_name, sendnn_configured, sendnn_version, expected_blocks
    ):
        """Test version-aware GPU blocks and env var overrides for granite models."""
        from unittest import mock
        from vllm.config import CacheConfig, ModelConfig

        # Must ensure no env vars have been overridden before testing
        with (
            mock.patch.dict(os.environ, clear=True),
            mock.patch(
                "vllm_spyre.platform.SpyrePlatform.sendnn_configured", new=lambda: sendnn_configured
            ),
            mock.patch(
                "vllm_spyre.platform.SpyrePlatform.sendnn_version", new=lambda: sendnn_version
            ),
        ):
            # Create mock vllm_config for CB with TP=4
            granite_config = Mock()
            granite_config.model_config = ModelConfig(
                model=str(FIXTURES_PATH / "ibm-granite" / model_name),
                max_model_len=32768,
            )
            granite_config.parallel_config = Mock(world_size=4)
            granite_config.scheduler_config = Mock(max_num_seqs=32, max_num_batched_tokens=None)
            granite_config.cache_config = CacheConfig(swap_space=0.001)

            # Get configurator and apply configuration
            configurator = registry.get_configurator_for_runtime(granite_config)
            assert configurator is not None, (
                f"Model {model_name} should have a matching configurator"
            )

            configurator.configure(granite_config)

            # Verify the configuration was applied correctly
            assert granite_config.cache_config.num_gpu_blocks_override == expected_blocks

            # Verify environment variables were set
            tkv_limit = os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT")
            assert tkv_limit is not None, "VLLM_DT_MAX_BATCH_TKV_LIMIT should be set"
            assert int(tkv_limit) == 128 * 1024

            hdma_size = os.getenv("FLEX_HDMA_P2PSIZE")
            assert hdma_size is not None, "FLEX_HDMA_P2PSIZE should be set"
            assert int(hdma_size) == 256 * 1024 * 1024


# Made with Bob

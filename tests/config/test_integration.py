"""Integration tests for the configuration system - end-to-end scenarios."""

import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from vllm_spyre.config.model_registry import ModelConfigRegistry
from vllm_spyre.config.configurators.model_configurator import ModelConfigurator

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def registry():
    """Fixture providing a registry loaded with real model_configs.yaml."""
    registry = ModelConfigRegistry()
    config_path = Path(__file__).parent.parent.parent / "vllm_spyre/config/model_configs.yaml"
    registry.initialize(config_path)
    return registry


def _load_hf_config(fixture_path: Path) -> Mock:
    """Helper to load HF config from JSON and convert to Mock object."""
    with open(fixture_path) as f:
        config_dict = json.load(f)

    hf_config = Mock()
    for key, value in config_dict.items():
        setattr(hf_config, key, value)
    return hf_config


@pytest.fixture
def granite_3_3_hf_config():
    """Fixture providing real granite-3.3-8b-instruct HF config."""
    fixture_path = (
        Path(__file__).parent.parent
        / "fixtures/model_configs/ibm-granite/granite-3.3-8b-instruct/config.json"
    )
    return _load_hf_config(fixture_path)


@pytest.fixture
def granite_4_hf_config():
    """Fixture providing real granite-4-8b-dense HF config."""
    fixture_path = (
        Path(__file__).parent.parent
        / "fixtures/model_configs/ibm-granite/granite-4-8b-dense/config.json"
    )
    return _load_hf_config(fixture_path)


@pytest.fixture
def embedding_hf_config():
    """Fixture providing real granite-embedding-125m-english HF config."""
    fixture_path = (
        Path(__file__).parent.parent
        / "fixtures/model_configs/ibm-granite/granite-embedding-125m-english/config.json"
    )
    return _load_hf_config(fixture_path)


class TestFullWorkflow:
    """End-to-end tests with real model configs."""

    def test_load_real_config_file(self, registry):
        """Test loading the actual model_configs.yaml file."""
        models = registry.list_models()

        # Should have all 9 models from the config file
        assert len(models) == 9

        # Verify specific models are present
        expected_models = [
            "ibm-granite/granite-3.3-8b-instruct",
            "ibm-granite/granite-3.3-8b-instruct-FP8",
            "ibm-granite/granite-4-8b-dense",
            "ibm-granite/granite-embedding-125m-english",
            "ibm-granite/granite-embedding-278m-multilingual",
            "intfloat/multilingual-e5-large",
            "BAAI/bge-reranker-v2-m3",
            "BAAI/bge-reranker-large",
            "sentence-transformers/all-roberta-large-v1",
        ]
        assert set(models) == set(expected_models)

    def test_match_granite_3_3_cb_config(self, registry, granite_3_3_hf_config):
        """Test matching granite-3.3-8b-instruct with CB config and getting configurator."""
        # Create mock vllm_config for CB with TP=4
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=granite_3_3_hf_config, max_model_len=32768)
        vllm_config.parallel_config = Mock(world_size=4)
        vllm_config.scheduler_config = Mock(max_num_seqs=32)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Get configurator
        configurator = registry.get_configurator_for_runtime(vllm_config)

        # Should match and return a configurator
        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-3.3-8b-instruct"

        # Should have device_config for TP=4 CB config
        assert configurator.device_config is not None
        assert configurator.device_config.env_vars is not None
        assert "VLLM_DT_MAX_BATCH_TKV_LIMIT" in configurator.device_config.env_vars

    def test_match_granite_3_3_static_config(self, registry, granite_3_3_hf_config):
        """Test matching granite-3.3-8b-instruct with static batching config."""
        # Create mock vllm_config for static batching with TP=4
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=granite_3_3_hf_config)
        vllm_config.parallel_config = Mock(world_size=4)
        vllm_config.scheduler_config = Mock(max_num_seqs=None)  # Static batching
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Warmup shapes that match config: [[6144, 2048, 1]]
        warmup_shapes = [(6144, 2048, 1)]

        # Get configurator
        configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)

        # Should match and return a configurator
        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-3.3-8b-instruct"

        # Static batching configs don't have device_config
        assert configurator.device_config is None

    def test_match_granite_4_cb_config(self, registry, granite_4_hf_config):
        """Test matching granite-4-8b-dense with CB config."""
        # Create mock vllm_config for CB with TP=4
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=granite_4_hf_config, max_model_len=32768)
        vllm_config.parallel_config = Mock(world_size=4)
        vllm_config.scheduler_config = Mock(max_num_seqs=32)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Get configurator
        configurator = registry.get_configurator_for_runtime(vllm_config)

        # Should match and return a configurator
        assert configurator is not None
        assert isinstance(configurator, ModelConfigurator)
        assert configurator.model_config.name == "ibm-granite/granite-4-8b-dense"

        # Should have device_config for TP=4 CB config
        assert configurator.device_config is not None

    def test_embedding_models_have_no_device_configs(self, registry, embedding_hf_config):
        """Test that embedding models don't have device_configs."""
        # Create mock vllm_config for static batching
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=embedding_hf_config)
        vllm_config.parallel_config = Mock(world_size=1)
        vllm_config.scheduler_config = Mock(max_num_seqs=None)  # Static batching
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Warmup shapes that match config: [[512, 0, 64]]
        warmup_shapes = [(512, 0, 64)]

        # Get configurator
        configurator = registry.get_configurator_for_runtime(vllm_config, warmup_shapes)

        # Should match
        assert configurator is not None
        assert configurator.model_config.name == "ibm-granite/granite-embedding-125m-english"

        # Embedding models don't have device_config
        assert configurator.device_config is None

    def test_generation_models_with_tp4_have_device_configs(self, registry, granite_3_3_hf_config):
        """Test that generation models with TP=4 have device_configs."""
        # Create mock vllm_config for CB with TP=4
        vllm_config = Mock()
        vllm_config.model_config = Mock(hf_config=granite_3_3_hf_config, max_model_len=32768)
        vllm_config.parallel_config = Mock(world_size=4)
        vllm_config.scheduler_config = Mock(max_num_seqs=32)
        vllm_config.cache_config = Mock(num_gpu_blocks_override=None)

        # Get configurator
        configurator = registry.get_configurator_for_runtime(vllm_config)

        # Should have device_config
        assert configurator is not None
        assert configurator.device_config is not None

        # Verify device_config has expected fields
        assert configurator.device_config.env_vars is not None
        assert configurator.device_config.num_gpu_blocks_override is not None
        assert configurator.device_config.chunked_prefill_config is not None

    @patch.dict(os.environ, {}, clear=True)
    def test_apply_configuration_sets_env_vars(self, registry, granite_3_3_hf_config):
        """Test that applying configuration sets env vars correctly."""
        # Create mock vllm_config for CB with TP=4
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
            summary = configurator.configure(vllm_config)

        # Verify env vars were set
        assert "VLLM_DT_MAX_BATCH_TKV_LIMIT" in summary.env_vars
        assert summary.env_vars["VLLM_DT_MAX_BATCH_TKV_LIMIT"] == "131072"
        assert "FLEX_HDMA_P2PSIZE" in summary.env_vars
        assert summary.env_vars["FLEX_HDMA_P2PSIZE"] == "268435456"

    @patch.dict(os.environ, {}, clear=True)
    def test_apply_configuration_sets_gpu_blocks(self, registry, granite_3_3_hf_config):
        """Test that applying configuration sets GPU blocks correctly."""
        # Create mock vllm_config for CB with TP=4
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
            summary = configurator.configure(vllm_config)

        # Verify GPU blocks were set
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
        fixture_path = (
            Path(__file__).parent.parent
            / "fixtures/model_configs/ibm-ai-platform/micro-g3.3-8b-instruct-1b/config.json"
        )
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


# Made with Bob

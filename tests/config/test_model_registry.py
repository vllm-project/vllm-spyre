"""Tests for ModelRegistry - registry operations and runtime matching."""

import pytest
import yaml
from unittest.mock import Mock

from vllm_spyre.config.model_config import ModelConfig
from vllm_spyre.config.model_registry import ModelConfigRegistry

pytestmark = pytest.mark.skip_global_cleanup


class TestCBConfigMatchingLogic:
    """Tests for continuous batching config matching logic."""

    def test_cb_config_matching_any_mismatch_skips(self):
        """Test that CB config is skipped if ANY parameter mismatches."""
        # Create a test registry with a model that has one CB config
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
        """

        data = yaml.safe_load(yaml_content)
        registry = ModelConfigRegistry()
        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        registry.register_model(model_config)

        # Create mock vllm configs for testing

        # Test 1: Only TP size mismatches -> should NOT match
        vllm_config = Mock()
        vllm_config.model_config = Mock()
        vllm_config.model_config.hf_config = Mock(model_type="test")
        vllm_config.parallel_config = Mock(world_size=2)  # Different TP
        vllm_config.model_config.max_model_len = 8192
        vllm_config.scheduler_config = Mock(max_num_seqs=32)

        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None, "Should not match when only TP size differs"

        # Test 2: Only max_model_len mismatches -> should NOT match
        vllm_config.parallel_config.world_size = 4  # Correct TP
        vllm_config.model_config.max_model_len = 4096  # Different max_model_len
        vllm_config.scheduler_config.max_num_seqs = 32

        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None, "Should not match when only max_model_len differs"

        # Test 3: Only max_num_seqs mismatches -> should NOT match
        vllm_config.parallel_config.world_size = 4
        vllm_config.model_config.max_model_len = 8192
        vllm_config.scheduler_config.max_num_seqs = 16  # Different max_num_seqs

        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is None, "Should not match when only max_num_seqs differs"

        # Test 4: All match -> SHOULD match
        vllm_config.parallel_config.world_size = 4
        vllm_config.model_config.max_model_len = 8192
        vllm_config.scheduler_config.max_num_seqs = 32

        configurator = registry.get_configurator_for_runtime(vllm_config)
        assert configurator is not None, "Should match when all parameters match"


class TestWarmupShapesSubset:
    """Tests for warmup shapes subset matching."""

    def test_runtime_warmup_shapes_can_be_subset_of_config(self):
        """Test that runtime warmup shapes can be a subset of config shapes."""
        registry = ModelConfigRegistry()

        # Config has 3 warmup shapes
        config_shapes = [(64, 20, 4), (128, 40, 2), (256, 80, 1)]

        # Runtime only uses 2 of them (subset) -> should match
        runtime_shapes = [(64, 20, 4), (128, 40, 2)]
        assert registry._warmup_shapes_compatible(config_shapes, runtime_shapes), (
            "Runtime shapes should be allowed to be subset of config shapes"
        )

        # Runtime uses 1 of them (subset) -> should match
        runtime_shapes = [(256, 80, 1)]
        assert registry._warmup_shapes_compatible(config_shapes, runtime_shapes), (
            "Single runtime shape should match if it's in config shapes"
        )

        # Runtime uses shapes not in config -> should NOT match
        runtime_shapes = [(512, 100, 1)]
        assert not registry._warmup_shapes_compatible(config_shapes, runtime_shapes), (
            "Runtime shapes not in config should not match"
        )

        # Runtime uses mix of valid and invalid -> should NOT match
        runtime_shapes = [(64, 20, 4), (512, 100, 1)]
        assert not registry._warmup_shapes_compatible(config_shapes, runtime_shapes), (
            "Runtime shapes with any invalid shape should not match"
        )

    def test_warmup_shapes_order_independent(self):
        """Test that warmup shapes matching is order-independent."""
        registry = ModelConfigRegistry()

        config_shapes = [(64, 20, 4), (128, 40, 2), (256, 80, 1)]

        # Same shapes, different order -> should match
        runtime_shapes = [(256, 80, 1), (64, 20, 4)]
        assert registry._warmup_shapes_compatible(config_shapes, runtime_shapes), (
            "Warmup shapes should match regardless of order"
        )

    def test_empty_runtime_shapes_does_not_match(self):
        """Test that empty runtime shapes does not match."""
        registry = ModelConfigRegistry()

        config_shapes = [(64, 20, 4), (128, 40, 2)]
        runtime_shapes = []

        assert not registry._warmup_shapes_compatible(config_shapes, runtime_shapes), (
            "Empty runtime shapes should not match"
        )


class TestDuplicateRuntimeConfigDetection:
    """Tests for duplicate runtime configuration detection."""

    def test_registry_rejects_duplicate_cb_configs(self):
        """Test that registry detects and rejects duplicate CB configs."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
        """

        data = yaml.safe_load(yaml_content)

        with pytest.raises(ValueError, match="Duplicate runtime configuration"):
            ModelConfig.from_dict("test-model", data["models"]["test-model"])

    def test_registry_rejects_duplicate_static_configs(self):
        """Test that registry detects and rejects duplicate static batching configs."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    static_batching_configs:
      - tp_size: 1
        warmup_shapes: [[64, 20, 4], [128, 40, 2]]
      - tp_size: 1
        warmup_shapes: [[64, 20, 4], [128, 40, 2]]
        """

        data = yaml.safe_load(yaml_content)

        with pytest.raises(ValueError, match="Duplicate runtime configuration"):
            ModelConfig.from_dict("test-model", data["models"]["test-model"])

    def test_registry_rejects_duplicate_static_configs_different_order(self):
        """Test that duplicate detection works even with different warmup shape order."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    static_batching_configs:
      - tp_size: 1
        warmup_shapes: [[64, 20, 4], [128, 40, 2]]
      - tp_size: 1
        warmup_shapes: [[128, 40, 2], [64, 20, 4]]
        """

        data = yaml.safe_load(yaml_content)

        with pytest.raises(ValueError, match="Duplicate runtime configuration"):
            ModelConfig.from_dict("test-model", data["models"]["test-model"])

    def test_registry_allows_different_cb_configs_same_tp(self):
        """Test that different CB configs with same TP are allowed."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    continuous_batching_configs:
      - tp_size: 4
        max_model_len: 8192
        max_num_seqs: 32
      - tp_size: 4
        max_model_len: 16384
        max_num_seqs: 16
        """

        data = yaml.safe_load(yaml_content)

        # Should not raise
        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        assert len(model_config.continuous_batching_configs) == 2

    def test_registry_allows_different_static_configs_same_tp(self):
        """Test that different static configs with same TP are allowed."""
        yaml_content = """
models:
  test-model:
    architecture:
      model_type: test
    static_batching_configs:
      - tp_size: 1
        warmup_shapes: [[64, 20, 4]]
      - tp_size: 1
        warmup_shapes: [[128, 40, 2]]
        """

        data = yaml.safe_load(yaml_content)

        # Should not raise
        model_config = ModelConfig.from_dict("test-model", data["models"]["test-model"])
        assert len(model_config.static_batching_configs) == 2


# Made with Bob

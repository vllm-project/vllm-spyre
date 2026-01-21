"""Tests for model-specific overrides for granite"""

import os
from pathlib import Path
from unittest import mock

import pytest
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

from vllm_spyre.config.model_registry import get_model_registry

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "model_configs"


def NO_SWAP_CONFIG():
    return CacheConfig(swap_space=0.001)


@pytest.mark.cpu
def test_granite_3_8b_detection():
    """Check that we can detect the model config for granite 3 8b"""

    granite_3_8b_config = VllmConfig(
        model_config=ModelConfig(
            model=str(FIXTURES_PATH / "ibm-granite" / "granite-3.3-8b-instruct")
        ),
        cache_config=NO_SWAP_CONFIG(),
    )

    registry = get_model_registry()
    matched_model = registry.find_matching_model(granite_3_8b_config.model_config)

    assert matched_model == "ibm-granite/granite-3.3-8b-instruct"


@pytest.mark.cpu
def test_granite_4_dense_detection():
    """Check that we can detect the model config for granite 4 8b (dense)"""

    granite_4_dense_config = VllmConfig(
        model_config=ModelConfig(model=str(FIXTURES_PATH / "ibm-granite" / "granite-4-8b-dense")),
        cache_config=NO_SWAP_CONFIG(),
    )

    registry = get_model_registry()
    matched_model = registry.find_matching_model(granite_4_dense_config.model_config)

    assert matched_model == "ibm-granite/granite-4-8b-dense"


@pytest.mark.cpu
def test_micro_model_not_in_registry():
    """Check that micro model does not match any registered model"""

    granite_micro_config = VllmConfig(
        model_config=ModelConfig(
            model=str(FIXTURES_PATH / "ibm-ai-platform" / "micro-g3.3-8b-instruct-1b")
        ),
        cache_config=NO_SWAP_CONFIG(),
    )

    registry = get_model_registry()
    matched_model = registry.find_matching_model(granite_micro_config.model_config)

    assert matched_model is None


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
def test_granite_overrides(model_name, sendnn_configured, sendnn_version, expected_blocks):
    """Check that the correct values are overridden for granite 8b dense variants"""

    # Must ensure no env vars have been overridden before testing
    with (
        mock.patch.dict(os.environ, clear=True),
        mock.patch(
            "vllm_spyre.platform.SpyrePlatform.sendnn_configured", new=lambda: sendnn_configured
        ),
        mock.patch("vllm_spyre.platform.SpyrePlatform.sendnn_version", new=lambda: sendnn_version),
    ):
        tp4_config = ParallelConfig(tensor_parallel_size=4)

        granite_config = VllmConfig(
            model_config=ModelConfig(model=str(FIXTURES_PATH / "ibm-granite" / model_name)),
            parallel_config=tp4_config,
            cache_config=NO_SWAP_CONFIG(),
        )

        # Apply model-specific configuration using the registry
        registry = get_model_registry()
        matched_model = registry.find_matching_model(granite_config.model_config)
        assert matched_model is not None, f"Model {model_name} should be in registry"

        configurator = registry.get_configurator(matched_model)
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

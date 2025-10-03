"""Tests for model-specific overrides for granite"""
from pathlib import Path

import pytest
from vllm.config import ModelConfig, ParallelConfig, VllmConfig

from vllm_spyre.platform import SpyrePlatform

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "models"


@pytest.mark.cpu
def test_granite_3_8b_detection():
    """Check that we can detect the model config for granite 3 8b"""

    granite_3_8b_config = VllmConfig(model_config=ModelConfig(
        model=str(FIXTURES_PATH / "granite-3.3-8b-instruct-config-only")))

    granite_micro_config = VllmConfig(model_config=ModelConfig(
        model=str(FIXTURES_PATH / "granite-3.3-micro-config-only")))

    assert SpyrePlatform.is_granite_3_8b(granite_3_8b_config.model_config)

    assert not SpyrePlatform.is_granite_3_8b(granite_micro_config.model_config)


@pytest.mark.cpu
def test_granite_3_8b_overrides():
    """Check that the correct values are overridden for g3.3 8b"""

    tp4_config = ParallelConfig(tensor_parallel_size=4)

    granite_3_8b_config = VllmConfig(model_config=ModelConfig(
        model=str(FIXTURES_PATH / "granite-3.3-8b-instruct-config-only")),
                                     parallel_config=tp4_config)

    assert granite_3_8b_config.cache_config.num_gpu_blocks_override == 2080

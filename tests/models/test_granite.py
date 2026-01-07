"""Tests for model-specific overrides for granite"""

import os
from pathlib import Path
from unittest import mock

import pytest
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

from vllm_spyre.platform import SpyrePlatform

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

    granite_micro_config = VllmConfig(
        model_config=ModelConfig(
            model=str(FIXTURES_PATH / "ibm-ai-platform" / "micro-g3.3-8b-instruct-1b")
        ),
        cache_config=NO_SWAP_CONFIG(),
    )

    assert SpyrePlatform.is_granite_3_8b(granite_3_8b_config.model_config)

    assert not SpyrePlatform.is_granite_3_8b(granite_micro_config.model_config)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "sendnn_configured, sendnn_version, expected_blocks",
    [
        (True, (0, 0, 0), 8192),
        (True, (1, 0, 2), 2080),
        (True, (1, 1, 0), 8192),
        (False, (1, 0, 2), 8192),
    ],
    ids=lambda vals: f"{vals}",
)
def test_granite_3_8b_overrides(sendnn_configured, sendnn_version, expected_blocks):
    """Check that the correct values are overridden for g3.3 8b"""

    # Must ensure no env vars have been overridden before testing
    with (
        mock.patch.dict(os.environ, clear=True),
        mock.patch(
            "vllm_spyre.platform.SpyrePlatform.sendnn_configured", new=lambda: sendnn_configured
        ),
        mock.patch("vllm_spyre.platform.SpyrePlatform.sendnn_version", new=lambda: sendnn_version),
    ):
        tp4_config = ParallelConfig(tensor_parallel_size=4)

        granite_3_8b_config = VllmConfig(
            model_config=ModelConfig(
                model=str(FIXTURES_PATH / "ibm-granite" / "granite-3.3-8b-instruct")
            ),
            parallel_config=tp4_config,
            cache_config=NO_SWAP_CONFIG(),
        )

        assert granite_3_8b_config.cache_config.num_gpu_blocks_override == expected_blocks

        assert int(os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT")) == 128 * 1024
        assert int(os.getenv("FLEX_HDMA_P2PSIZE")) == 256 * 1024 * 1024

"""Tests for model-specific overrides for granite"""

import os
from pathlib import Path
from unittest import mock

import pytest
from vllm.config import CacheConfig, ModelConfig, ParallelConfig, VllmConfig

import vllm_spyre.envs as envs_spyre
from vllm_spyre.platform import SpyrePlatform

FIXTURES_PATH = Path(__file__).parent.parent / "fixtures" / "model_configs"

NO_SWAP_CONFIG = CacheConfig(swap_space=0.001)


@pytest.mark.cpu
def test_granite_3_8b_detection():
    """Check that we can detect the model config for granite 3 8b"""

    granite_3_8b_config = VllmConfig(
        model_config=ModelConfig(model=str(FIXTURES_PATH / "ibm-granite" /
                                           "granite-3.3-8b-instruct")),
        cache_config=NO_SWAP_CONFIG,
    )

    granite_micro_config = VllmConfig(
        model_config=ModelConfig(model=str(FIXTURES_PATH / "ibm-ai-platform" /
                                           "micro-g3.3-8b-instruct-1b")),
        cache_config=NO_SWAP_CONFIG,
    )

    assert SpyrePlatform.is_granite_3_8b(granite_3_8b_config.model_config)

    assert not SpyrePlatform.is_granite_3_8b(granite_micro_config.model_config)


@pytest.mark.cpu
def test_granite_3_8b_overrides():
    """Check that the correct values are overridden for g3.3 8b"""

    # Must ensure no env vars have been overridden before testing
    with mock.patch.dict(os.environ, clear=True):
        tp4_config = ParallelConfig(tensor_parallel_size=4)

        granite_3_8b_config = VllmConfig(
            model_config=ModelConfig(model=str(FIXTURES_PATH / "ibm-granite" /
                                               "granite-3.3-8b-instruct")),
            parallel_config=tp4_config,
            cache_config=NO_SWAP_CONFIG,
        )
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn":
            try:
                from torch_sendnn import torch_sendnn  # noqa: F401
            except ImportError:
                print("WARNING: Disabled: torch_sendnn")
            version_str = torch_sendnn._version.__version__
            version = tuple(map(int, version_str.split(".")))

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn" and \
                (0, 0, 0) < version < (1, 0, 3):
            blocks_override_expected = 2080
        else:
            blocks_override_expected = 8192

        assert (granite_3_8b_config.cache_config.num_gpu_blocks_override ==
                blocks_override_expected)

        assert int(os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT")) == 128 * 1024
        assert int(os.getenv("FLEX_HDMA_P2PSIZE")) == 256 * 1024 * 1024

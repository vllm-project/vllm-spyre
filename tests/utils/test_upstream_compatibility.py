import os

import pytest
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_spyre.compat_utils import dataclass_fields

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


def test_mm_inputs():

    if VLLM_VERSION == "vLLM:lowest":
        # Can remove "mm_kwargs", "mm_hashes", "mm_positions"
        # (replaced by mm_features)
        assert 'mm_kwargs' in dataclass_fields(NewRequestData)


def test_get_sampler():
    if VLLM_VERSION == "vLLM:lowest":
        try:
            from vllm.model_executor.layers.sampler import (  # # noqa
                get_sampler)
        except ImportError as e:
            raise AssertionError(
                "Remove backwards compatibility for get_sampler") from e


def test_use_mla():
    if VLLM_VERSION == "vLLM:lowest":
        # Can remove backwards compatibility for use_mla
        assert "use_mla" in dataclass_fields(FullAttentionSpec)

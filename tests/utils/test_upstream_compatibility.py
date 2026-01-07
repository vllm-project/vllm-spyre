import os

import pytest
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_spyre.compat_utils import dataclass_fields, has_argument

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


def test_mm_inputs():
    if VLLM_VERSION == "vLLM:lowest":
        # Can remove "mm_kwargs", "mm_hashes", "mm_positions"
        # (replaced by mm_features)
        assert "mm_kwargs" in dataclass_fields(NewRequestData)


def test_get_sampler():
    if VLLM_VERSION == "vLLM:lowest":
        try:
            from vllm.model_executor.layers.sampler import (  # # noqa
                get_sampler,
            )
        except ImportError as e:
            raise AssertionError("Remove backwards compatibility for get_sampler") from e


def test_use_mla():
    if VLLM_VERSION == "vLLM:lowest":
        # Can remove backwards compatibility for use_mla
        assert "use_mla" in dataclass_fields(FullAttentionSpec)


def test_pin_memory_available():
    if VLLM_VERSION == "vLLM:lowest":
        try:
            from vllm.utils import is_pin_memory_available  # # noqa
            from vllm.utils import make_tensor_with_pad  # # noqa
            from vllm.utils import init_cached_hf_modules  # # noqa
        except ImportError as e:
            raise AssertionError(
                "remove backwards compatibility imports for "
                "is_pin_memory_available, "
                "make_tensor_with_pad and init_cached_hf_modules"
            ) from e


def test_multi_modal_cache_stats():
    if VLLM_VERSION == "vLLM:lowest":
        # If this import succeeds then remove the backwards compatibility type
        # def for MultiModalCacheStats
        with pytest.raises(ImportError):
            from vllm.v1.metrics.stats import MultiModalCacheStats  # # noqa


def test_v0_worker_base():
    if VLLM_VERSION == "vLLM:lowest":
        try:
            from vllm.worker.worker_base import WorkerBase  # # noqa
        except ImportError as e:
            raise AssertionError(
                "remove the backwards compatibility code from the SpyreWorker initializer"
            ) from e


def test_structured_output_request_ids():
    if VLLM_VERSION == "vLLM:lowest":
        # Can remove "structured_output_request_ids" and "grammar_bitmask"
        # from backwards compat
        assert "structured_output_request_ids" in dataclass_fields(SchedulerOutput)


def test_hash_block_size():
    if VLLM_VERSION == "vLLM:lowest":
        # Can supply `hash_block_size` everywhere, this was added in 0.12.0
        assert not has_argument(BlockPool, "hash_block_size")


def test_alignment_tokens():
    if VLLM_VERSION == "vLLM:lowest":
        # Can supply `alignment_tokens` everywhere, this was added in 0.12.0
        assert not has_argument(FullAttentionManager.find_longest_cache_hit, "alignment_tokens")

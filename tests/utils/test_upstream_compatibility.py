import inspect
import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


@pytest.mark.cpu
def test_init_distributed_environment():
    """Tests whether vllm's init_distributed_environment
    has the custom timeout argument"""
    from vllm.distributed import init_distributed_environment

    annotations = inspect.getfullargspec(
        init_distributed_environment).annotations

    if VLLM_VERSION == "vLLM:lowest":
        assert 'timeout' \
                not in annotations, ("we should remove compat code which is now"
                " part of released vllm version")

import inspect
import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


@pytest.mark.cpu
def test_init_builtin_logitsprocs():

    import vllm.v1.sample.logits_processor
    has_init_builtin_logitsprocs = hasattr(vllm.v1.sample.logits_processor,
                                           "init_builtin_logitsprocs")

    if VLLM_VERSION == "vLLM:main":
        assert not has_init_builtin_logitsprocs
    elif VLLM_VERSION == "vLLM:lowest":
        assert has_init_builtin_logitsprocs, (
            "The lowest supported vLLM version already"
            "refactored init_builtin_logitsprocs.")
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/443


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

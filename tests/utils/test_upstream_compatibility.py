import inspect
import os

import pytest

from vllm_spyre.compat_utils import dataclass_fields

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


@pytest.mark.cpu
def test_multi_step_scheduling():

    from vllm.config import SchedulerConfig
    has_multi_step = hasattr(SchedulerConfig, "is_multi_step")

    if VLLM_VERSION == "vLLM:main":
        assert not has_multi_step
    elif VLLM_VERSION == "vLLM:lowest":
        assert has_multi_step, ("The lowest supported vLLM version already"
                                "removed multi-step scheduling.")
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/374


@pytest.mark.cpu
def test_engine_core_add_request():

    from vllm.v1.engine import EngineCoreRequest
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.request import Request

    sig = inspect.signature(EngineCore.add_request)

    if VLLM_VERSION == "vLLM:main":
        assert sig.parameters["request"].annotation == Request
    elif VLLM_VERSION == "vLLM:lowest":
        assert sig.parameters["request"].annotation == EngineCoreRequest, (
            "The lowest supported vLLM version already"
            "switched to the new definition of EngineCore.add_request()")
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/354


@pytest.mark.cpu
def test_mm_inputs():

    from vllm.v1.core.sched.output import NewRequestData
    has_mm_inputs = 'mm_inputs' in dataclass_fields(NewRequestData)

    if VLLM_VERSION == "vLLM:main":
        assert not has_mm_inputs
    elif VLLM_VERSION == "vLLM:lowest":
        assert has_mm_inputs, ("The lowest supported vLLM version already"
                               "renamed mm_inputs to mm_kwargs.")
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/380


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

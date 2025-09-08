import inspect
import os

import pytest

from vllm_spyre.compat_utils import dataclass_fields

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


@pytest.mark.cpu
def test_vllm_bert_support():
    '''
    Test if the vllm version under test already has Bert support for V1
    '''

    from vllm.model_executor.models.bert import BertEmbeddingModel

    bert_supports_v0_only = getattr(BertEmbeddingModel, "supports_v0_only",
                                    False)

    if VLLM_VERSION == "vLLM:main":
        assert not bert_supports_v0_only
    elif VLLM_VERSION == "vLLM:lowest":
        assert bert_supports_v0_only, (
            "The lowest supported vLLM version already"
            "supports Bert in V1. Remove the compatibility workarounds.")
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/277


@pytest.mark.cpu
def test_model_config_task(model: str):

    from vllm.engine.arg_utils import EngineArgs

    vllm_config = EngineArgs(model=model).create_engine_config()
    model_config = vllm_config.model_config

    task = getattr(model_config, "task", None)

    if VLLM_VERSION == "vLLM:main":
        assert task is None
    elif VLLM_VERSION == "vLLM:lowest":
        assert task is not None, (
            "The lowest supported vLLM version already"
            "switched to the new definition of runners and task.")
        # The compat code introduced in the PRs below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/341
        # https://github.com/vllm-project/vllm-spyre/pull/352


@pytest.mark.cpu
def test_has_tasks():

    try:
        from vllm import tasks  # noqa
        has_tasks = True
    except Exception:
        has_tasks = False

    if VLLM_VERSION == "vLLM:main":
        assert has_tasks
    elif VLLM_VERSION == "vLLM:lowest":
        assert not has_tasks, (
            "The lowest supported vLLM version already"
            "switched to the new definition of runners and task.")
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/338


@pytest.mark.cpu
def test_pooler_default_args():

    from vllm.model_executor.layers.pooler import Pooler
    has_from_config = hasattr(Pooler, "from_config_with_defaults")

    if not has_from_config:
        annotations = inspect.getfullargspec(Pooler.for_embed).annotations
        if VLLM_VERSION == "vLLM:main":
            assert 'default_normalize' not in annotations
            assert 'default_softmax' not in annotations
        elif VLLM_VERSION == "vLLM:lowest":
            assert 'default_normalize' in annotations
            assert 'default_softmax' in annotations
            # The compat code introduced in the PR below can now be removed:
            # https://github.com/vllm-project/vllm-spyre/pull/361


@pytest.mark.cpu
def test_pooler_default_pooling_type():

    from vllm.model_executor.layers.pooler import Pooler
    has_from_config = hasattr(Pooler, "from_config_with_defaults")

    if not has_from_config:
        annotations = inspect.getfullargspec(Pooler.for_embed).annotations
        if VLLM_VERSION == "vLLM:main":
            assert 'default_pooling_type' not in annotations
        elif VLLM_VERSION == "vLLM:lowest":
            assert 'default_pooling_type' in annotations
            # The compat code introduced in the PR below can now be removed:
            # https://github.com/vllm-project/vllm-spyre/pull/374


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

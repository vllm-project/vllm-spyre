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


def test_request():

    from vllm.v1.request import Request

    annotations = inspect.getfullargspec(Request).annotations

    if VLLM_VERSION == "vLLM:main":
        assert 'multi_modal_kwargs' not in annotations
        assert 'multi_modal_hashes' not in annotations
        assert 'multi_modal_placeholders' not in annotations
    elif VLLM_VERSION == "vLLM:lowest":
        assert 'multi_modal_hashes' in annotations
        assert 'multi_modal_placeholders' in annotations
        assert 'multi_modal_placeholders' in annotations
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/463


def test_model_runner_output():

    from vllm.v1.outputs import ModelRunnerOutput

    annotations = inspect.getfullargspec(ModelRunnerOutput).annotations

    if VLLM_VERSION == "vLLM:main":
        assert 'spec_token_ids' not in annotations
    elif VLLM_VERSION == "vLLM:lowest":
        assert 'spec_token_ids' in annotations
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/463


def test_pooling_metadata():

    from vllm.v1.pool.metadata import PoolingMetadata

    has_build_pooling_cursor = getattr(PoolingMetadata, "build_pooling_cursor",
                                       False)

    if VLLM_VERSION == "vLLM:main":
        assert has_build_pooling_cursor
    elif VLLM_VERSION == "vLLM:lowest":
        assert not has_build_pooling_cursor
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/463


def test_scheduler_output():

    from vllm.v1.core.sched.output import SchedulerOutput
    annotations = inspect.getfullargspec(SchedulerOutput).annotations

    if VLLM_VERSION == "vLLM:main":
        assert 'free_encoder_mm_hashes' in annotations
    elif VLLM_VERSION == "vLLM:lowest":
        assert 'free_encoder_mm_hashes' not in annotations
        # The compat code introduced in the PR below can now be removed:
        # https://github.com/vllm-project/vllm-spyre/pull/463

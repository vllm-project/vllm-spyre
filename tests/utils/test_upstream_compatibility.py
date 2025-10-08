import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")

# def test_scheduler_output():

#     from vllm.v1.core.sched.output import SchedulerOutput
#     annotations = inspect.getfullargspec(SchedulerOutput).annotations

#     if VLLM_VERSION == "vLLM:main":
#         assert 'free_encoder_mm_hashes' in annotations
#     elif VLLM_VERSION == "vLLM:lowest":
#         assert 'free_encoder_mm_hashes' not in annotations
#         # The compat code introduced in the PR below can now be removed:
#         # https://github.com/vllm-project/vllm-spyre/pull/463

"""
Tests checking for vLLM upstream compatibility requirements.

As we remove support for old vLLM versions, we want to keep track of the
compatibility code that can be cleaned up.
"""

import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


def test_inputs_reorganization():
    """vllm >= 0.19.0 reorganized vllm.inputs: PR #35182."""
    if VLLM_VERSION == "vLLM:lowest":
        try:
            from vllm.inputs.data import token_inputs  # noqa: F401
        except ImportError as e:
            raise AssertionError(
                "remove backwards compatibility shims for token_inputs -> tokens_input, "
                "ProcessorInputs -> EngineInput, TokenInputs -> TokensInput in:\n"
                "  vllm_spyre/platform.py\n"
                "  vllm_spyre/v1/worker/spyre_model_runner.py\n"
                "  tests/utils/test_platform_validation.py"
            ) from e


def test_request_status_waiting_for_fsm_rename():
    """vllm >= 0.19.0 renamed RequestStatus.WAITING_FOR_FSM: PR #38048."""
    if VLLM_VERSION == "vLLM:lowest":
        from vllm.v1.request import RequestStatus

        if not hasattr(RequestStatus, "WAITING_FOR_FSM"):
            raise AssertionError(
                "remove the _WAITING_FOR_GRAMMAR compat shim in "
                "tests/v1/core/test_scheduler_structured_outputs.py"
            )


def test_pooler_activations_reorganization():
    """vllm >= 0.19.0 merged get_cross_encoder_act_fn into get_act_fn: PR #37537."""
    if VLLM_VERSION == "vLLM:lowest":
        try:
            from vllm.model_executor.layers.pooler.activations import (  # noqa: F401
                get_cross_encoder_act_fn,
            )
        except ImportError as e:
            raise AssertionError(
                "remove backwards compatibility shim for get_cross_encoder_act_fn -> get_act_fn "
                "in vllm_spyre/v1/worker/spyre_model_runner.py"
            ) from e

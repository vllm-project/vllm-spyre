"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

import json
import math
from pathlib import Path

import pytest
from spyre_util import get_spyre_backend_list
from vllm import LLM, RequestOutput, SamplingParams
from vllm.config import ModelConfig, VllmConfig

from vllm_spyre.platform import SpyrePlatform


@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prompt_logprobs(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    '''
    This test uses expected prompt_logprob values recorded from running vllm
    v0.9.0 on an A100 GPU.
    These are valid for the model "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    only.
    '''
    model = "ibm-ai-platform/micro-g3.3-8b-instruct-1b"
    num_prompt_logprobs = 5

    json_path = Path(__file__).parent.parent / "expected_prompt_logprobs.json"
    with open(json_path) as f:
        expected_prompt_logprobs: dict[str, list] = json.load(f)

    monkeypatch.setenv("VLLM_USE_V1", 1)
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", 1)
    llm = LLM(model)

    # dict is prompt -> prompt_logprob list
    prompts = list(expected_prompt_logprobs.keys())

    responses: list[RequestOutput] = llm.generate(
        prompts,
        sampling_params=SamplingParams(prompt_logprobs=num_prompt_logprobs))

    for prompt, response in zip(prompts, responses):
        actual_logprobs = response.prompt_logprobs
        expected_logprobs = expected_prompt_logprobs[prompt]
        _compare_prompt_logprobs(expected_logprobs, actual_logprobs)


@pytest.mark.cpu
def test_prompt_logprobs_must_be_enabled(monkeypatch: pytest.MonkeyPatch):
    # If prompt logprobs is disabled, requests are rejected
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", 0)
    params = SamplingParams(prompt_logprobs=5)

    with pytest.raises(ValueError, match="VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS"):
        SpyrePlatform.validate_request("test-any-prompt", params)


@pytest.mark.cpu
def test_prompt_logprobs_not_supported_with_cb(
        monkeypatch: pytest.MonkeyPatch):
    # Server shouldn't boot with both prompt logprobs and continuous batching
    # enabled
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", 1)
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", 1)

    with pytest.raises(ValueError, match="continuous batching"):
        VllmConfig(model_config=ModelConfig(task="generate"))


@pytest.mark.cpu
def test_prompt_logprobs_on_single_requests_only(
        monkeypatch: pytest.MonkeyPatch):
    # Only bs=1 is supported for prompt logprobs
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", 1)
    monkeypatch.setenv("VLLM_SPYRE_WARMUP_BATCH_SIZES", 2)

    with pytest.raises(ValueError, match="batch size 1"):
        VllmConfig(model_config=ModelConfig(task="generate"))


def _compare_prompt_logprobs(expected: list, actual: list):
    # Super rough comparison of prompt logprob outputs
    assert len(expected) == len(actual)

    for i in range(len(actual)):
        expected_dict = expected[i]
        actual_dict = actual[i]

        if expected_dict is None and actual_dict is None:
            continue

        expected_token_set = set(int(k) for k in expected_dict)
        actual_token_set = set(actual_dict.keys())

        # Very lenient- we want at least the first rank token and the actual
        # prompt token to match.
        assert len(actual_token_set.intersection(expected_token_set)) >= 2

        for token, actual_logprob in actual_dict.items():
            # skip tokens not in the expected set
            if str(token) not in expected_dict:
                continue

            expected_logprob = expected_dict[str(token)]

            # 60% tolerance- pretty big difference in results atm
            assert math.isclose(expected_logprob["logprob"],
                                actual_logprob.logprob,
                                rel_tol=0.6)

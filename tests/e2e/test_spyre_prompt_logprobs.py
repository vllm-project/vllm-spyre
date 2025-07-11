"""Tests validating the correctness and configuration of prompt_logprobs.

Run `python -m pytest tests/e2e/test_spyre_prompt_logprobs.py`.
"""
import math

import pytest
import torch
import torch.nn.functional
from spyre_util import (get_chicken_soup_prompts, get_spyre_backend_list,
                        get_spyre_model_list, skip_unsupported_tp_size)
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, RequestOutput, SamplingParams
from vllm.config import ModelConfig, VllmConfig

from vllm_spyre.platform import SpyrePlatform


@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("tp_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.multi),
    pytest.param(4, marks=pytest.mark.multi)
],
                         ids=lambda val: f"TP({val})")
def test_prompt_logprobs(
    backend: str,
    model: str,
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    '''
    This test checks the prompt_logprobs output from vllm against a reference
    implementation using huggingface.
    '''
    skip_unsupported_tp_size(tp_size, backend)
    num_prompt_logprobs = 5

    prompts = get_chicken_soup_prompts(4)

    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", "1")
    llm = LLM(model, tensor_parallel_size=tp_size, tokenizer=model)

    responses: list[RequestOutput] = llm.generate(
        prompts,
        sampling_params=SamplingParams(prompt_logprobs=num_prompt_logprobs))

    expected_prompt_logprobs: dict[str, list] = _get_hf_prompt_logprobs(
        model_name=model, prompts=prompts, n=num_prompt_logprobs)

    for prompt, response in zip(prompts, responses):
        actual_logprobs = response.prompt_logprobs
        expected_logprobs = expected_prompt_logprobs[prompt]
        _compare_prompt_logprobs(expected_logprobs,
                                 actual_logprobs,
                                 max_different_tokens=1,
                                 relative_tolerance=0.15)


@pytest.mark.cpu
@pytest.mark.decoder
def test_prompt_logprobs_must_be_enabled(monkeypatch: pytest.MonkeyPatch):
    # If prompt logprobs is disabled, requests are rejected
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", "0")
    params = SamplingParams(prompt_logprobs=5)

    with pytest.raises(ValueError, match="VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS"):
        SpyrePlatform.validate_request("test-any-prompt", params)


@pytest.mark.cpu
@pytest.mark.decoder
def test_prompt_logprobs_not_supported_with_cb(
        monkeypatch: pytest.MonkeyPatch):
    # Server shouldn't boot with both prompt logprobs and continuous batching
    # enabled
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", "1")
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")

    with pytest.raises(ValueError, match="continuous batching"):
        VllmConfig(model_config=ModelConfig(task="generate"))


@pytest.mark.cpu
@pytest.mark.decoder
def test_prompt_logprobs_on_single_requests_only(
        monkeypatch: pytest.MonkeyPatch):
    # Only bs=1 is supported for prompt logprobs
    monkeypatch.setenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", "1")
    monkeypatch.setenv("VLLM_SPYRE_WARMUP_BATCH_SIZES", "2")

    with pytest.raises(ValueError, match="batch size 1"):
        VllmConfig(model_config=ModelConfig(task="generate"))


def _compare_prompt_logprobs(expected: list, actual: list,
                             max_different_tokens: int,
                             relative_tolerance: float):
    # Fuzzy comparison of prompt logprob outputs
    # max_different_tokens is the number of candidate tokens that are allowed to
    # differ at each token in the prompt.
    # relative_tolerance is the tolerance between the expected and actual
    # logprob for each token
    assert len(expected) == len(actual)

    for i in range(len(actual)):
        expected_dict = expected[i]
        actual_dict = actual[i]

        if expected_dict is None and actual_dict is None:
            continue

        expected_token_set = set(expected_dict.keys())
        actual_token_set = set(actual_dict.keys())

        # Check that (most of) the top n tokens are the same
        assert len(expected_token_set -
                   actual_token_set) <= max_different_tokens

        for token, actual_logprob in actual_dict.items():
            # skip tokens not in the expected set
            if token not in expected_dict:
                continue

            expected_logprob = expected_dict[token]

            # 60% tolerance- pretty big difference in results atm
            assert math.isclose(expected_logprob["logprob"],
                                actual_logprob.logprob,
                                rel_tol=relative_tolerance)


def _get_hf_prompt_logprobs(model_name, prompts, n) -> dict[str, list]:
    """Get prompt logprobs from HF model directly, including top n candidates 
    for each token"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt_logprobs = {}
    for prompt in prompts:

        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        # Get logits (model output before softmax)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Shift logits and labels so that tokens align with their predicted
        # logprobs
        shifted_logits = logits[:, :-1, :]  # Remove last logit
        shifted_input_ids = input_ids[:, 1:]  # Remove first token (BOS)

        # Get log-softmax over vocabulary
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)

        # Get top n logprobs:
        topk_logprobs, topk_indices = torch.topk(log_probs, dim=2, k=n)

        # Gather log-probabilities of the actual prompt logprobs
        token_logprobs = log_probs.gather(
            2, shifted_input_ids.unsqueeze(-1)).squeeze(-1)

        # Squeeze out batch dimension 1
        token_logprobs = token_logprobs.squeeze()
        topk_logprobs = topk_logprobs.squeeze()
        topk_indices = topk_indices.squeeze()

        # No prompt logprobs for first token
        output_prompt_logprobs = [None]
        for idx in range(len(token_logprobs)):
            logprobs_dict = {}
            # Loop over each token and:
            # Get the set of top N tokens + the actual prompt token
            # (The prompt token may or may not be in the top N)
            # Detokenize and store the token w/ its logprob
            for i, token in enumerate(topk_indices[idx]):
                logprob = topk_logprobs[idx][i]
                text = tokenizer.convert_ids_to_tokens(token.item())
                logprobs_dict[token.item()] = {
                    "decoded_token": text,
                    "logprob": logprob.item(),
                }

            prompt_token = input_ids[0][idx + 1]
            prompt_logprob = token_logprobs[idx]
            decoded_prompt_token = tokenizer.convert_ids_to_tokens(
                prompt_token.item())
            logprobs_dict[prompt_token.item()] = {
                "decoded_token": decoded_prompt_token,
                "logprob": prompt_logprob.item(),
            }

            output_prompt_logprobs.append(logprobs_dict)

        prompt_logprobs[prompt] = output_prompt_logprobs

    return prompt_logprobs

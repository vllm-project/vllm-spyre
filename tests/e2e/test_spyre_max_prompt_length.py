"""Verification of handling prompt length exceeding warmup shapes

Run `python -m pytest tests/test_spyre_max_prompt_length.py`.
"""

import pytest
from spyre_util import (get_spyre_backend_list, get_spyre_model_list,
                        patch_warmup_shapes)
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "warmup_shapes",
    [[(64, 20, 4)], [(64, 20, 4),
                     (128, 20, 4)]])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_max_prompt_length(model: str, warmup_shapes: list[tuple[int, int,
                                                                 int]],
                           backend: str, vllm_version: str,
                           monkeypatch) -> None:
    '''
    Simple test that for static batching:
    - prompts cannot exceed the maximu prompt length of all warmup shapes
    - max_tokens cannot exceed the max new token length of the matching warmup 
        shape
    '''
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    patch_warmup_shapes(warmup_shapes, monkeypatch)
    monkeypatch.setenv("VLLM_USE_V1", "1" if vllm_version == "V1" else "0")

    max_prompt_length = max([t[0] for t in warmup_shapes])
    max_new_tokens = max([t[1] for t in warmup_shapes])
    llm = LLM(model)

    # Craft a request with a prompt that is too long, but where prompt_len +
    # max_tokens is still
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    pepper = "🌶️"
    pepper_tokens = len(hf_tokenizer.encode(pepper, add_special_tokens=False))

    prompt = pepper * (max_prompt_length // pepper_tokens + 1)
    prompt_len = len(hf_tokenizer.encode(prompt))
    assert max_prompt_length < prompt_len < max_prompt_length + max_new_tokens
    sampling_params = SamplingParams(max_tokens=1)

    try:
        results = llm.generate(prompts=[prompt],
                               sampling_params=sampling_params)
        assert results[0].outputs[0].text == ""
    except ValueError as e:
        # V1 will raise on vllm > 0.8.4
        assert vllm_version == "V1"
        assert "matching" in str(e)

    # Craft a request with a prompt that fits, but where too many tokens are
    # requested
    prompt = pepper
    sampling_params = SamplingParams(max_tokens=max_new_tokens + 1)
    try:
        results = llm.generate(prompts=[prompt],
                               sampling_params=sampling_params)
        assert results[0].outputs[0].text == ""
    except ValueError as e:
        # V1 will raise on vllm > 0.8.4
        assert vllm_version == "V1"
        assert "matching" in str(e)

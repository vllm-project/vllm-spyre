"""Verification of handling prompt length exceeding warmup shapes

Run `python -m pytest tests/test_spyre_max_prompt_length.py`.
"""

import pytest
from spyre_util import (create_text_prompt, get_spyre_backend_list,
                        get_spyre_model_list, patch_warmup_shapes)
from vllm import LLM, SamplingParams


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "warmup_shapes",
    [[(64, 20, 4)], [(64, 20, 4),
                     (128, 20, 4)]])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("vllm_version",
                         [pytest.param("V1", marks=pytest.mark.v1, id="v1")
                          ])  # v0 doesn't support multiple shapes
def test_max_prompt_len_and_new_tokens(model: str,
                                       warmup_shapes: list[tuple[int, int,
                                                                 int]],
                                       backend: str, vllm_version: str,
                                       monkeypatch) -> None:
    '''
    Simple test that for static batching:
    - prompts cannot exceed the maximum prompt length of all warmup shapes
    - max_tokens cannot exceed the max new token length of the matching warmup 
        shape

    These two cases are combined to reduce the cost of starting each `LLM`
    '''
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    patch_warmup_shapes(warmup_shapes, monkeypatch)
    monkeypatch.setenv("VLLM_USE_V1", "1" if vllm_version == "V1" else "0")

    max_prompt_length = max([t[0] for t in warmup_shapes])
    max_new_tokens = max([t[1] for t in warmup_shapes])
    llm = LLM(model)

    # Craft a request with a prompt that is slightly too long for the warmup
    # shape
    prompt = create_text_prompt(model,
                                min_tokens=max_prompt_length,
                                max_tokens=max_prompt_length + max_new_tokens -
                                1)
    sampling_params = SamplingParams(max_tokens=1)

    with pytest.raises(ValueError, match="warmup"):
        results = llm.generate(prompts=[prompt],
                               sampling_params=sampling_params)
        assert results[0].outputs[0].text == ""

    # Craft a request with a prompt that fits, but where too many tokens are
    # requested
    prompt = "hello"
    sampling_params = SamplingParams(max_tokens=max_new_tokens + 1)
    with pytest.raises(ValueError, match="warmup"):
        results = llm.generate(prompts=[prompt],
                               sampling_params=sampling_params)
        assert results[0].outputs[0].text == ""

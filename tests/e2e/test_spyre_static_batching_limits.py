"""Verification of handling prompt length exceeding warmup shapes

Run `python -m pytest tests/e2e/test_spyre_max_prompt_length.py`.
"""

import pytest
from llm_cache import get_cached_llm
from spyre_util import DecodeWarmupShapes, ModelInfo, create_text_prompt
from vllm import SamplingParams


@pytest.mark.parametrize(
    "warmup_shapes", [[(64, 20, 4)], [(64, 20, 4), (128, 20, 2)]]
)  # (prompt_length/new_tokens/batch_size)
def test_max_prompt_len_and_new_tokens(
    model: ModelInfo, warmup_shapes: DecodeWarmupShapes, backend: str, use_llm_cache, monkeypatch
) -> None:
    """
    Simple test that for static batching:
    - prompts cannot exceed the maximum prompt length of all warmup shapes
    - max_tokens cannot exceed the max new token length of the matching warmup
        shape

    These two cases are combined to reduce the cost of starting each `LLM`
    """
    # monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    # patch_warmup_shapes(warmup_shapes, monkeypatch)

    max_prompt_length = max([t[0] for t in warmup_shapes])
    max_new_tokens = max([t[1] for t in warmup_shapes])
    # llm = LLM(model)

    llm = get_cached_llm(
        model=model,
        max_model_len=256,  # unused
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
        max_num_seqs=None,
        use_cb=False,
    )

    # Craft a request with a prompt that is slightly too long for the warmup
    # shape
    prompt = create_text_prompt(
        model,
        min_token_length=max_prompt_length,
        max_token_length=max_prompt_length + max_new_tokens - 1,
    )
    sampling_params = SamplingParams(max_tokens=1)

    with pytest.raises(ValueError, match="warmup"):
        results = llm.generate(prompts=[prompt], sampling_params=sampling_params)
        assert results[0].outputs[0].text == ""

    # Craft a request with a prompt that fits, but where too many tokens are
    # requested
    prompt = "hello"
    sampling_params = SamplingParams(max_tokens=max_new_tokens + 1)
    with pytest.raises(ValueError, match="warmup"):
        results = llm.generate(prompts=[prompt], sampling_params=sampling_params)
        assert results[0].outputs[0].text == ""

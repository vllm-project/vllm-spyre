"""Verification of Spyre warmup shapes

Run `python -m pytest tests/e2e/test_spyre_warmup_shapes.py`.
"""

import pytest
from output_util import generate_spyre_vllm_output, validate_vllm_vs_hf_output
from spyre_util import DecodeWarmupShapes, ModelInfo, get_chicken_soup_prompts
from vllm import SamplingParams


@pytest.mark.parametrize(
    "warmup_shapes", [[(64, 20, 4), (128, 20, 2)]]
)  # (prompt_length/new_tokens/batch_size)
def test_multiple_warmup_shapes(
    model: ModelInfo,
    warmup_shapes: DecodeWarmupShapes,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
) -> None:
    """
    The warmup is based on two shapes, that 'overlap' each
    other. After the warmup, one request with the provided
    prompts is input to vLLM. There should be at least one
    prompt corresponding to each of the two warmup shapes.
    It is useful to define enough prompts to fill multiple
    batches entirely and partially, in order to test the
    handling of overlapping warmup shapes also in relation
    with the position of a prompt within a batch (not
    likely that this will be an issue, but just to be sure).
    The same prompts are also input to HF. The generated
    output including text, token ids, and logprobs, is
    verified to be identical for vLLM and HF.
    """

    prompts = get_chicken_soup_prompts(4)

    max_new_tokens = max([t[1] for t in warmup_shapes])

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True,
    )

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        warmup_shapes=warmup_shapes,
        max_model_len=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        max_new_tokens=max_new_tokens,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("prompts", [["Hello"]])
@pytest.mark.parametrize("warmup_shapes", [[(65, 1, 1)]])
def test_invalid_prompt_len(
    model: ModelInfo,
    prompts: list[str],
    warmup_shapes: DecodeWarmupShapes,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
) -> None:
    """
    Expects an error to be raised if the warmup prompt length
    is not divisible by 64.
    """

    vllm_sampling_params = SamplingParams(max_tokens=1, temperature=0, logprobs=0, ignore_eos=True)

    with pytest.raises(RuntimeError, match="VLLM_SPYRE_WARMUP_PROMPT_LENS"):
        generate_spyre_vllm_output(
            model=model,
            prompts=prompts,
            warmup_shapes=warmup_shapes,
            max_model_len=2048,
            sampling_params=vllm_sampling_params,
            tensor_parallel_size=1,
            backend=backend,
            monkeypatch=monkeypatch,
        )

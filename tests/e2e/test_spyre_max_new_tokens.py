"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_max_new_tokens.py`.
"""

import pytest
from output_util import validate_vllm_vs_hf_output
from spyre_util import DecodeWarmupShapes, ModelInfo, get_chicken_soup_prompts
from vllm import SamplingParams

sb_mark = pytest.param("sb", marks=pytest.mark.sb, id="sb")
cb_mark = pytest.param("cb", marks=pytest.mark.cb, id="cb")
cp_mark = pytest.param("cp", marks=pytest.mark.chunked_prefill, id="cp")


@pytest.mark.parametrize("stop_last", [True, False])
@pytest.mark.parametrize("mode", [sb_mark, cb_mark, cp_mark])
def test_output(
    model: ModelInfo,
    stop_last: bool,
    max_model_len: int,
    max_num_seqs: int,
    warmup_shapes: DecodeWarmupShapes,
    backend: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
) -> None:
    """
    Checks that `max_tokens` parameter of `SamplingParams` works correctly
    For each batch, one prompt has max_tokens set to 1 and the others don't.
    This checks that the correct request has only a single output token, while
    the others are not affected.
    """

    prompts = get_chicken_soup_prompts(4)

    max_new_tokens_long = 6
    max_new_tokens_early_stop = 1

    vllm_sampling_params_normal = SamplingParams(
        max_tokens=max_new_tokens_long,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=False,
    )

    vllm_sampling_params_early_stop = SamplingParams(
        max_tokens=max_new_tokens_early_stop,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=False,
    )

    vllm_sampling_params = [vllm_sampling_params_normal.clone() for _ in range(3)]
    hf_max_new_tokens = [max_new_tokens_long] * 3

    # stop last or first sequence in batch early
    if stop_last:
        vllm_sampling_params = vllm_sampling_params + [vllm_sampling_params_early_stop]
        hf_max_new_tokens = hf_max_new_tokens + [max_new_tokens_early_stop]
    else:
        vllm_sampling_params = [vllm_sampling_params_early_stop] + vllm_sampling_params
        hf_max_new_tokens = [max_new_tokens_early_stop] + hf_max_new_tokens

    kwargs = (
        {
            "max_num_seqs": max_num_seqs,
            "use_cb": True,
            "max_num_batched_tokens": 128 if mode == "cp" else None,
        }
        if mode in ["cb", "cp"]
        else {"warmup_shapes": warmup_shapes}
    )

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        max_new_tokens=hf_max_new_tokens,
        max_model_len=max_model_len,
        **kwargs,
    )

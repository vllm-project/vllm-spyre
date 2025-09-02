"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_max_new_tokens.py`.
"""

import pytest
from llm_cache import DecodeWarmupShapes
from spyre_util import (check_output_against_hf, generate_spyre_vllm_output,
                        get_chicken_soup_prompts)
from vllm import SamplingParams


@pytest.mark.parametrize("stop_last", [True, False])
def test_output(model: str, stop_last: bool, max_model_len: int,
                max_num_seqs: int, warmup_shapes: DecodeWarmupShapes,
                backend: str, cb: int, monkeypatch: pytest.MonkeyPatch,
                use_llm_cache) -> None:
    '''
    Checks that `max_tokens` parameter of `SamplingParams` works correctly
    For each batch, one prompt has max_tokens set to 1 and the others don't.
    This checks that the correct request has only a single output token, while
    the others are not affected.
    '''

    prompts = get_chicken_soup_prompts(4)

    max_new_tokens_long = 6
    max_new_tokens_early_stop = 1

    vllm_sampling_params_normal = SamplingParams(
        max_tokens=max_new_tokens_long,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=False)

    vllm_sampling_params_early_stop = SamplingParams(
        max_tokens=max_new_tokens_early_stop,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=False)

    vllm_sampling_params = [vllm_sampling_params_normal] * 3
    hf_max_new_tokens = [max_new_tokens_long] * 3

    # stop last or first sequence in batch early
    if stop_last:
        vllm_sampling_params = vllm_sampling_params + [
            vllm_sampling_params_early_stop
        ]
        hf_max_new_tokens = hf_max_new_tokens + [max_new_tokens_early_stop]
    else:
        vllm_sampling_params = [vllm_sampling_params_early_stop
                                ] + vllm_sampling_params
        hf_max_new_tokens = [max_new_tokens_early_stop] + hf_max_new_tokens

    kwargs = ({
        "max_num_seqs": max_num_seqs,
        "use_cb": True,
    } if cb == 1 else {
        "warmup_shapes": warmup_shapes
    })

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        max_model_len=max_model_len,
        **kwargs)

    check_output_against_hf(model, backend, hf_max_new_tokens, vllm_results,
                            prompts)

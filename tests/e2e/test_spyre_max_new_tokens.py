"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_max_new_tokens.py`.
"""

import pytest
from spyre_util import (DecodeWarmupShapes, check_output_against_hf,
                        default_sb_cb_params, generate_spyre_vllm_output,
                        get_chicken_soup_prompts, get_spyre_backend_list,
                        get_spyre_model_list)
from vllm import SamplingParams


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("stop_last", [True, False])
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@default_sb_cb_params
def test_output(model: str, stop_last: bool, max_model_len: int,
                max_num_seqs: int, warmup_shapes: DecodeWarmupShapes,
                backend: str, cb: int, monkeypatch: pytest.MonkeyPatch,
                use_llm_cache) -> None:
    '''
    Checks that `max_tokens` parameter of `SamplingParams` works correctly
    
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.
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

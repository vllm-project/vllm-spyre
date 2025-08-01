"""Verification of vLLM output by comparing with HF
with VLLM_SPYRE_MAX_LOAD_PROCESSES enabled.

Run `python -m pytest tests/e2e/test_stagger_spyre_basic.py`.
"""

import pytest
from spyre_util import (check_output_against_hf, generate_spyre_vllm_output,
                        get_chicken_soup_prompts, get_spyre_backend_list,
                        get_spyre_model_list, skip_unsupported_tp_size)
from vllm import SamplingParams


@pytest.mark.parametrize("cb",
                         [pytest.param(1, marks=pytest.mark.cb, id="cb"), 0])
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1),
        pytest.param(2, marks=pytest.mark.multi),
        pytest.param(4, marks=pytest.mark.multi),
        pytest.param(8, marks=pytest.mark.multi),
    ],
    ids=lambda val: f"TP({val})",
)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("max_num_seqs", [4],
                         ids=lambda val: f"max_num_seqs({val})")
def test_stagger_output(
    model: str,
    tp_size: int,
    backend: str,
    cb: int,
    max_num_seqs: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.
    
    Configuration for CB - parameters are combinatorial:
        * max_num_seqs: 4
        * tensor parallelism: 1, 2, 4, 8
        * number of prompts: 4 (Chicken soup prompts)
        * max tokens: 20 (same for all the prompts)

    VLLM_SPYRE_MAX_LOAD_PROCESSES is set to 1, allowing
    only a single worker to load or compile the model at
    a time.
    '''

    skip_unsupported_tp_size(tp_size, backend)
    monkeypatch.setenv("VLLM_SPYRE_MAX_LOAD_PROCESSES", "1")

    prompts = get_chicken_soup_prompts(4)
    warmup_shape = (64, 20, 4)

    kwargs = ({
        "max_num_seqs": max_num_seqs,
        "use_cb": True,
        "max_model_len": 256,
        "block_size": 256,
    } if cb == 1 else {
        "warmup_shapes": (warmup_shape, ),
        "max_model_len": 2048,
        "block_size": 2048,
    })

    max_new_tokens = warmup_shape[1]

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend,
        monkeypatch=monkeypatch,
        **kwargs)
    check_output_against_hf(model, backend, max_new_tokens, vllm_results,
                            prompts)

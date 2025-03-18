"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/test_spyre_basic.py`.
"""

from typing import List, Tuple

import pytest
from spyre_util import (compare_results, generate_hf_output,
                        generate_spyre_vllm_output, get_spyre_backend_list,
                        get_spyre_model_list)
from vllm import SamplingParams


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("prompts", [[
    "Provide a list of instructions for preparing"
    " chicken soup for a family of four.", "Hello",
    "What is the weather today like?", "Who are you?"
]])
@pytest.mark.parametrize("warmup_shape", [(64, 20, 4), (64, 20, 8),
                                          (128, 20, 4), (128, 20, 8)]
                         )  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_output(
    model: str,
    prompts: List[str],
    warmup_shape: Tuple[int, int, int],
    backend: str,
    vllm_version: str,
) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.

    If errors occur, these can be analyzed/debugged by setting
    'DISABLE_ASSERTS = True' in spyre_util.py and by rerunning the
    test using 'pytest --capture=no tests/spyre/test_spyre_basic.py'
    After debugging, DISABLE_ASSERTS should be reset to 'False'.
    '''

    max_new_tokens = warmup_shape[1]

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=[warmup_shape],
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        vllm_version=vllm_version)

    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=max_new_tokens)

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=[warmup_shape],
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_batch_handling(
    model: str,
    backend: str,
    vllm_version: str,
):
    """Test that the spyre worker correctly handles batches of requests that
    finish after different numbers of forward passes"""

    # Test with batch size 4
    warmup_shape = (64, 20, 4)

    # Have the model count down to zero and stop
    vllm_sampling_params = SamplingParams(max_tokens=20,
                                          temperature=0,
                                          stop="0",
                                          logprobs=0)
    # Importantly, these prompts are ordered so that they don't finish in the
    # order given
    prompts = [
        "6 5 4 3",
        "9 8 7 6",
        "7 6 5 4",
        "8 7 6 5",
    ]

    # Ensure that both:
    # - The model doesn't crash
    # - The output sequences are correct
    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=[warmup_shape],
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        vllm_version=vllm_version)

    assert vllm_results[0]["text"] == " 2 1 "
    assert vllm_results[1]["text"] == " 5 4 3 2 1 "
    assert vllm_results[2]["text"] == " 3 2 1 "
    assert vllm_results[3]["text"] == " 4 3 2 1 "

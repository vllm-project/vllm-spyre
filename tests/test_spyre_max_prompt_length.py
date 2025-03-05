"""Verification of handling prompt length exceeding warmup shapes

Run `python -m pytest tests/test_spyre_max_prompt_length.py`.
"""

from typing import List, Tuple

import pytest
from spyre_util import (compare_results, generate_hf_output,
                        generate_spyre_vllm_output, get_spyre_backend_list,
                        get_spyre_model_list)
from transformers import AutoTokenizer
from vllm import SamplingParams


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("prompts", [
    7 * [
        "Hello",
        "Below is an instruction that describes a task. Write a response"
        " that appropriately completes the request. Be polite in your response"
        " to the user. Provide a list of instructions for preparing chicken "
        "soup for a family of four. Indicate if the weather forecast looks "
        "good for today. Explain in a brief summary comprised of at most 50"
        " words what you are."
    ]
])
@pytest.mark.parametrize("warmup_shapes",
                         [[(64, 20, 4)], [(64, 20, 4), (128, 20, 4)]]
                         )  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_output(
    model: str,
    prompts: List[str],
    warmup_shapes: List[Tuple[int, int, int]],
    backend: str,
) -> None:
    '''
    The warmup is based on one or multiple shapes. After the warmup,
    one request with multiple provided prompts is input to vLLM.
    At least one provided prompt should have a length longer than the
    maximum prompt length defined by the warmup shapes. It is useful
    to define enough prompts to fill multiple batches entirely and
    partially, in order to test the maximum prompt length check
    also in relation with the position of a prompt within a batch (not
    likely that this will be an issue, but just to be sure).
    It is verified that only for the prompts that
    do not exceed the maximum prompt length, "non-empty" output is
    generated. The output is verified using HF.

    If errors occur, these can be analyzed/debugged by setting
    'DISABLE_ASSERTS = True' in spyre_util.py and by rerunning the test
    using 'pytest --capture=no tests/spyre/test_spyre_max_prompt_length.py'
    After debugging, DISABLE_ASSERTS should be reset to 'False'.
    '''

    max_prompt_length = max([t[0] for t in warmup_shapes])
    max_new_tokens = max([t[1] for t in warmup_shapes])

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=warmup_shapes,
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend)

    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=max_new_tokens)

    # for prompts longer than the max_prompt_length, the corresponding
    # output in hf_results is reset to 'empty' in order to create the
    # expected output for vLLM
    hf_tokenizer = AutoTokenizer.from_pretrained(model)
    for prompt_index, prompt in enumerate(prompts):
        hf_input_tokens = hf_tokenizer(prompt, return_tensors="pt").input_ids
        if len(hf_input_tokens[0]) > max_prompt_length:
            hf_results[prompt_index] = {
                'text': '',
                'token_ids': (),
                'tokens': (),
                'logprobs': ()
            }

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=warmup_shapes,
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)
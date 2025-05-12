"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/test_spyre_max_new_tokens.py`.
"""

import pytest
from spyre_util import (VLLM_VERSIONS, compare_results, generate_hf_output,
                        generate_spyre_vllm_output, get_spyre_backend_list,
                        get_spyre_model_list)
from vllm import SamplingParams

template = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user.\n\n### Instruction:\n{}\n\n### Response:")

prompt1 = template.format(
    "Provide a list of instructions for preparing chicken soup.")
prompt2 = template.format("Explain to me why ignorance is bliss.")


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("prompts", [[prompt1, prompt2, prompt2, prompt2],
                                     [prompt2, prompt2, prompt2, prompt1],
                                     [prompt2, prompt2, prompt2, prompt2]])
@pytest.mark.parametrize("stop_last", [True, False])
@pytest.mark.parametrize(
    "warmup_shape", [(64, 10, 4)])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("vllm_version", VLLM_VERSIONS)
def test_output(
    model: str,
    prompts: list[str],
    stop_last: bool,
    warmup_shape: tuple[int, int, int],
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
    test using 'pytest --capture=no tests/spyre/test_spyre_max_new_tokens.py'
    After debugging, DISABLE_ASSERTS should be reset to 'False'.
    '''

    max_new_tokens_warmup = warmup_shape[1]
    max_new_tokens_early_stop = 1

    vllm_sampling_params_normal = SamplingParams(
        max_tokens=max_new_tokens_warmup,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=False)

    vllm_sampling_params_early_stop = SamplingParams(
        max_tokens=max_new_tokens_early_stop,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=False)

    vllm_sampling_params = [vllm_sampling_params_normal] * 3
    hf_max_new_tokens = [max_new_tokens_warmup] * 3

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
                                    max_new_tokens=hf_max_new_tokens)

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=[warmup_shape],
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)

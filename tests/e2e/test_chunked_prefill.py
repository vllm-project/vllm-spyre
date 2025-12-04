"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

from typing import cast
from unittest.mock import patch

import pytest
from llm_cache import get_cached_llm
from output_util import (compare_results, extract_output, generate_hf_output,
                         setup_golden_token)
from pytest_mock.plugin import MockerFixture
from spyre_util import (ModelInfo, get_chicken_soup_prompts,
                        get_longer_chicken_soup_prompts)
from vllm import LLM, SamplingParams

from vllm_spyre.v1.worker.spyre_model_runner import SamplingForwardInputs


def get_model_runner(cp_model: LLM):
    return cp_model.llm_engine.\
        engine_core.engine_core.model_executor.\
            driver_worker.worker.model_runner


chicken_soup_prompts = get_longer_chicken_soup_prompts(4)

# NOTE: considering granite 3.3 tokenizer
# Should have 51 tokens
prompt_51 = get_chicken_soup_prompts(1)[0]
# Should have 95 tokens
prompt_95 = chicken_soup_prompts[0]
# Should have 251
prompt_251 = chicken_soup_prompts[0] + chicken_soup_prompts[
    1] + chicken_soup_prompts[2]
# Should have 260 tokens
prompt_260 = chicken_soup_prompts[0] + chicken_soup_prompts[2] + \
    chicken_soup_prompts[3]

USE_CASES = {
    # Case I - Prompt fits in a single chunk
    "case_Ia": (prompt_95, 128, 1, 0),
    # Case I - Prompt fits in a single chunk with left padding
    "case_Ib": (prompt_51, 128, 1, 64),
    # Case II - Has left padding
    "case_II": (prompt_260, 128, 3, 64),
    # Case III again - no padding
    "case_III": (prompt_251, 128, 2, 0),
}


@pytest.mark.chunked_prefill
@pytest.mark.parametrize("use_case", list(USE_CASES.keys()))
def test_chunked_prefill_correctness(model: ModelInfo, backend: str,
                                     max_num_seqs: int, max_model_len: int,
                                     monkeypatch: pytest.MonkeyPatch,
                                     mocker: MockerFixture, use_case: str,
                                     use_llm_cache) -> None:
    """
    Minimal test to check if vllm-spyre activate code for chunked prefill for
    a prompt greater than the chunk size
    """


    (prompt, chunk_size, expected_chunk_count, expected_left_padding) =\
          USE_CASES[use_case]
    max_new_tokens = 8

    hf_outputs = generate_hf_output(
        model=model,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
        ignore_eos=True,
    )
    ### NB: May not be guaranteed to be set
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    cp_model = get_cached_llm(model=model,
                              max_model_len=max_model_len,
                              tensor_parallel_size=1,
                              backend=backend,
                              monkeypatch=monkeypatch,
                              max_num_seqs=max_num_seqs,
                              use_cb=True,
                              max_num_batched_tokens=chunk_size)
    model_runner = get_model_runner(cp_model)

    sampling_params = SamplingParams(max_tokens=max_new_tokens,
                                     temperature=0,
                                     logprobs=0,
                                     ignore_eos=True)
    gti_sampling_params = setup_golden_token(model, sampling_params,
                                             hf_outputs)

    _prepare_chunked_prefill = model_runner._prepare_chunked_prefill
    records = []

    def wrapper(self, *args, **kwargs):
        model_input = \
            _prepare_chunked_prefill(self, *args, **kwargs)
        records.append(model_input)
        return model_input

    with patch.object(model_runner, "_prepare_chunked_prefill",
                      wraps=wrapper) as spy:
        results = cp_model.generate(prompt, gti_sampling_params)
        vllm_results = [extract_output(results[0])]

        for r in records:
            model_input = cast(SamplingForwardInputs, r)
            # Must be a single value
            left_padding = model_input.left_padded_prompt_mask[0].item()
            assert left_padding == expected_left_padding

    # Validate if the prefill was chunked
    spy.assert_called()
    assert spy.call_count == expected_chunk_count

    # Validate output
    compare_results(model=model,
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_outputs,
                    prompts=[prompt])

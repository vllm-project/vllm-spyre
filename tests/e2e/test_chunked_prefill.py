"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

from unittest.mock import patch

import pytest
from llm_cache import get_cached_llm
from pytest_mock.plugin import MockerFixture
from spyre_util import ModelInfo
from vllm import LLM, SamplingParams

CHUNK_SIZE = 128


def get_model_runner(cp_model: LLM):
    return cp_model.llm_engine.\
        engine_core.engine_core.model_executor.\
            driver_worker.worker.model_runner


@pytest.mark.cb
def test_prepare_chunked_prefill_called(model: ModelInfo, backend: str,
                                        max_num_seqs: int, max_model_len: int,
                                        monkeypatch: pytest.MonkeyPatch,
                                        mocker: MockerFixture,
                                        use_llm_cache) -> None:
    """
    Minimal test to check if vllm-spyre activate code for chunked prefill for
    a prompt greater than the chunk size
    """

    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    cp_model = get_cached_llm(model=model,
                              max_model_len=max_model_len,
                              tensor_parallel_size=1,
                              backend=backend,
                              monkeypatch=monkeypatch,
                              max_num_seqs=max_num_seqs,
                              use_cb=True,
                              max_num_batched_tokens=CHUNK_SIZE)
    model_runner = get_model_runner(cp_model)

    # This should have ~167 tokens, Needs 2 prefills for 128 chunk size
    prompt = \
     ("Lorem ipsum dolor sit amet, consectetur"
     "adipiscing elit, sed do eiusmod tempor incididunt ut labore"
     "et dolore magna aliqua. Ut enim ad minim veniam,"
     "quis nostrud exercitation ullamco laboris nisi ut"
     "aliquip ex ea commodo consequat. Duis aute irure dolor"
     "in reprehenderit in voluptate velit esse cillum"
     "dolore eu fugiat nulla pariatur. Excepteur sint"
     "occaecat cupidatat non proident, sunt in culpa qui"
     "officia deserunt mollit anim id est laborum.Lorem ipsum"
     "dolor sit amet, consectetur adipiscing elit, sed do"
     "eiusmod tempor incididunt ut labore et dolore magna"
     "aliqua. Ut enim ad minim veniam, quis nostrud"
     "exercitation ullamco laboris nisi ut aliquip ex ea commodo"
     "consequat. Duis aute irure dolor in reprehenderit in"
     "voluptate velit esse cillum dolore eu fugiat nulla"
     "pariatur. Excepteur sint occaecat cupidatat non proident,"
     "sunt in culpa qui officia deserunt mollit anim id est"
     "laborum.")

    sampling_params = SamplingParams(max_tokens=8,
                                     temperature=0,
                                     logprobs=0,
                                     ignore_eos=True)

    with patch.object(model_runner,
                      "_prepare_chunked_prefill",
                      wraps=model_runner._prepare_chunked_prefill) as spy:
        cp_model.generate(prompt, sampling_params)

    #TODO: validate output
    spy.assert_called()

    assert spy.call_count == 2

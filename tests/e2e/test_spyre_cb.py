"""Verification of continuous batching

Run `python -m pytest tests/e2e/test_spyre_cb.py`.
"""

import pickle
from pathlib import Path
from typing import Any

import pytest
from llm_cache_util import force_engine_shutdown
from openai import BadRequestError
from output_util import check_output_against_hf, extract_output, generate_spyre_vllm_output
from spyre_util import (
    ModelInfo,
    RemoteOpenAIServer,
    create_seq_prompt,
    get_chicken_soup_prompts,
    skip_unsupported_tp_size,
)
from vllm import LLM, SamplingParams

cb_mark = pytest.param("cb", marks=pytest.mark.cb, id="cp")
cp_mark = pytest.param("cp", marks=pytest.mark.chunked_prefill, id="cp")


@pytest.mark.parametrize("mode", [cb_mark, cp_mark])
@pytest.mark.parametrize("backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_cb_max_tokens(
    model: ModelInfo,
    backend: str,
    max_model_len: int,
    max_num_seqs: int,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
    mode: str,
):
    """Test that continuous batches of requests that
    are longer than the `max_model_len` are correctly rejected"""
    max_tokens = 20

    overflow_prompt = " ".join(["a"] * max_model_len)

    vllm_sampling_params = SamplingParams(
        max_tokens=max_tokens, temperature=0, ignore_eos=True, logprobs=0
    )

    # The text of the error raised by vllm changed from 0.11.0 to 0.11.1
    with pytest.raises(ValueError, match="(max model context length|maximum model length)"):
        generate_spyre_vllm_output(
            model=model,
            prompts=overflow_prompt,
            max_model_len=max_model_len,
            sampling_params=vllm_sampling_params,
            tensor_parallel_size=1,
            backend=backend,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=(128 if mode == "cp" else None),
            monkeypatch=monkeypatch,
        )


@pytest.mark.parametrize("mode", [cb_mark, cp_mark])
@pytest.mark.parametrize("backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_api_cb_rejects_oversized_request(
    remote_openai_server: RemoteOpenAIServer,
    model: ModelInfo,
    backend: str,
    max_model_len: int,
    max_num_seqs: int,
    mode: str,
):
    """Verify API rejects request that exceed max_model_len with CB enabled"""

    client = remote_openai_server.get_client()
    overflow_prompt = " ".join(["hi"] * max_model_len)
    max_tokens = 10

    with pytest.raises(BadRequestError, match="maximum context length is"):
        client.completions.create(
            model=model.name,
            prompt=overflow_prompt,
            max_tokens=max_tokens,
        )


@pytest.mark.parametrize("mode", [cb_mark, cp_mark])
@pytest.mark.parametrize("backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_api_cb_generates_correct_max_tokens(
    remote_openai_server: RemoteOpenAIServer,
    model: ModelInfo,
    backend: str,
    max_model_len: int,
    max_num_seqs: int,
    mode: bool,
):
    """Verify API generates the correct numbers of tokens with CB enabled"""

    client = remote_openai_server.get_client()
    max_tokens = 10

    response = client.completions.create(
        model=model.name, prompt=get_chicken_soup_prompts(1), max_tokens=max_tokens, temperature=0
    )

    assert response.usage.completion_tokens == max_tokens

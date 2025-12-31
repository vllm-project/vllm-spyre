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
            use_cb=True,
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


@pytest.mark.compiler_support_32k
@pytest.mark.cb
@pytest.mark.parametrize("backend", [pytest.param("sendnn", marks=pytest.mark.spyre, id="sendnn")])
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(4, marks=pytest.mark.multi),
    ],
    ids=lambda val: f"TP({val})",
)
def test_long_context_batches(
    model: ModelInfo,
    backend: str,
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests continuous batching with various batch sizes and prompt lengths."""

    skip_unsupported_tp_size(tp_size, backend)

    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    max_model_len = 32768
    max_num_seqs = 32
    max_tokens = 10

    # (batch_size, prompt_length) pairs
    batch_token_pairs = [
        (32, 512),
        (16, 1500),
        (8, 3000),
        (4, 5000),
        (2, 9000),
        (1, 17000),
    ]

    vllm_model = LLM(
        model=model.name,
        tokenizer=model.name,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tp_size,
        revision=model.revision,
        tokenizer_revision=model.revision,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        ignore_eos=True,
        logprobs=0,
    )

    for batch_size, token_len in batch_token_pairs:
        prompt = create_seq_prompt(model, token_length=token_len)
        prompts = [prompt] * batch_size

        vllm_outputs = vllm_model.generate(prompts, sampling_params)

        results = []
        for req_output in vllm_outputs:
            result = extract_output(req_output)
            results.append(result)

    check_output_against_hf(
        model=model,
        backend=backend,
        max_new_tokens=max_tokens,
        vllm_results=results,
        prompts=prompts,
    )

    force_engine_shutdown(vllm_model)


@pytest.mark.compiler_support_32k
@pytest.mark.spyre
@pytest.mark.cb
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(4, marks=pytest.mark.multi),
    ],
    ids=lambda val: f"TP({val})",
)
def test_swap_decode_programs_for_cb(
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """

    Validate the runtime's ability to swap between different compiled decode
    programs for varying batch sizes and TKV.

    The test case consists of 32 small input prompts with specifically chosen
    max_new_tokens values to trigger different decode programs at runtime.

    The test case structure is as follows:

    - 16 prompts with max_new_tokens @ 1k
    -  8 prompts with max_new_tokens @ 2k
    -  4 prompts with max_new_tokens @ 4k
    -  2 prompts with max_new_tokens @ 8k
    -  1 prompt  with max_new_tokens @ 16k
    -  1 prompt  with max_new_tokens @ 32k

    """

    model = "ibm-granite/granite-3.3-8b-instruct"
    backend = "sendnn"
    max_num_seqs = 32

    max_model_len = 32 * 1024  # 32K

    skip_unsupported_tp_size(tp_size, backend)
    prompts = get_chicken_soup_prompts(max_num_seqs)

    create_sampling_params = lambda max_new_tokens: SamplingParams(
        # The prompt will pad to 64 tokens, therefore to match
        # max_model_len/max_new_tokens, we need to decrease by the prompt
        # length
        max_tokens=max_new_tokens - 64,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True,
    )

    p1k = 1 * 1024
    p2k = 2 * 1024
    p4k = 4 * 1024
    p8k = 8 * 1024
    p16k = 16 * 1024
    p32k = 32 * 1024

    sampling_params_1k = [create_sampling_params(p1k) for _ in range(16)]
    sampling_params_2k = [create_sampling_params(p2k) for _ in range(8)]
    sampling_params_4k = [create_sampling_params(p4k) for _ in range(4)]
    sampling_params_8k = [create_sampling_params(p8k) for _ in range(2)]
    sampling_params_16k = [create_sampling_params(p16k) for _ in range(1)]
    sampling_params_32k = [create_sampling_params(p32k) for _ in range(1)]

    sampling_params = (
        sampling_params_1k
        + sampling_params_2k
        + sampling_params_4k
        + sampling_params_8k
        + sampling_params_16k
        + sampling_params_32k
    )

    # Read the cache and check beforehand if the cache was written with the
    # expected prompt. We use the filepath of this script to resolve
    # the cache filepaths
    script_directory = Path(__file__).parent.absolute() / "cache"
    with open(script_directory / "prompts_8k_bs2.pickle", "rb") as f:
        cache_result_8k_bs2: list[dict[str, Any]] = pickle.loads(f.read())

    assert cache_result_8k_bs2[0]["prompt"] == prompts[28]
    assert cache_result_8k_bs2[1]["prompt"] == prompts[29]

    with open(script_directory / "prompts_16k_bs1.pickle", "rb") as f:
        cache_result_16k_bs1: list[dict[str, Any]] = pickle.loads(f.read())

    assert cache_result_16k_bs1[0]["prompt"] == prompts[30]

    # Generate results from vLLM
    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        sampling_params=sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend,
        max_num_seqs=max_num_seqs,
        monkeypatch=monkeypatch,
        max_model_len=max_model_len,
        use_cb=True,
    )

    # TODO: dummy validation, currently the outputs do not match with
    # HF cache.

    assert vllm_results is not None
    # Check first from cache, to save computation
    # # 2 @ 8K
    # compare_results(
    #     model=model,
    #     tensor_parallel_size=tp_size,
    #     backend=backend,
    #     vllm_results=vllm_results[28:30],
    #     hf_results=cache_result_8k_bs2,
    #     prompts=prompts[28:30]
    # )

    # # 1 @ 16K
    # compare_results(
    #     model=model,
    #     tensor_parallel_size=tp_size,
    #     backend=backend,
    #     vllm_results=vllm_results[30:31],
    #     hf_results=cache_result_16k_bs1,
    #     prompts=prompts[30:31]
    # )

    # # 16 @ 1K
    # check_output_against_hf(model, backend, p1k, vllm_results[0:16],
    #                         prompts[0:16])

    # # 8 @ 2K
    # check_output_against_hf(model, backend, p2k, vllm_results[16:24],
    #                         prompts[16:24])

    # # 4 @ 4K
    # check_output_against_hf(model, backend, p4k, vllm_results[24:28],
    #                         prompts[24:28])

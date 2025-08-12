"""Verification of continuous batching

Run `python -m pytest tests/e2e/test_spyre_cb.py`.
"""

import pytest
from openai import BadRequestError
from spyre_util import (RemoteOpenAIServer, create_text_prompt,
                        force_engine_shutdown, generate_spyre_vllm_output,
                        get_chicken_soup_prompts, get_spyre_backend_list,
                        get_spyre_model_list, skip_unsupported_tp_size)
from vllm import LLM, SamplingParams


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_cb_max_tokens(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that continuous batches of requests that
    are longer than the `max_model_len` are correctly rejected"""

    max_model_len = 256
    max_tokens = 20

    overflow_prompt = " ".join(["a"] * max_model_len)

    vllm_sampling_params = SamplingParams(max_tokens=max_tokens,
                                          temperature=0,
                                          ignore_eos=True,
                                          logprobs=0)

    with pytest.raises(ValueError, match="max model context length"):
        generate_spyre_vllm_output(model=model,
                                   prompts=overflow_prompt,
                                   max_model_len=max_model_len,
                                   block_size=max_model_len,
                                   sampling_params=vllm_sampling_params,
                                   tensor_parallel_size=1,
                                   backend=backend,
                                   max_num_seqs=2,
                                   use_cb=True,
                                   monkeypatch=monkeypatch)


@pytest.mark.cb
@pytest.mark.parametrize("cb", [True])
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test__api_cb_rejects_oversized_request(
    remote_openai_server: RemoteOpenAIServer,
    model: str,
    backend: str,
    cb: bool,
    max_model_len: int,
    max_num_seqs: int,
):
    """Verify API rejects request that exceed max_model_len with CB enabled"""

    client = remote_openai_server.get_client()
    overflow_prompt = " ".join(["hi"] * max_model_len)
    max_tokens = 10

    with pytest.raises(BadRequestError,
                       match="This model's maximum context length is"):
        client.completions.create(
            model=model,
            prompt=overflow_prompt,
            max_tokens=max_tokens,
        )


@pytest.mark.cb
@pytest.mark.parametrize("cb", [True])
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_api_cb_generates_correct_max_tokens(
    remote_openai_server: RemoteOpenAIServer,
    model: str,
    backend: str,
    cb: bool,
    max_model_len: int,
    max_num_seqs: int,
):
    """Verify API generates the correct numbers of tokens with CB enabled"""

    client = remote_openai_server.get_client()
    max_tokens = 10

    response = client.completions.create(model=model,
                                         prompt=get_chicken_soup_prompts(1),
                                         max_tokens=max_tokens,
                                         temperature=0)

    assert response.usage.completion_tokens == max_tokens


@pytest.mark.cb
@pytest.mark.spyre
@pytest.mark.xfail  # TODO: remove once a spyre-base image supports this
@pytest.mark.parametrize("model", get_spyre_model_list())
def test_continuous_batching_with_long_contexts(model, monkeypatch):
    """Tests that continuous batching generates the same outputs on the spyre
    cards as it does on cpu, when the max context length is set to 4k.
    This ensures that the compiler is generating the correct programs for long
    context cases, but we test here with small prompts for speed.

    Importantly, we're generating the cpu results to compare against using vllm
    as well, instead of using transformers directly. This ensures that the model
    code is all the same, and the only difference is the torch compilation
    backend.
    """
    max_model_len = 4096
    max_num_seqs = 4
    prompts = get_chicken_soup_prompts(4)

    sampling_params = SamplingParams(max_tokens=20,
                                     temperature=0,
                                     ignore_eos=True,
                                     logprobs=0)

    vllm_cpu_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=max_model_len,
        block_size=max_model_len,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        backend="eager",
        max_num_seqs=max_num_seqs,
        use_cb=True,
        monkeypatch=monkeypatch)

    vllm_spyre_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=max_model_len,
        block_size=max_model_len,
        sampling_params=sampling_params,
        tensor_parallel_size=1,
        backend="sendnn",
        max_num_seqs=max_num_seqs,
        use_cb=True,
        monkeypatch=monkeypatch)

    for i in range(len(vllm_cpu_results)):
        # As long as no sequences have top candidate tokens with very close
        # logprobs, the generated text should be identical.
        assert vllm_cpu_results[i]["text"] == vllm_spyre_results[i]["text"]


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(4, marks=pytest.mark.multi),
    ],
    ids=lambda val: f"TP({val})",
)
def test_long_context_batches(
    model: str,
    backend: str,
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Tests continuous batching with various batch sizes and prompt lengths."""

    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    max_model_len = 32768
    max_num_seqs = 32
    max_tokens = 10

    # (batch_size, prompt_length) pairs
    batch_prompt_pairs = [
        (32, 512),
        (16, 1500),
        (8, 3000),
        (4, 5000),
        (2, 9000),
        (1, 17000),
    ]

    skip_unsupported_tp_size(tp_size, backend)

    vllm_model = LLM(
        model=model,
        tokenizer=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        block_size=max_model_len,
        tensor_parallel_size=tp_size,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        ignore_eos=True,
        logprobs=0,
    )

    for batch_size, prompt_len in batch_prompt_pairs:
        prompt = create_text_prompt(model,
                                    min_token_length=prompt_len,
                                    max_token_length=prompt_len + 10)
        prompts = [prompt] * batch_size

        vllm_outputs = vllm_model.generate(prompts, sampling_params)

        results = []
        for req_output in vllm_outputs:
            token_ids = [t for t in req_output.outputs[0].token_ids if t >= 0]
            results.append({
                "text":
                req_output.outputs[0].text,
                "token_ids":
                tuple(token_ids),
                "tokens":
                tuple([
                    req_output.outputs[0].logprobs[i][t].decoded_token
                    for i, t in enumerate(token_ids)
                ]),
                "logprobs":
                tuple([
                    req_output.outputs[0].logprobs[i][t].logprob
                    for i, t in enumerate(token_ids)
                ]),
            })

        assert len(results) == batch_size

    force_engine_shutdown(vllm_model)

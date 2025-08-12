"""Verification of continuous batching

Run `python -m pytest tests/e2e/test_spyre_cb.py`.
"""

import pytest
from openai import BadRequestError
from spyre_util import (RemoteOpenAIServer, generate_spyre_vllm_output,
                        get_chicken_soup_prompts, get_spyre_model_list,
                        skip_unsupported_tp_size)
from vllm import SamplingParams


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


@pytest.mark.spyre
@pytest.mark.cb
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(4, marks=pytest.mark.multi),
    ],
    ids=lambda val: f"TP({val})",
)
@pytest.mark.parametrize("model", get_spyre_model_list())
def test_swap_decode_programs_for_cb(
    model: str,
    tp_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    '''
    
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
    
    NOTE: currently the model `ibm-granite/granite-3.3-8b-instruct` will set 
    VLLM_DT_MAX_BATCH_TKV_LIMIT to a limit of 128K which will change the 
    behavior of how this test runs. To verify set the environment variable 
    VLLM_SPYRE_TEST_MODEL_LIST="ibm-granite/granite-3.3-8b-instruct"
    '''

    backend = 'sendnn'
    max_num_seqs = 32
    # max_model_len = 32 * 1024
    max_model_len = 8 * 1024

    skip_unsupported_tp_size(tp_size, backend)
    # prompts = get_chicken_soup_prompts(max_num_seqs)
    prompts = get_chicken_soup_prompts(max_num_seqs - 2)

    create_sampling_params = lambda max_new_tokens: SamplingParams(
        # The prompt will pad to 64 tokens, therefore to match
        # max_model_len/max_new_tokens, we need to decrease the prompt length
        max_tokens=max_new_tokens - 64,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    p1k = 1 * 128
    p2k = 2 * 128
    p4k = 4 * 128
    p8k = 8 * 128
    p16k = 16 * 128
    p32k = 32 * 128

    sampling_params_1k = [create_sampling_params(p1k) for _ in range(16)]
    sampling_params_2k = [create_sampling_params(p2k) for _ in range(8)]
    sampling_params_4k = [create_sampling_params(p4k) for _ in range(4)]
    sampling_params_8k = [create_sampling_params(p8k) for _ in range(2)]
    sampling_params_16k = [create_sampling_params(p16k) for _ in range(1)]
    sampling_params_32k = [create_sampling_params(p32k) for _ in range(1)]

    sampling_params = sampling_params_1k + sampling_params_2k + \
        sampling_params_4k + sampling_params_8k + sampling_params_16k + \
            sampling_params_32k

    vllm_results = generate_spyre_vllm_output(model=model,
                                              prompts=prompts,
                                              sampling_params=sampling_params,
                                              tensor_parallel_size=tp_size,
                                              backend=backend,
                                              max_num_seqs=max_num_seqs,
                                              monkeypatch=monkeypatch,
                                              block_size=max_model_len,
                                              max_model_len=max_model_len,
                                              use_cb=True)

    # If we passed here then, everything is alright
    assert len(vllm_results) == max_num_seqs

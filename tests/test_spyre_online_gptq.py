import pytest

from tests.spyre_util import RemoteOpenAIServer, get_spyre_model_list


@pytest.mark.parametrize("model", get_spyre_model_list(isGPTQ=True))
@pytest.mark.parametrize("warmup_shape", [[
    (64, 20, 4),
]])
@pytest.mark.parametrize("backend", ["sendnn_decoder"])
@pytest.mark.parametrize("quantization", ["gptq"])
@pytest.mark.parametrize("vllm_version", ["V0"])
def test_gptq_online(model, warmup_shape, backend, quantization, vllm_version):
    """Test online serving with a GPTQ model using the `vllm serve` CLI"""

    warmup_prompt_length = [t[0] for t in warmup_shape]
    warmup_new_tokens = [t[1] for t in warmup_shape]
    warmup_batch_size = [t[2] for t in warmup_shape]
    v1_flag = "1" if vllm_version == "V1" else "0"
    env_dict = {
        "VLLM_SPYRE_WARMUP_PROMPT_LENS":
        ','.join(str(val) for val in warmup_prompt_length),
        "VLLM_SPYRE_WARMUP_NEW_TOKENS":
        ','.join(str(val) for val in warmup_new_tokens),
        "VLLM_SPYRE_WARMUP_BATCH_SIZES":
        ','.join(str(val) for val in warmup_batch_size),
        "VLLM_SPYRE_DYNAMO_BACKEND":
        backend,
        "VLLM_USE_V1":
        v1_flag
    }

    with RemoteOpenAIServer(model, ["--quantization", quantization],
                            env_dict=env_dict) as server:
        client = server.get_client()
        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=5,
                                               temperature=0.0)
        assert len(completion.choices) == 1
        assert len(completion.choices[0].text) > 0

        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=5,
                                               temperature=1.0,
                                               n=2)
        assert len(completion.choices) == 2
        assert len(completion.choices[0].text) > 0

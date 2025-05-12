import openai
import pytest

from tests.spyre_util import (VLLM_VERSIONS, get_spyre_backend_list,
                              get_spyre_model_list)


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("warmup_shape", [[
    (64, 20, 4),
]])
@pytest.mark.parametrize("vllm_version", VLLM_VERSIONS)
def test_openai_serving(remote_openai_server, model, warmup_shape, backend,
                        vllm_version):
    """Test online serving using the `vllm serve` CLI"""

    client = remote_openai_server.get_client()
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

    # Check some basic error handling as well. This is all done in one test
    # now to avoid server boot-up overhead to test each case.
    # To change this we'll need:
    # - A better way to share a server as a test fixture, or
    # - Much less overhead on server boot (e.g. cached compiled graphs)
    with pytest.raises(openai.APIError):
        # Prompt too long should raise
        long_prompt = "Hello " * 1000
        client.completions.create(model=model,
                                  prompt=long_prompt,
                                  max_tokens=500)

    # Short prompt under context length but requesting too many tokens for
    # the warmup shape should return an empty result
    try:
        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=25)
        # V1 should raise
        assert vllm_version == "V0"
        assert len(completion.choices) == 1
        assert len(completion.choices[0].text) == 0
    except openai.BadRequestError as e:
        assert "warmup" in str(e)


@pytest.mark.parametrize("model", get_spyre_model_list(quantization="gptq"))
@pytest.mark.parametrize("backend", ["sendnn_decoder"])
@pytest.mark.parametrize("quantization", ["gptq"])
@pytest.mark.parametrize("warmup_shape", [[(64, 20, 4)]])
@pytest.mark.parametrize("vllm_version", VLLM_VERSIONS)
def test_openai_serving_gptq(remote_openai_server, model, backend,
                             warmup_shape, vllm_version, quantization):
    """Test online serving a GPTQ model with the sendnn_decoder backend only"""

    client = remote_openai_server.get_client()
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

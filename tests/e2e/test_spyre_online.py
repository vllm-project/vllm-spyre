import openai
import pytest

from tests.spyre_util import (RemoteOpenAIServer, get_spyre_backend_list,
                              get_spyre_model_list)
from vllm_spyre.v1.core.scheduler import NO_WARMUP_FIT_STOP_REASON


def get_test_combinations():
    combinations = []

    # Base model tests across all backends
    for backend in get_spyre_backend_list():
        for model in get_spyre_model_list():
            combinations.append((model, backend, None))

    # GPTQ model only tests on sendnn_decoder
    for model in get_spyre_model_list(quantization="gptq"):
        combinations.append((model, "sendnn_decoder", "gptq"))

    return combinations


@pytest.mark.parametrize("model,backend,quantization", get_test_combinations())
@pytest.mark.parametrize("warmup_shape", [[
    (64, 20, 4),
]])
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_openai_serving(remote_openai_server, model, warmup_shape, backend, vllm_version,
                        quantization):
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
    completion = client.completions.create(model=model,
                                           prompt="Hello World!",
                                           max_tokens=25)

    assert len(completion.choices) == 1

    # TODO: V0 and V1 have slight different behavior for requests
    # that do not fit in a warmup shape

    assert len(completion.choices[0].text) == 0
    if vllm_version == 'V1':
        assert completion.choices[0].stop_reason == \
            NO_WARMUP_FIT_STOP_REASON
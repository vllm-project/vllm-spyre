import pytest

from tests.spyre_util import (VLLM_VERSIONS, get_spyre_backend_list,
                              get_spyre_model_list)


@pytest.mark.multi
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("warmup_shape", [[
    (64, 20, 4),
]])
@pytest.mark.parametrize(
    "backend", [b for b in get_spyre_backend_list() if "eager" not in str(b)])
@pytest.mark.parametrize("tensor_parallel_size", ["2"])
@pytest.mark.parametrize("vllm_version", VLLM_VERSIONS)
def test_openai_tp_serving(remote_openai_server, model, warmup_shape, backend,
                           vllm_version, tensor_parallel_size):
    """Test online serving with tensor parallelism using the `vllm serve` CLI"""

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

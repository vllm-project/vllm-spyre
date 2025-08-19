import openai
import pytest
from spyre_util import get_spyre_backend_list, get_spyre_model_list


def _check_result(client, model, max_tokens=8, temperature=0.0, n=1) -> None:
    completion = client.completions.create(
        model=model,
        prompt="Hello World!",
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
    )
    assert len(completion.choices) == n
    assert len(completion.choices[0].text) > 0


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "tp_size",
    [
        pytest.param(1, marks=pytest.mark.basic),
        pytest.param(2, marks=pytest.mark.multi),
        pytest.param(4, marks=pytest.mark.multi),
        pytest.param(8, marks=pytest.mark.multi),
    ],
    ids=lambda val: f"TP({val})",
)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize(
    "warmup_shape",
    [[
        (64, 20, 1),
    ]],
)
@pytest.mark.parametrize("cb",
                         [pytest.param(1, marks=pytest.mark.cb, id="cb"), 0])
@pytest.mark.parametrize("max_num_seqs", [4],
                         ids=lambda val: f"max_num_seqs({val})")
@pytest.mark.parametrize("max_model_len", [256],
                         ids=lambda val: f"max_model_len({val})")
def test_openai_serving(
    remote_openai_server,
    model,
    warmup_shape,
    backend,
    tp_size,
    cb,
    max_num_seqs,
    max_model_len,
):
    """Test online serving using the `vllm serve` CLI"""

    client = remote_openai_server.get_client()

    _check_result(client, model, n=1)
    _check_result(client, model, temperature=1.0, n=2)

    # rest are SB tests
    if cb:
        return
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
        client.completions.create(model=model,
                                  prompt="Hello World!",
                                  max_tokens=25)
    except openai.BadRequestError as e:
        assert "warmup" in str(e)

import openai
import pytest


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


def test_openai_serving(
    remote_openai_server,
    model,
    warmup_shapes,
    backend,
    tp_size,
    mode,
    max_num_seqs,
    max_model_len,
):
    """Test online serving using the `vllm serve` CLI"""

    client = remote_openai_server.get_client()
    model = model.name

    _check_result(client, model, n=1)
    _check_result(client, model, temperature=1.0, n=2)

    # rest are SB tests
    if mode == "sb":
        return
    # Check some basic error handling as well. This is all done in one test
    # now to avoid server boot-up overhead to test each case.
    # To change this we'll need:
    # - A better way to share a server as a test fixture, or
    # - Much less overhead on server boot (e.g. cached compiled graphs)
    with pytest.raises(openai.APIError):
        # Prompt too long should raise
        long_prompt = "Hello " * 1000
        client.completions.create(model=model, prompt=long_prompt, max_tokens=500)

    # Short prompt under context length but requesting too many tokens for
    # the warmup shape should return an empty result
    try:
        client.completions.create(model=model, prompt="Hello World!", max_tokens=25)
    except openai.BadRequestError as e:
        assert "warmup" in str(e)

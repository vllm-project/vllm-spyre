"""Verification of seeded random sampling behavior.

Tests for both single requests and in batched requests. Due to numerical
differences when batching, we cannot compare responses within a batch from a
single request. However, multiple batched requests should exhibit the desired
behavior where the seed induces deterministic sampling.
"""

import math

import pytest
from output_util import generate_spyre_vllm_output, kwargs_for_mode
from spyre_util import ModelInfo, get_chicken_soup_prompts
from vllm import SamplingParams


def _generate_two_outputs(
    seed: int | None = None,
    **kwargs,
):
    """Helper to generate multiple separate requests with the same configuration."""
    max_new_tokens = 4
    prompts = get_chicken_soup_prompts(1) * kwargs["batch_size"]

    sampling_params = SamplingParams(
        min_tokens=max_new_tokens,
        max_tokens=max_new_tokens,
        temperature=kwargs["temperature"],
        logprobs=0,
        ignore_eos=True,
        seed=seed,
    )

    mode_kwargs = kwargs_for_mode(kwargs["mode"])

    results = []
    for _ in range(2):
        outputs = generate_spyre_vllm_output(
            model=kwargs["model"],
            prompts=prompts,
            max_model_len=kwargs["max_model_len"],
            sampling_params=sampling_params,
            tensor_parallel_size=1,
            backend=kwargs["backend"],
            monkeypatch=kwargs["monkeypatch"],
            max_num_seqs=kwargs["max_num_seqs"],
            **mode_kwargs,
        )
        results.append(outputs)

    return results


@pytest.mark.parametrize("temperature", [1.5])
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_seed_deterministic(
    model: ModelInfo,
    temperature: float,
    seed: int,
    batch_size: int,
    max_model_len: int,
    max_num_seqs: int,
    backend: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
    runtime_xfail,
) -> None:
    """Test that seeded sampling produces identical results across requests.

    Tests both single requests (batch_size=1) and batched requests (batch_size>1).
    With seeding, the entire batch should be reproducible across multiple requests,
    even though individual items within a batch may differ from each other due
    to numerical variance when batching.
    """
    results = _generate_two_outputs(**locals())

    results_1, results_2 = results[0], results[1]

    # Verify batch results are identical across requests
    assert len(results_1) == len(results_2) == batch_size

    for i in range(batch_size):
        result_1 = results_1[i]
        result_2 = results_2[i]

        assert result_1["text"] == result_2["text"]
        assert len(result_1["logprobs"]) == len(result_2["logprobs"])

        for token_id_1, logprob_1, token_id_2, logprob_2 in zip(
            result_1["token_ids"],
            result_1["logprobs"],
            result_2["token_ids"],
            result_2["logprobs"],
        ):
            assert token_id_1 == token_id_2
            assert math.isclose(logprob_1, logprob_2, rel_tol=0.1)


@pytest.mark.parametrize("temperature", [1.5])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_seed_variability(
    model: ModelInfo,
    temperature: float,
    batch_size: int,
    max_model_len: int,
    max_num_seqs: int,
    backend: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
) -> None:
    """Test that unseeded sampling produces different results across requests.

    Tests both single requests (batch_size=1) and batched requests (batch_size>1).
    Without seeding, results should vary within a batch and across multiple
    requests.
    """
    results = _generate_two_outputs(**locals())

    results_1, results_2 = results[0], results[1]

    # Verify at least one result differs within the batch
    if batch_size > 1:
        any_different = False
        for i in range(batch_size):
            if i + 1 >= batch_size:
                break
            if results_1[i]["text"] != results_1[i + 1]["text"]:
                any_different = True
                break
        assert any_different, "Unseeded outputs should produce different results within a batch"

    # Verify at least one result differs between requests
    assert len(results_1) == len(results_2) == batch_size

    any_different = False
    for i in range(batch_size):
        if results_1[i]["text"] != results_2[i]["text"]:
            any_different = True
            break

    assert any_different, "Unseeded outputs should produce different results between requests"


# Made with Bob

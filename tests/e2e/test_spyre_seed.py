"""Verification of seeded random sampling to be deterministic

Run `python -m pytest tests/e2e/test_spyre_seed.py`.
"""

import math

import pytest
from output_util import generate_spyre_vllm_output
from spyre_util import DecodeWarmupShapes, ModelInfo, get_chicken_soup_prompts
from vllm import SamplingParams


@pytest.mark.xfail(reason="Failing currently because of output mismatch")
@pytest.mark.parametrize("temperature", [0.1, 1.0])
@pytest.mark.parametrize("seed", [42])
def test_seed(model: ModelInfo, temperature: float, seed: int,
              max_model_len: int, max_num_seqs: int,
              warmup_shapes: DecodeWarmupShapes, backend: str, cb: int,
              monkeypatch: pytest.MonkeyPatch, use_llm_cache) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    output is generated for one request with 5 identical prompts
    using random sampling (non-zero temperature) in combination
    with a seed. The generated output, including text, token ids,
    logprobs is verified to be identical for all 5 sequences.
    '''

    max_new_tokens = warmup_shapes[0][1]

    prompts = get_chicken_soup_prompts(1) * 5

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=temperature,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True,
        seed=seed)

    if bool(cb):
        # Turn off warmup shapes for CB
        warmup_shapes = None

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=warmup_shapes,
        max_model_len=max_model_len,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        use_cb=bool(cb),
        max_num_seqs=max_num_seqs,
        monkeypatch=monkeypatch)

    # compare all generated outputs against the first generated output
    for vllm_result in vllm_results:
        assert vllm_result['text'] == vllm_results[0]['text']

        # compare logprobs for all tokens between
        # the current and the first sequence
        assert len(vllm_result['logprobs']) == len(vllm_results[0]['logprobs'])
        for token_id, logprob, token_id_0, logprob_0 in zip(
                vllm_result['token_ids'], vllm_result['logprobs'],
                vllm_results[0]['token_ids'], vllm_results[0]['logprobs']):
            assert token_id == token_id_0
            assert math.isclose(logprob, logprob_0, rel_tol=0.1)

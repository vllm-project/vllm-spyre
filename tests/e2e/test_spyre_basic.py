"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

import pytest
from output_util import validate_vllm_vs_hf_output, kwargs_for_mode
from spyre_util import (
    ModelInfo,
    create_random_request,
    get_chicken_soup_prompts,
    patch_environment,
    skip_unsupported_tp_size,
)
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import PoolingSpyreScheduler


@pytest.mark.full_model
@pytest.mark.basic
def test_output(
    model: ModelInfo,
    tp_size: int,
    backend: str,
    mode: str,
    max_num_seqs: int,
    max_model_len: int,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
) -> None:
    """
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.

    Configuration for CB - parameters are combinatorial:
        * max_num_seqs: 4
        * tensor parallelism: 1, 2, 4, 8
        * number of prompts: 4 (Chicken soup prompts)
        * max tokens: 20 (same for all the prompts)
    """

    skip_unsupported_tp_size(tp_size, backend)


    prompts = get_chicken_soup_prompts(4)

    kwargs = {
        "max_num_seqs": max_num_seqs,
        "use_cb": True,
        "max_num_batched_tokens": 128 if (mode == "cp" or mode == "pc") else None,
    }

    max_new_tokens = 4

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True,
    )

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend,
        monkeypatch=monkeypatch,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        **kwargs_for_mode(mode, max_num_seqs, warmup_shapes),
    )


def test_batch_handling(
    model: ModelInfo,
    backend: str,
    mode: str,
    max_num_seqs: int,
    max_model_len: int,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
):
    """Test that the spyre worker correctly handles
    continuous batches of requests that
    finish after different numbers of forward passes

    Configuration for CB - parameters are combinatorial:
        * max_num_seqs: 2
        * number of prompts: 4 (Chicken soup prompts)
        * max tokens: [5, 20, 10, 5]
    """

    prompts = get_chicken_soup_prompts(4)
    max_new_tokens = [5, 20, 10, 5]
    vllm_sampling_params = [
        SamplingParams(
            max_tokens=max_new_tokens[i],
            min_tokens=max_new_tokens[i],
            temperature=0,
            ignore_eos=True,
            logprobs=0,
        )
        for i in range(len(max_new_tokens))
    ]

    kwargs = {
        "max_num_seqs": max_num_seqs,
        "use_cb": True,
        "max_num_batched_tokens": 128 if (mode == "cp" or mode == "pc") else None,
    }

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        max_model_len=max_model_len,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        max_new_tokens=max_new_tokens,
        **kwargs_for_mode(mode, max_num_seqs),
    )

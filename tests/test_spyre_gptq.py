"""Verification of vLLM execution of GPTQ models.

Run `python -m pytest tests/test_spyre_gptq.py`.

"""

from typing import List, Tuple

import pytest
from spyre_util import (generate_spyre_vllm_output, get_spyre_backend_list, get_spyre_model_list)
from vllm import SamplingParams


@pytest.mark.parametrize("model", get_spyre_model_list(isGPTQ=True))
@pytest.mark.parametrize("prompts", [[
    "The capital of France is Paris."
    "Provide a list of instructions for preparing"
    " chicken soup for a family of four.", "Hello",
    "What is the weather today like?", "Who are you?"
]])
@pytest.mark.parametrize("warmup_shape", [(64, 20, 4), (64, 20, 8),
                                          (128, 20, 4), (128, 20, 8)]
                         )  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("vllm_version", ["V0"])
def test_gptq_exec(
    model: str,
    prompts: List[str],
    warmup_shape: Tuple[int, int, int],
    backend: str,
    vllm_version: str,
) -> None:
    """
    Tests if a GPTQ model can be loaded and run via vLLM Spyre.
    Verifies that output is generated without crashing.
    """

    max_new_tokens = warmup_shape[1]

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=[warmup_shape],
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        vllm_version=vllm_version,
        quantization="gptq"
    )

    # Ensure results were produced for all prompts
    assert isinstance(vllm_results, list), "Expected a list of results"
    assert len(vllm_results) == len(prompts), \
        f"Expected {len(prompts)} results, but got {len(vllm_results)}"

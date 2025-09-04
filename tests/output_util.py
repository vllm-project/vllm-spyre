"""Contains utilities for generating output and returning the results"""

from typing import Any, Optional, Union

import pytest
from llm_cache import LLM_CACHE, get_cached_llm
from spyre_util import DecodeWarmupShapes
from vllm import LLM, SamplingParams


# vLLM / Spyre
def spyre_vllm_embeddings(
    model: str,
    prompts: list[str],
    max_model_len: int,
    tensor_parallel_size: int,
    backend: str,
) -> list[dict[str, Any]]:
    # NB: This doesn't use the same LLM caching as generate_spyre_vllm_output
    # There aren't as many embedding tests so it's not worth the effort atm to
    # cache

    # Clear any cached decoder model
    LLM_CACHE.clear()

    vllm_model = LLM(
        model=model,
        tokenizer=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
    )

    vllm_outputs = vllm_model.embed(prompts)

    results = []
    for req_output in vllm_outputs:
        result = {}
        result["embeddings"] = req_output.outputs.embedding
        results.append(result)

    return results


# vLLM / Spyre
def generate_spyre_vllm_output(
    model: str,
    prompts: Union[list[str], list[list[int]]],
    max_model_len: int,
    sampling_params: Union[SamplingParams, list[SamplingParams]],
    tensor_parallel_size: int,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    warmup_shapes: DecodeWarmupShapes | None = None,
    max_num_seqs: Optional[int] = None,
    use_cb: bool = False,
) -> list[dict[str, Any]]:
    # Allows to run multiprocess V1 engine without dumping meaningless logs at
    # shutdown engine this context.
    monkeypatch.setenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1")

    vllm_model = get_cached_llm(
        model=model,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        backend=backend,
        monkeypatch=monkeypatch,
        warmup_shapes=warmup_shapes,
        max_num_seqs=max_num_seqs,
        use_cb=use_cb,
    )

    vllm_outputs = vllm_model.generate(prompts, sampling_params)
    results = []

    for req_output in vllm_outputs:
        result = extract_output(req_output)
        results.append(result)

    return results


def extract_output(req_output):
    """Extract text, token_ids, tokens, and logprobs from request output."""

    result = {}
    result["text"] = req_output.outputs[0].text

    # TODO: Workaround for V1, if request does not fit in a warmup shape
    # token_ids may be filled with -1.
    token_ids = [t for t in req_output.outputs[0].token_ids if t >= 0]
    result["token_ids"] = tuple(token_ids)
    result["tokens"] = tuple(req_output.outputs[0].logprobs[i][t].decoded_token
                             for i, t in enumerate(token_ids))
    result["logprobs"] = tuple(
        req_output.outputs[0].logprobs[i][t].logprob \
            for i, t in enumerate(token_ids)
    )

    return result

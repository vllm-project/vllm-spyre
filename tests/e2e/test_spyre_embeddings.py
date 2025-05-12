"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/test_spyre_embeddings.py`.
"""

import pytest
from spyre_util import (compare_embedding_results, get_spyre_backend_list,
                        get_spyre_model_list, spyre_vllm_embeddings,
                        st_embeddings)


@pytest.mark.parametrize("model", get_spyre_model_list(isEmbeddings=True))
@pytest.mark.parametrize("prompts", [[
    "The capital of France is Paris."
    "Provide a list of instructions for preparing"
    " chicken soup for a family of four.", "Hello",
    "What is the weather today like?", "Who are you?"
]])
@pytest.mark.parametrize("warmup_shape",
                         [(64, 4), (64, 8), (128, 4),
                          (128, 8)])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize(
    "vllm_version",
    [pytest.param("V0", marks=pytest.mark.v0, id="v0")
     ])  # TODO: Replace with VLLM_VERSIONS when v1 is supported.
def test_output(
    model: str,
    prompts: list[str],
    warmup_shape: tuple[int, int],
    backend: str,
    vllm_version: str,
) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated embeddings
    are verified to be identical for vLLM and SentenceTransformers.
    '''

    vllm_results = spyre_vllm_embeddings(model=model,
                                         prompts=prompts,
                                         warmup_shapes=[warmup_shape],
                                         max_model_len=256,
                                         block_size=256,
                                         tensor_parallel_size=1,
                                         backend=backend,
                                         vllm_version=vllm_version)

    hf_results = st_embeddings(model=model, prompts=prompts)

    compare_embedding_results(model=model,
                              prompts=prompts,
                              warmup_shapes=[warmup_shape],
                              tensor_parallel_size=1,
                              backend=backend,
                              vllm_results=vllm_results,
                              hf_results=hf_results)

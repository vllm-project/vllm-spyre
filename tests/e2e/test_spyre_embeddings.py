"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_embeddings.py`.
"""

import os

import pytest
from spyre_util import (compare_embedding_results, get_chicken_soup_prompts,
                        get_spyre_backend_list, get_spyre_model_list,
                        spyre_vllm_embeddings, st_embeddings)
from vllm import LLM


@pytest.mark.parametrize("model", get_spyre_model_list(isEmbeddings=True))
@pytest.mark.parametrize("warmup_shape",
                         [(64, 4), (64, 8), (128, 4),
                          (128, 8)])  # (prompt_length/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
# TODO: Add it when v1 is supported.
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_output(
    model: str,
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

    prompts = get_chicken_soup_prompts(1)

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


@pytest.fixture
def example_prompts():
    return [
        "The capital of France is Paris.", "Hello",
        "What is the weather today like?", "Who are you?"
    ]


@pytest.mark.parametrize("warmup_shapes", [
    (64, 1),
    (64, 2),
    (64, 4),
])  # (prompt_length/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("model", get_spyre_model_list(isEmbeddings=True))
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_scheduling_invariance(
    example_prompts,
    model,
    backend,
    warmup_shapes,
    vllm_version,
) -> None:

    os.environ["VLLM_SPYRE_DYNAMO_BACKEND"] = backend
    os.environ['VLLM_USE_V1'] = "1" if vllm_version == "V1" else "0"

    prompts = [str(s).strip() for s in example_prompts]
    reference_embeds = st_embeddings(model, example_prompts)

    vllm_model = LLM(model=model,
                     task="embed",
                     tokenizer=model,
                     max_model_len=256,
                     block_size=256,
                     tensor_parallel_size=1)

    # Four requests with one prompt each
    results = []
    for i in range(4):
        results.append(vllm_model.embed(prompts[i]))

    vllm_outputs = []
    for req_output in results:
        result = {'embeddings': req_output[0].outputs.embedding}
        vllm_outputs.append(result)

    compare_embedding_results(model=model,
                              prompts=example_prompts,
                              warmup_shapes=[warmup_shapes],
                              tensor_parallel_size=1,
                              backend=backend,
                              vllm_results=vllm_outputs,
                              hf_results=reference_embeds)

    # Two requests with two prompt each
    results = []
    for i in range(2):
        results.append(vllm_model.embed([prompts[i * 2], prompts[i * 2 + 1]]))

    vllm_outputs = []
    for req_output in results:
        result1 = {'embeddings': req_output[0].outputs.embedding}
        result2 = {'embeddings': req_output[1].outputs.embedding}

        vllm_outputs.extend([result1, result2])

    compare_embedding_results(model=model,
                              prompts=example_prompts,
                              warmup_shapes=[warmup_shapes],
                              tensor_parallel_size=1,
                              backend=backend,
                              vllm_results=vllm_outputs,
                              hf_results=reference_embeds)

    # One requests with four prompts
    results = vllm_model.embed(prompts)
    vllm_outputs = []
    for req_output in results:
        result1 = {'embeddings': req_output.outputs.embedding}
        vllm_outputs.append(result)

    compare_embedding_results(model=model,
                              prompts=example_prompts,
                              warmup_shapes=[warmup_shapes],
                              tensor_parallel_size=1,
                              backend=backend,
                              vllm_results=vllm_outputs,
                              hf_results=reference_embeds)

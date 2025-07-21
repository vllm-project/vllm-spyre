"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_embeddings.py`.
"""

import os
from functools import partial

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


@pytest.mark.parametrize("warmup_shapes", [
    (64, 1),
    (64, 2),
    (64, 4),
])  # (prompt_length/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("model", get_spyre_model_list(isEmbeddings=True))
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_scheduling_invariance(
    model,
    backend,
    warmup_shapes,
    vllm_version,
) -> None:

    os.environ["VLLM_SPYRE_DYNAMO_BACKEND"] = backend
    os.environ['VLLM_USE_V1'] = "1" if vllm_version == "V1" else "0"

    prompts = get_chicken_soup_prompts(4)
    reference_embeds = st_embeddings(model, prompts)

    vllm_model = LLM(model=model,
                     task="embed",
                     tokenizer=model,
                     max_model_len=256,
                     block_size=256,
                     tensor_parallel_size=1)

    def chunk_embeds(step):
        vllm_outputs = []
        for i in range(0, len(prompts), step):
            emb_outputs = [
                req.outputs for req in vllm_model.embed(prompts[i:i + step])
            ]
            for emb_output in emb_outputs:
                vllm_outputs.append({'embeddings': emb_output})
        return vllm_outputs

    verify_vllm_results = partial(compare_embedding_results,
                                  model=model,
                                  prompts=prompts,
                                  warmup_shapes=[warmup_shapes],
                                  tensor_parallel_size=1,
                                  backend=backend,
                                  hf_results=reference_embeds)

    # Four requests with one prompt each
    verify_vllm_results(vllm_results=chunk_embeds(1))

    # Two requests with two prompt each
    verify_vllm_results(vllm_results=chunk_embeds(2))

    # One requests with four prompts
    verify_vllm_results(vllm_results=chunk_embeds(4))

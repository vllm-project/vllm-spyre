"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_embeddings.py`.
"""

from functools import partial

import pytest
from spyre_util import (compare_embedding_results, get_chicken_soup_prompts,
                        get_spyre_backend_list, get_spyre_model_list,
                        patch_warmup_shapes, spyre_vllm_embeddings,
                        st_embeddings)
from vllm import LLM


@pytest.mark.parametrize("model", get_spyre_model_list(isEmbeddings=True))
@pytest.mark.parametrize(
    "warmup_shape",
    [  # (prompt_length/batch_size)
        pytest.param((64, 4), marks=pytest.mark.basic),
        pytest.param((64, 8)),
        pytest.param((128, 4)),
        pytest.param((128, 8))
    ])
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_output(
    model: str,
    warmup_shape: tuple[int, int],
    backend: str,
    monkeypatch,
) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated embeddings
    are verified to be identical for vLLM and SentenceTransformers.
    '''

    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    patch_warmup_shapes([warmup_shape], monkeypatch)

    prompts = get_chicken_soup_prompts(1)

    vllm_results = spyre_vllm_embeddings(model=model,
                                         prompts=prompts,
                                         max_model_len=256,
                                         block_size=256,
                                         tensor_parallel_size=1,
                                         backend=backend)

    hf_results = st_embeddings(model=model, prompts=prompts)

    compare_embedding_results(model=model,
                              prompts=prompts,
                              warmup_shapes=[warmup_shape],
                              tensor_parallel_size=1,
                              backend=backend,
                              vllm_results=vllm_results,
                              hf_results=hf_results)


@pytest.mark.parametrize("warmup_shape", [
    (128, 1),
    (128, 2),
    (128, 4),
])  # (prompt_length/batch_size)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("model", get_spyre_model_list(isEmbeddings=True))
def test_scheduling_invariance(
    model,
    backend,
    warmup_shape: tuple[int, int],
    monkeypatch,
) -> None:
    '''
    This test is meant to verify that the embedding result are neither
    dependent on the batch size nor the position within the batch.
    We should always get results that are consistent with the reference
    implementation (sentence-transformers).
    To verify this we take a batch of 4 prompts and run it 1) as 4 batches
    of 1; 2) as 2 batches of 2; 3) as 1 batch of 4.
    '''

    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    patch_warmup_shapes([warmup_shape], monkeypatch)

    prompts = get_chicken_soup_prompts(4)
    reference_embeds = st_embeddings(model, prompts)

    vllm_model = LLM(model=model,
                     task="embed",
                     tokenizer=model,
                     max_model_len=256,
                     block_size=256,
                     tensor_parallel_size=1)

    def batch_embeds(step):
        vllm_outputs = []
        for i in range(0, len(prompts), step):
            emb_outputs = [
                req.outputs for req in vllm_model.embed(prompts[i:i + step])
            ]
            for emb_output in emb_outputs:
                vllm_outputs.append({'embeddings': emb_output.embedding})
        return vllm_outputs

    verify_vllm_results = partial(compare_embedding_results,
                                  model=model,
                                  prompts=prompts,
                                  warmup_shapes=[warmup_shape],
                                  tensor_parallel_size=1,
                                  backend=backend,
                                  hf_results=reference_embeds)

    # Four requests with one prompt each
    verify_vllm_results(vllm_results=batch_embeds(1))

    # Two requests with two prompt each
    verify_vllm_results(vllm_results=batch_embeds(2))

    # One requests with four prompts
    verify_vllm_results(vllm_results=batch_embeds(4))

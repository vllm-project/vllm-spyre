import pytest
from sentence_transformers import CrossEncoder
from spyre_util import (get_spyre_backend_list, get_spyre_model_list,
                        patch_warmup_shapes)
from vllm import LLM


@pytest.mark.parametrize("model", get_spyre_model_list(isScoring=True))
@pytest.mark.parametrize(
    "warmup_shape",
    [  # (prompt_length/batch_size)
        pytest.param((64, 4)),
    ])
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.scoring
def test_output(
    model: str,
    warmup_shape: tuple[int, int],
    backend: str,
    monkeypatch,
) -> None:

    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    patch_warmup_shapes([warmup_shape], monkeypatch)

    query = "What is the capital of France?"
    docs = [
        "The capital of France is Paris.", "The capital of Germany is Berlin."
    ]

    ce_model = CrossEncoder(model)
    ce_scores = ce_model.predict([(query, docs[0]), (query, docs[1])])

    vllm_model = LLM(model=model)
    vllm_outputs = vllm_model.score(query, docs)
    vllm_scores = [o.outputs.score for o in vllm_outputs]

    print(f"{ce_scores=}")
    print(f"{vllm_scores=}")

    assert ce_scores[0] == pytest.approx(vllm_scores[0], rel=0.01)
    assert ce_scores[1] == pytest.approx(vllm_scores[1], rel=0.01)

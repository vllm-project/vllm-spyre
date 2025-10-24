import pytest
import requests
from sentence_transformers import CrossEncoder
from spyre_util import get_spyre_backend_list, get_spyre_model_list


@pytest.mark.parametrize("model", get_spyre_model_list(isScoring=True))
@pytest.mark.parametrize(
    "warmup_shapes",
    [  # (prompt_length/new_tokens/batch_size)
        pytest.param([(64, 0, 4)]),
    ],
)
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.scoring
def test_serving(remote_openai_server, model, warmup_shapes, backend):
    """Test online serving using the `vllm serve` CLI"""

    score_url = remote_openai_server.url_for("/score")

    query = "What is the capital of France?"
    # Number of inputs larger than the warmup batch size of 4
    # and with a non-uniform token length
    docs = [
        "The capital of France is Paris.",
        "The capital of Germany is Berlin.",
        "The capital of Brazil is Brasilia.",
        "The capital of the country with the best beer is Berlin.",
        "The capital of the USA is Washington.",
        "The capital city of Spain is Madrid.",
    ]
    vllm_outputs = requests.post(
        url=score_url,
        json={
            "text_1": query,
            "text_2": docs,
        },
    ).json()
    vllm_scores = [o["score"] for o in vllm_outputs["data"]]

    ce_model = CrossEncoder(model.name, revision=model.revision)
    ce_scores = ce_model.predict(
        [
            (query, docs[0]),
            (query, docs[1]),
            (query, docs[2]),
            (query, docs[3]),
            (query, docs[4]),
            (query, docs[5]),
        ]
    )

    assert ce_scores[0] == pytest.approx(vllm_scores[0], rel=0.02)
    assert ce_scores[1] == pytest.approx(vllm_scores[1], rel=0.02)
    assert ce_scores[2] == pytest.approx(vllm_scores[2], rel=0.02)
    assert ce_scores[3] == pytest.approx(vllm_scores[3], rel=0.02)
    assert ce_scores[4] == pytest.approx(vllm_scores[4], rel=0.02)
    assert ce_scores[5] == pytest.approx(vllm_scores[5], rel=0.02)

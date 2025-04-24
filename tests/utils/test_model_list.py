import pytest
from spyre_util import get_spyre_model_list

@pytest.mark.util
def test_get_spyre_model_list(monkeypatch):
    '''
    Ensure we return the model_list correctly
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_MODEL_DIR", "models")
        m.setenv("VLLM_SPYRE_TEST_MODEL_LIST", "llama-194m, " \
                 "all-roberta-large-v1")
        assert get_spyre_model_list()[0] == "models/llama-194m"
        assert get_spyre_model_list()[1] == \
        "models/all-roberta-large-v1"

    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_MODEL_DIR", "")
        m.setenv("VLLM_SPYRE_TEST_MODEL_LIST", "llama-194m, " \
            "all-roberta-large-v1")
        assert get_spyre_model_list()[0] == "llama-194m"
        assert get_spyre_model_list()[1] == "all-roberta-large-v1"
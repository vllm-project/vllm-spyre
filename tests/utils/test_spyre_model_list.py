import pytest
from spyre_util import get_spyre_model_list


@pytest.mark.utils
def test_get_spyre_model_list(monkeypatch):
    '''
    Tests returning the expected models
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_MODEL_DIR", "models")
        m.setenv("VLLM_SPYRE_TEST_MODEL_LIST", "llama-194m, " \
                 "all-roberta-large-v1")
        model_list = get_spyre_model_list()
        assert model_list[0].values[0] == "models/llama-194m"
        assert model_list[1].values[0] == "models/all-roberta-large-v1"

    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_MODEL_DIR", "")
        m.setenv("VLLM_SPYRE_TEST_MODEL_LIST", "llama-194m, " \
            "all-roberta-large-v1")
        model_list = get_spyre_model_list()
        assert model_list[0].values[0] == "llama-194m"
        assert model_list[1].values[0] == "all-roberta-large-v1"

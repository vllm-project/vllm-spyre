import pytest
from spyre_util import get_spyre_model_list, get_spyre_model_list_w_tokenizer


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


# model and tokenizer is the same since
# VLLM_SPYRE_TEST_TOKENIZER_LIST is empty
def test_get_spyre_model_list_w_tokenizer_default(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_MODEL_LIST", "llama-194m, "
                 "all-roberta-large-v1")
        m.setenv("VLLM_SPYRE_TEST_MODEL_DIR", "models")

        model_tokenizer_list = get_spyre_model_list_w_tokenizer()
        # model
        assert model_tokenizer_list[0].values[0] == "models/llama-194m"
        # tokenizer
        assert model_tokenizer_list[0].values[1] == "models/llama-194m"

        # model
        assert model_tokenizer_list[1].values[
            1] == "models/all-roberta-large-v1"
        # tokenizer
        assert model_tokenizer_list[1].values[
            0] == "models/all-roberta-large-v1"


# Models and tokenizers are set based on env values
def test_get_spyre_model_list_w_tokenizer_w_values(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_MODEL_LIST", "llama-194m, "
                 "tiny-granite")
        m.setenv("VLLM_SPYRE_TEST_MODEL_DIR", "models")
        m.setenv("VLLM_SPYRE_TEST_TOKENIZER_LIST", "models/llama-tokenizer, "
                 "ibm-granite/granite-3.2-8b")

        model_tokenizer_list = get_spyre_model_list_w_tokenizer()
        # model
        assert model_tokenizer_list[0].values[0] == "models/llama-194m"
        # tokenizer
        assert model_tokenizer_list[0].values[1] == "models/llama-tokenizer"
        # id used in tests
        assert (model_tokenizer_list[0].id ==
                "model(models/llama-194m), tokenizer(models/llama-tokenizer)")

        # model
        assert model_tokenizer_list[1].values[0] == "models/tiny-granite"
        # tokenizer
        assert model_tokenizer_list[1].values[
            1] == "ibm-granite/granite-3.2-8b"
        # id used in tests
        assert (
            model_tokenizer_list[1].id ==
            "model(models/tiny-granite), tokenizer(ibm-granite/granite-3.2-8b)"
        )

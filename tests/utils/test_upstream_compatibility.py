import os

import pytest

pytestmark = pytest.mark.compat

VLLM_VERSION = os.getenv("TEST_VLLM_VERSION", "default")


def test_vllm_bert_support(monkeypatch):
    '''
    Test if the vllm version under test already has Bert support for V1
    '''

    from vllm.model_executor.models.bert import BertEmbeddingModel

    bert_supports_v0_only = getattr(BertEmbeddingModel, "supports_v0_only",
                                    False)

    if VLLM_VERSION == "main":
        assert not bert_supports_v0_only
    else:
        assert bert_supports_v0_only, (
            "The currently supported vLLM version already"
            "supports Bert in V1. Remove the compatibility workarounds.")

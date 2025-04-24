import pytest
from spyre_util import get_spyre_backend_list


@pytest.mark.utils
def test_get_spyre_backend_list(monkeypatch):
    '''
    Ensure we return the backend list correctly
    '''
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_BACKEND_LIST",
                 "eager,inductor,sendnn_decoder")
        backend_list = get_spyre_backend_list()
        assert backend_list[0].values[0] == "eager"
        assert backend_list[1].values[0] == "inductor"
        assert backend_list[2].values[0] == "sendnn_decoder"

    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_TEST_BACKEND_LIST", "sendnn")
        backend_list = get_spyre_backend_list()
        assert backend_list[0].values[0] == "sendnn"

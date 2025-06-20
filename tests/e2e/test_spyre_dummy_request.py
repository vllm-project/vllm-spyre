import pytest
from spyre_util import get_spyre_model_list
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
@pytest.mark.parametrize("vllm_version",
                         [pytest.param("V1", marks=pytest.mark.v1, id="v1")])
def test_cb_dummy_req_creation(model: str, backend: str, vllm_version: str,
                               monkeypatch: pytest.MonkeyPatch):
    """Test that the a dummy request is created 
        when there is only one request."""
    with monkeypatch.context() as m:
        max_tokens = 20
        max_num_seqs = 2  # defines max batch size
        prompt1 = "7 6 5 4"
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            # stop="1",  # replace with stop_tokens
            ignore_eos=True,
        )
        # set env vars
        m.setenv("VLLM_SPYRE_USE_CB", "1")
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
        # To get deterministic execution in V1
        # and to enable InprocClient
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # start the engine
        engine_args = EngineArgs(model=model,
                                 max_model_len=2048,
                                 max_num_seqs=max_num_seqs)

        engine = V1LLMEngine.from_engine_args(engine_args)
        engine_core = engine.engine_core.engine_core

        requests = engine_core.model_executor.driver_worker.\
                                        worker.model_runner.req_ids2blocks
        dummy_requests = engine_core.model_executor.driver_worker.\
                                        worker.model_runner.dummy_req_ids2blocks

        assert len(requests) == 0
        assert len(dummy_requests) == 0

        engine.add_request("1", prompt1, sampling_params)
        engine.step()

        requests = engine_core.model_executor.driver_worker.\
                                            worker.model_runner.req_ids2blocks
        dummy_requests = engine_core.model_executor.driver_worker.\
                                            worker.model_runner.dummy_req_ids2blocks
        engine.step()
        requests = engine_core.model_executor.driver_worker.\
                                            worker.model_runner.req_ids2blocks
        dummy_requests = engine_core.model_executor.driver_worker.\
                                            worker.model_runner.dummy_req_ids2blocks

        assert len(requests) == 1
        assert len(dummy_requests) > 0


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
@pytest.mark.parametrize("vllm_version",
                         [pytest.param("V1", marks=pytest.mark.v1, id="v1")])
def test_cb_dummy_req_removal_after_new_req(model: str, backend: str,
                                            vllm_version: str,
                                            monkeypatch: pytest.MonkeyPatch):
    """Test that the a dummy request is removed
        when another request is received."""
    with monkeypatch.context() as m:
        max_tokens = 20
        max_num_seqs = 2  # defines max batch size
        prompt1 = "7 6 5 4"
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            # stop="1",  # replace with stop_tokens
            ignore_eos=True,
        )
        # set env vars
        m.setenv("VLLM_SPYRE_USE_CB", "1")
        m.setenv("VLLM_USE_V1", "1")
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
        # To get deterministic execution in V1
        # and to enable InprocClient
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # start the engine
        engine_args = EngineArgs(model=model,
                                 max_model_len=2048,
                                 max_num_seqs=max_num_seqs)

        engine = V1LLMEngine.from_engine_args(engine_args)
        engine_core = engine.engine_core.engine_core

        requests = engine_core.model_executor.driver_worker.\
                                        worker.model_runner.req_ids2blocks
        dummy_requests = engine_core.model_executor.driver_worker.\
                                        worker.model_runner.dummy_req_ids2blocks

        # ensure there are no requests at the beginning
        assert len(requests) == 0
        assert len(dummy_requests) == 0

        engine.add_request("1", prompt1, sampling_params)

        # two steps to pass the dummy request creation
        engine.step()
        engine.step()

        dummy_requests_id = dummy_requests

        engine.add_request("2", prompt1, sampling_params)
        engine.step()

        requests = engine_core.model_executor.driver_worker.\
                                    worker.model_runner.req_ids2blocks
        dummy_requests = engine_core.model_executor.driver_worker.\
                                    worker.model_runner.dummy_req_ids2blocks

        assert len(requests) == 2
        assert len(dummy_requests) == 0

        free_blocks = engine_core.model_executor.driver_worker.\
                                            worker.model_runner.free_blocks

        # ensure blocks used by dummy requests are free
        assert dummy_requests_id[0] in free_blocks
        assert dummy_requests_id[1] in free_blocks

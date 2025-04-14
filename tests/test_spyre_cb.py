"""Verification of continuous batching

Run `python -m pytest tests/test_spyre_cb.py`.
"""

import pytest
from spyre_util import generate_cb_spyre_vllm_output, get_spyre_model_list
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine


@pytest.mark.parametrize("max_num_seqs", [1, 2, 3, 4])
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_cb_handling(
    model: str,
    backend: str,
    max_num_seqs: int,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the spyre worker correctly handles
    continuous batches of requests that
    finish after different numbers of forward passes"""

    vllm_sampling_params = SamplingParams(max_tokens=20,
                                          temperature=0,
                                          stop="1",
                                          ignore_eos=True)

    # These prompts are ordered so that they don't finish in the
    # order given
    prompts = [
        "7 6 5 4",
        "10 9 8 7",
        "8 7 6 5",
        "9 8 7 6",
    ]

    # Ensure that both:
    # - The model doesn't crash
    # - The output sequences are correct
    vllm_results = generate_cb_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        max_num_seqs=max_num_seqs,
        monkeypatch=monkeypatch,
    )

    assert vllm_results[0]["text"] == " 3 2 "
    assert vllm_results[1]["text"] == " 6 5 4 3 2 "
    assert vllm_results[2]["text"] == " 4 3 2 "
    assert vllm_results[3]["text"] == " 5 4 3 2 "


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_cb_with_steps(model: str, backend: str,
                       monkeypatch: pytest.MonkeyPatch):
    """Test that the spyre worker correctly handles
    continuous batches of requests and one sequence
    an exit the batch while a new ones gets prefilled
    and appended to the batch"""
    with monkeypatch.context() as m:

        max_tokens = 20
        max_num_seqs = 2  # defines max batch size

        prompt1 = "7 6 5 4"
        prompt2 = "10 9 8 7"
        prompt3 = "8 7 6 5"

        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         temperature=0,
                                         stop="1",
                                         ignore_eos=True)

        # set env vars
        m.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS", "64")
        m.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS", str(max_tokens))

        m.setenv("VLLM_SPYRE_USE_CB", "1")
        m.setenv("VLLM_USE_V1", "1")

        m.setenv("VLLM_SPYRE_MAX_CONTEXT_LENGTH", "2048")
        m.setenv("VLLM_SPYRE_MAX_BATCH_SIZE", str(max_num_seqs))
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

        # To get deterministic execution in V1
        # and to enable InprocClient
        m.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        # start the engine
        engine_args = EngineArgs(model=model, )

        engine = V1LLMEngine.from_engine_args(engine_args)
        engine_core = engine.engine_core.engine_core

        # add first request
        engine.add_request("1", prompt1, sampling_params)

        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0

        request_outputs = engine.step()
        assert len(request_outputs) == 1  # only 1 request
        assert request_outputs[0].request_id == "1"  # req 1 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        # add another request
        engine.add_request("2", prompt2, sampling_params)

        assert len(engine_core.scheduler.waiting) == 1  # req 2 is waiting
        assert len(engine_core.scheduler.running) == 1

        request_outputs = engine.step()
        assert len(request_outputs) == 1  # still only 1 request
        assert request_outputs[0].request_id == "2"  # req 2 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        request_outputs = engine.step()
        assert len(request_outputs) == 2  # still only 1 request
        assert request_outputs[0].request_id == "1"  # req 1 is decoding
        assert request_outputs[1].request_id == "2"  # req 2 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        request_outputs = engine.step()
        assert len(request_outputs) == 2
        assert request_outputs[0].request_id == "1"  # req 1 is decoding
        assert request_outputs[1].request_id == "2"  # req 2 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        request_outputs = engine.step()
        assert len(request_outputs) == 2
        assert request_outputs[0].request_id == "1"  # req 1 is decoding
        assert request_outputs[1].request_id == "2"  # req 2 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        # add the third request
        # but since VLLM_SPYRE_MAX_BATCH_SIZE=2
        # this request is waiting
        engine.add_request("3", prompt3, sampling_params)

        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 2

        request_outputs = engine.step()
        assert len(request_outputs) == 2  # both requests decoding now
        assert request_outputs[0].request_id == "1"  # req 1 is decoding
        assert request_outputs[1].request_id == "2"  # req 2 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        request_outputs = engine.step()
        assert len(request_outputs) == 2  # both requests decoding now
        assert request_outputs[0].request_id == "1"  # req 1 is decoding
        assert request_outputs[1].request_id == "2"  # req 2 is decoding
        assert request_outputs[0].finished  # request 1 is done
        assert request_outputs[0].outputs[0].text == " 3 2 "

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        # req 3 is scheduled now
        request_outputs = engine.step()
        assert len(request_outputs) == 1
        assert request_outputs[0].request_id == "3"  # req 3 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        request_outputs = engine.step()
        assert len(request_outputs) == 2  # requests 2 and 3 decoding now
        assert request_outputs[0].request_id == "2"  # req 2 is decoding
        assert request_outputs[1].request_id == "3"  # req 3 is decoding

        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

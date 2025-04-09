"""Verification of continuous batching

Run `python -m pytest tests/test_spyre_cb.py`.
"""

import time
import uuid

import pytest
from spyre_util import generate_cb_spyre_vllm_output, get_spyre_model_list
from transformers import AutoTokenizer
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
from vllm.v1.executor.abstract import Executor


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_cb_handling(
    model: str,
    backend: str,
):
    """Test that the spyre worker correctly handles 
    continuous batches of requests that
    finish after different numbers of forward passes"""

    max_tokens1 = 20
    max_tokens2 = 17
    max_tokens3 = 11

    sampling_params1 = SamplingParams(max_tokens=max_tokens1,
                                      temperature=0.0,
                                      stop="1",
                                      ignore_eos=True)

    sampling_params2 = SamplingParams(max_tokens=max_tokens2,
                                      temperature=0.0,
                                      stop="1",
                                      ignore_eos=True)

    sampling_params3 = SamplingParams(max_tokens=max_tokens3,
                                      temperature=0.0,
                                      stop="1",
                                      ignore_eos=True)

    vllm_sampling_params = [
        sampling_params1,
        sampling_params2,
        sampling_params3,
    ]

    # These prompts are ordered so that they don't finish in the
    # order given
    prompts = [
        "7 6 5 4",
        "10 9 8 7",
        "8 7 6 5",
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
    )

    assert vllm_results[0]["text"] == " 3 2 "
    assert vllm_results[1]["text"] == " 6 5 4 3 2 "
    assert vllm_results[2]["text"] == " 4 3 2 "


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_cb_with_engine_steps(model: str, backend: str,
                              monkeypatch: pytest.MonkeyPatch):
    """Test that the spyre worker correctly handles 
    continuous batches of requests and one sequence 
    an exit the batch while a new ones gets prefilled 
    and appended to the batch"""
    with monkeypatch.context() as m:
        max_tokens1 = 20
        max_tokens2 = 17

        sampling_params1 = SamplingParams(max_tokens=max_tokens1,
                                          temperature=0.0,
                                          stop="1",
                                          ignore_eos=True)

        sampling_params2 = SamplingParams(max_tokens=max_tokens2,
                                          temperature=0.0,
                                          stop="1",
                                          ignore_eos=True)

        prompt1 = "7 6 5 4"
        prompt2 = "10 9 8 7"

        # set env vars
        m.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS", "64")
        m.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS",
                 str(max([max_tokens1, max_tokens2])))

        m.setenv("VLLM_SPYRE_USE_CB", "1")
        m.setenv("VLLM_USE_V1", "1")

        m.setenv("VLLM_SPYRE_MAX_CONTEXT_LENGTH", "2048")
        m.setenv("VLLM_SPYRE_MAX_BATCH_SIZE", str(2))
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

        # start the engine
        engine_args = EngineArgs(
            model=model,
            enforce_eager=True,  # reduce test time
        )

        engine = V1LLMEngine.from_engine_args(engine_args)

        # add first request
        engine.add_request("0", prompt1, sampling_params1)

        request_outputs = engine.step()
        assert len(request_outputs) == 1  # only 1 request
        assert request_outputs[0].request_id == "0"

        request_outputs = engine.step()
        assert len(request_outputs) == 1  # still only 1 request
        assert request_outputs[0].request_id == "0"

        # add another request
        engine.add_request("1", prompt2, sampling_params2)

        request_outputs = engine.step()
        assert len(request_outputs) == 1  # still only 1 request
        assert request_outputs[0].request_id == "0"

        request_outputs = engine.step()
        assert len(request_outputs) == 1  # still only 1 request
        assert request_outputs[0].request_id == "1"  # req 1 is decoding

        request_outputs = engine.step()
        assert len(request_outputs) == 2  # both requests decoding now

        request_outputs = engine.step()
        assert len(request_outputs) == 2  # both requests decoding now

        request_outputs = engine.step()
        assert len(request_outputs) == 2  # both requests decoding now
        assert request_outputs[0].finished  # request 1 is done
        assert request_outputs[0].outputs[0].text == " 3 2 "

        request_outputs = engine.step()
        assert len(request_outputs) == 1  # only req #2 is left
        assert request_outputs[0].request_id == "1"  # req 1 is decoding


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", ["eager", "inductor"])
def test_cb_with_core_steps(model: str, backend: str,
                            monkeypatch: pytest.MonkeyPatch):
    """Test that the spyre worker correctly handles
    continuous batches of requests and one sequence
    an exit the batch while a new ones gets prefilled
    and appended to the batch"""
    with monkeypatch.context() as m:
        max_tokens1 = 20
        max_tokens2 = 17

        sampling_params1 = SamplingParams(max_tokens=max_tokens1,
                                          temperature=0.0,
                                          stop="1",
                                          ignore_eos=True)

        sampling_params2 = SamplingParams(max_tokens=max_tokens2,
                                          temperature=0.0,
                                          stop="1",
                                          ignore_eos=True)

        prompt1 = "7 6 5 4"
        prompt2 = "10 9 8 7"

        # set env vars
        m.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS", "64")
        m.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS",
                 str(max([max_tokens1, max_tokens2])))

        m.setenv("VLLM_SPYRE_USE_CB", "1")
        m.setenv("VLLM_USE_V1", "1")

        m.setenv("VLLM_SPYRE_MAX_CONTEXT_LENGTH", "2048")
        m.setenv("VLLM_SPYRE_MAX_BATCH_SIZE", str(2))
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

        # start the engine
        engine_args = EngineArgs(model=model, )

        vllm_config = engine_args.create_engine_config()
        executor_class = Executor.get_class(vllm_config)

        engine_core = EngineCore(vllm_config=vllm_config,
                                 executor_class=executor_class,
                                 log_stats=True)
        # engine = V1LLMEngine.from_engine_args(engine_args)

        tokenizer = AutoTokenizer.from_pretrained(model)
        prompt_tokens1 = tokenizer(prompt1).input_ids
        prompt_tokens2 = tokenizer(prompt2).input_ids

        # # add first request
        engine_core.add_request(
            EngineCoreRequest(
                request_id=uuid.uuid4(),
                prompt=prompt1,
                prompt_token_ids=prompt_tokens1,
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=sampling_params1,
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
            ))
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 0

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        engine_core.add_request(
            EngineCoreRequest(
                request_id=uuid.uuid4(),
                prompt=prompt2,
                prompt_token_ids=prompt_tokens2,
                mm_inputs=None,
                mm_hashes=None,
                mm_placeholders=None,
                sampling_params=sampling_params2,
                eos_token_id=None,
                arrival_time=time.time(),
                lora_request=None,
            ))
        assert len(engine_core.scheduler.waiting) == 1
        assert len(engine_core.scheduler.running) == 1

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 1

        _ = engine_core.step()
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

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

        _ = engine_core.step()
        assert len(engine_core.scheduler.waiting) == 0
        assert len(engine_core.scheduler.running) == 2

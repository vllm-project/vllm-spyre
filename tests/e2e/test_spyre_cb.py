"""Verification of continuous batching

Run `python -m pytest tests/test_spyre_cb.py`.
"""

from collections import deque
from typing import Any

import pytest
from spyre_util import generate_cb_spyre_vllm_output, get_spyre_model_list
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.engine.llm_engine import LLMEngine as V1LLMEngine
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import ContinuousBatchingSpyreScheduler


@pytest.mark.parametrize("max_num_seqs", [2, 3, 4],
                         ids=lambda val: f"max_num_seqs({val})")
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
@pytest.mark.parametrize("cb",
                         [pytest.param(1, marks=pytest.mark.cb, id="cb")])
# commenting v1 since we don't want this test to run with v1 marker yet
# @pytest.mark.parametrize("vllm_version",
#                          [pytest.param("V1", marks=pytest.mark.v1, id="v1")])
@pytest.mark.parametrize(
    "prompts",
    [
        [
            "7 6 5 4",
            "10 9 8 7",
        ],
        [
            "7 6 5 4",
            "10 9 8 7",
            "8 7 6 5",
        ],
        [
            "7 6 5 4",
            "10 9 8 7",
            "8 7 6 5",
            "9 8 7 6",
        ],
    ],
    ids=lambda val: f"num_prompts({len(val)})",
)
def test_cb_handling(
    model: str,
    backend: str,
    max_num_seqs: int,
    cb: int,
    prompts: list[str],
    # vllm_version: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that the spyre worker correctly handles
    continuous batches of requests that
    finish after different numbers of forward passes"""

    vllm_sampling_params = SamplingParams(max_tokens=20,
                                          temperature=0,
                                          stop="1",
                                          ignore_eos=True,
                                          logprobs=0)

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
        use_cb=cb,
        monkeypatch=monkeypatch,
    )

    for i, prompt in enumerate(prompts):
        assert (vllm_results[i]["text"] == [
            " " + " ".join(
                str(i)
                for i in range(int(prompt.split()[-1]) - 1, 1, -1)) + " "
        ][0])


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
# @pytest.mark.parametrize("vllm_version",
#                          [pytest.param("V1", marks=pytest.mark.v1, id="v1")])
def test_cb_with_steps(model: str, backend: str,
                       monkeypatch: pytest.MonkeyPatch):
    """Test that the spyre worker correctly handles
    continuous batches of requests and one sequence
    an exit the batch while a new ones gets prefilled
    and appended to the batch"""

    max_tokens = 20
    max_num_seqs = 2  # defines max batch size

    prompt1 = "7 6 5 4"
    prompt2 = "10 9 8 7"
    prompt3 = "8 7 6 5"

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0,
        stop="1",
        ignore_eos=True,
    )

    # set env vars
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

    # To get deterministic execution in V1
    # and to enable InprocClient
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # start the engine
    engine_args = EngineArgs(model=model,
                             max_model_len=2048,
                             max_num_seqs=max_num_seqs)

    engine = V1LLMEngine.from_engine_args(engine_args)
    engine_core = engine.engine_core.engine_core

    # add first request
    engine.add_request("1", prompt1, sampling_params)

    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 0

    request_outputs = engine.step()
    assert len(request_outputs) == 1  # only 1 request
    assert request_outputs[0].request_id == "1"  # req 1 is decoding (prefill step)

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # add another request
    engine.add_request("2", prompt2, sampling_params)

    assert len(engine_core.scheduler.waiting) == 1  # req 2 is waiting
    assert len(engine_core.scheduler.running) == 1

    request_outputs = engine.step()
    assert len(request_outputs) == 1  # still only 1 request
    assert request_outputs[0].request_id == "2"  # req 2 is decoding (prefill step)

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    request_outputs = engine.step()
    assert len(request_outputs) == 2  # both requests decoding now
    assert request_outputs[0].request_id == "2"  # req 2 is decoding
    assert request_outputs[1].request_id == "1"  # req 1 is decoding

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    request_outputs = engine.step()
    assert len(request_outputs) == 2  # both requests decoding now
    assert request_outputs[0].request_id == "2"  # req 2 is decoding
    assert request_outputs[1].request_id == "1"  # req 1 is decoding

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    request_outputs = engine.step()
    assert len(request_outputs) == 2  # both requests decoding now
    assert request_outputs[0].request_id == "2"  # req 2 is decoding
    assert request_outputs[1].request_id == "1"  # req 1 is decoding

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
    assert request_outputs[0].request_id == "2"  # req 2 is decoding
    assert request_outputs[1].request_id == "1"  # req 1 is decoding

    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 2

    request_outputs = engine.step()
    assert len(request_outputs) == 2  # both requests decoding now
    assert request_outputs[0].request_id == "2"  # req 2 is decoding
    assert request_outputs[1].request_id == "1"  # req 1 is decoding
    assert request_outputs[1].finished  # request 1 is done
    assert request_outputs[1].outputs[0].text == " 3 2 "

    assert len(engine_core.scheduler.waiting) == 1
    assert len(engine_core.scheduler.running) == 1

    # req 3 is scheduled now
    request_outputs = engine.step()
    assert len(request_outputs) == 1
    assert request_outputs[0].request_id == "3"  # req 3 is decoding (prefill step)

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2

    request_outputs = engine.step()
    assert len(request_outputs) == 2  # requests 2 and 3 decoding now
    assert request_outputs[0].request_id == "3"  # req 3 is decoding
    assert request_outputs[1].request_id == "2"  # req 2 is decoding

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2
    request_outputs = engine.step()
    assert len(request_outputs) == 2  # requests 2 and 3 decoding now
    assert request_outputs[0].request_id == "3"  # req 3 is decoding
    assert request_outputs[1].request_id == "2"  # req 2 is decoding

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2
    request_outputs = engine.step()
    assert len(request_outputs) == 2  # requests 2 and 3 decoding now
    assert request_outputs[0].request_id == "3"  # req 3 is decoding
    assert request_outputs[1].request_id == "2"  # req 2 is decoding

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2
    request_outputs = engine.step()
    assert len(request_outputs) == 2  # requests 2 and 3 decoding now
    assert request_outputs[0].request_id == "3"  # req 3 is decoding
    assert request_outputs[1].request_id == "2"  # req 2 is decoding

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2
    request_outputs = engine.step()
    assert len(request_outputs) == 2  # requests 2 and 3 decoding now
    assert request_outputs[0].request_id == "3"  # req 3 is decoding
    assert request_outputs[1].request_id == "2"  # req 2 is decoding

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 2
    request_outputs = engine.step()
    assert len(request_outputs) == 2  # requests 2 and 3 decoding now
    assert request_outputs[0].request_id == "3"  # req 3 is decoding
    assert request_outputs[1].request_id == "2"  # req 2 is decoding
    assert request_outputs[1].finished  # request 2 is done
    assert request_outputs[1].outputs[0].text == " 6 5 4 3 2 "

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1


def create_random_request(
        request_id: int, num_tokens: int,
        sampling_params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=str(request_id),
        prompt_token_ids=[request_id] * num_tokens,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=sampling_params,
        eos_token_id=None,
        arrival_time=0,
        lora_request=None,
    )


def get_params_test_blocks_borders_aligned_prompts():
    seqs_max_tokens = [65, 67, 7]
    prompts_lengths = [49, 41, 7]
    steps_add_reqs = [0, 0, 0]  # add all requests in the beginning

    checked_steps = [{
        "step": 0,
        "tkv": 0,
        "num_waiting": 3,
        "num_running": 0,
        "request_outputs": []
    }, {
        "step": 1,  # Prefill sequence 0
        "tkv": 64,
        "num_waiting": 2,
        "num_running": 1,
        "request_outputs": ["0"]  
    }, {
        "step": 2,  # Prefill sequence 1
        "tkv": 64,  # Still 64 because this step is also a prefill
        "num_waiting": 1,
        "num_running": 2,
        "request_outputs": ["1"]  
    }, {
        "step": 3,
        "tkv": 65,  # Two decodes increases the tkv
        "num_waiting": 1,
        "num_running": 2,
        "request_outputs": ["1", "0"]  # Two sequences are decoded
    }, {
        "step": 4,  # Check normal decode continuation
        "tkv": 66,
        "num_waiting": 1,
        "num_running": 2,
        "request_outputs": ["1", "0"]
    }, {
        "step": 65,  # Last step before fist sequence finishes
        "tkv": 127,
        "num_waiting": 1,
        "num_running": 2,
        "request_outputs": ["1", "0"]
    # }, {
    #     "step": 66,  # Sequence 0 finishes at step 66 (2 prefills + 64 decodes)
    #     "tkv": 128,
    #     "num_waiting": 1,
    #     "num_running": 1,
    #     "request_outputs": ["1"]  # TODO this fails?? Why do we observe ["1", "0"]?
    }, {
        "step": 67,  # Prefill sequence 2
        "tkv": 128,  # Tkv doesn't increase because it is a prefill
        "num_waiting": 0,
        "num_running": 2,
        "request_outputs": ["2"]
    }, {
        "step": 68,  # Decode sequences 1 and 2
        "tkv": 129,
        "num_waiting": 0,
        "num_running": 2,
        "request_outputs": ["2", "1"]
    }, {
        # Sequence 1 finishes at step 71 (start step + 2 prefills + 67 decodes)
        "step": 71,
        "tkv": 132,
        "num_waiting": 0,
        "num_running": 1,
        "request_outputs": ["2"]
    }, {
        "step": 72,  # Decode sequence 2
        "tkv": 133,
        "num_waiting": 0,
        "num_running": 1,
        "request_outputs": ["2"]
    }, {
        # Sequence 2 finishes at step 73 (start step + 1 prefill + 7 decodes)
        "step": 73,  
        "tkv": 134,
        "num_waiting": 0,
        "num_running": 1,
        "request_outputs": ["2"]
    }]
    
    last_step = 73
    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            last_step)


@pytest.mark.a
@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize(
    "seqs_max_tokens,prompts_lengths,steps_add_reqs,checked_steps,last_step",
    [
        # TODO: add all sensitive steps to testing lists
        get_params_test_blocks_borders_aligned_prompts(),
        # get_params_test_blocks_borders_misaligned_prompts(),  # TODO
        # get_params_test_new_prompt_arrives_when_one_finishes(),  # TODO
    ])
def test_scheduler_cb_steps_tkv(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    max_num_seqs: int,
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    steps_add_reqs: list[int],
    checked_steps: list[dict[str, Any]],
    last_step: int,
):
    """
    Test that the scheduler correctly schedules requests and that the 
    tkv produced by the model runner is correct at each step.
    Tested for different scenarios: 
    * prompts aligning with the boundaries of the blocks, 
    * prompts misaligning with the boundaries of the blocks
    """

    # set env vars
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

    # To get deterministic execution in V1
    # and to enable InprocClient
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    # Input parameters sanity check, not actual testing
    # ------
    if not (len(prompts_lengths) == len(seqs_max_tokens)
            and len(prompts_lengths) == len(steps_add_reqs)):
        raise ValueError(
            "Number of prompts should be consistent with number of max tokens."
        )

    if not (steps_add_reqs == sorted(steps_add_reqs)
            and steps_add_reqs[0] == 0):
        raise ValueError(
            "The list of steps where requests are added should be increasing "
            "start with 0")

    if not (checked_steps == sorted(checked_steps, key=lambda x: x["step"])
            and len(checked_steps) == len(set(x["step"]
                                              for x in checked_steps))):
        raise ValueError(
            "List of checked steps needs to be of increasing order of step")
    checked_steps = deque(checked_steps)
    # ------

    # Setup the engine
    engine_args = EngineArgs(model=model,
                             tokenizer=model,
                             max_model_len=2048,
                             block_size=2048,
                             max_num_seqs=max_num_seqs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(vllm_config=vllm_config,
                             executor_class=executor_class,
                             log_stats=False)
    scheduler: ContinuousBatchingSpyreScheduler = engine_core.scheduler

    # Create random requests of specified lengths and max_tokens
    sorted_reqs_params = zip(steps_add_reqs, seqs_max_tokens, prompts_lengths)
    requests: deque[tuple[int, EngineCoreRequest]] = deque()
    for i, (add_step, max_tokens,
            prompt_length) in enumerate(sorted_reqs_params):
        # ignoring eos because we want to force the decoding to finish
        # after max_tokens exactly
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         temperature=0.0,
                                         ignore_eos=True)
        request = create_random_request(request_id=i,
                                        num_tokens=prompt_length,
                                        sampling_params=sampling_params)
        requests.append((add_step, request))

    # Run steps, until last step is reached (provided through parameter)
    request_outputs = []
    for step in range(last_step):
        # Add requests for this step
        while requests and requests[0][0] == step:
            engine_core.add_request(requests.popleft()[1])
        
        # If initial step: check that tkv is zero and all requests are waiting
        if step == 0:
            assert scheduler.tkv == 0, "Step 0 (init), tkv"
            assert len(scheduler.waiting) != 0, "Step 0 (init), num waiting"
            assert len(scheduler.running) == 0, "Step 0 (init), num running"

        # Check step if it is in the provided list of steps to check
        if checked_steps and step == checked_steps[0]["step"]:
            step_ref = checked_steps.popleft()
            reqs_output_ids = [
                req_output.request_id for req_output in request_outputs
            ]
            
            assert scheduler.tkv == step_ref["tkv"], \
                f"Step {step}, tkv"
            assert len(scheduler.waiting) == step_ref["num_waiting"], \
                f"Step {step}, num waiting"
            assert len(scheduler.running) == step_ref["num_running"], \
                f"Step {step}, num running"
            assert reqs_output_ids == step_ref["request_outputs"], \
                f"Step {step}, request outputs"

        # Perform next step
        request_outputs = engine_core.step().outputs

    # Assert there is no more running or waiting request after last step
    assert len(scheduler.waiting) == 0, \
        "Last step done but still there are waiting requests"
    assert len(scheduler.running) == 0, \
        "Last step done but still there are running requests"

    # Tkv in scheduler is cleared one step later
    _ = engine_core.step().outputs
    assert scheduler.tkv == 0, "Tkv not cleared two steps after last step"

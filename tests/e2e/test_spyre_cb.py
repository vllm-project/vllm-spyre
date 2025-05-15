"""Verification of continuous batching

Run `python -m pytest tests/test_spyre_cb.py`.
"""

from collections import deque
import inspect
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
    assert request_outputs[0].request_id == "1"  # req 1 is decoding (prefill)

    assert len(engine_core.scheduler.waiting) == 0
    assert len(engine_core.scheduler.running) == 1

    # add another request
    engine.add_request("2", prompt2, sampling_params)

    assert len(engine_core.scheduler.waiting) == 1  # req 2 is waiting
    assert len(engine_core.scheduler.running) == 1

    request_outputs = engine.step()
    assert len(request_outputs) == 1  # still only 1 request
    assert request_outputs[0].request_id == "2"  # req 2 is decoding (prefill)

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
    # but since max_num_seqs=2
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
    assert request_outputs[0].request_id == "3"  # req 3 is decoding (prefill)

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
    
    # Temporary until 'cache_salt' parameter makes it to a release version
    # in vllm
    if 'cache_salt' in [x[0] for x in inspect.getmembers(EngineCoreRequest)]:
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
            cache_salt=None
        )
    else:
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
    prompts_lengths = [49, 41, 47]
    steps_add_reqs = [0, 0, 0]  # add all requests in the beginning

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
            "running": [],
            "request_outputs": []
        },
        {
            "step": 1,  # Prefill sequence 0
            "tkv": 64,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"]
        },
        {
            "step": 2,  # Prefill sequence 1
            "tkv": 64,  # Still 64 because this step is also a prefill
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"]
        },
        {
            "step": 3,  # Decode sequences 0 and 1
            "tkv": 65,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"]  # Two sequences are decoded
        },
        {
            # Sequence 0 finishes at step 66
            # (start step + 2 prefills + 64 decodes - 1) = 1 + 2 + 64 - 1 = 66
            "step": 66,
            "tkv": 128,
            "waiting": ["2"],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"]
        },
        {
            "step": 67,  # Prefill sequence 2
            "tkv": 128,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"]
        },
        {
            "step": 68,  # Decode sequences 1 and 2
            "tkv": 129,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2", "1"]
        },
        {
            # Sequence 1 finishes at step 69
            # (start step + 2 prefills + 66 decodes - 1) = 2 + 2 + 66 - 1 = 69
            "step": 69,
            "tkv": 130,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2", "1"],
            "finished_requests": ["1"]
        },
        {
            "step": 70,  # Decode sequence 2
            "tkv": 131,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"]
        },
        {
            # Sequence 2 finishes at step 73
            # (start step + 1 prefill + 6 decodes - 1) = 67 + 1 + 6 - 1 = 73
            "step": 73,
            "tkv": 134,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"]
        },
        {
            # Tkv should be cleared one step later
            "step": 74,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": []
        }
    ]

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps)


@pytest.mark.a
@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize(
    "seqs_max_tokens,prompts_lengths,steps_add_reqs,checked_steps",
    [
        get_params_test_blocks_borders_aligned_prompts(),
        # get_params_test_blocks_borders_misaligned_prompts(),  # TODO

        # TODO to test additionally at some point:
        # * test additional constraints from the scheduler (e.g prompt too long)
        # * test stripping repeated left padding
        # * test what happens when tkv comes to the end of 2048 block
        # * test metadata cleanup after last request finishes
        # * Corner cases:
        #     * two sequences finish at the same time
        #     * new prompts arrives when another finishes
    ])
def test_scheduler_cb_steps_tkv(model: str, backend: str,
                                monkeypatch: pytest.MonkeyPatch,
                                max_num_seqs: int, seqs_max_tokens: list[int],
                                prompts_lengths: list[int],
                                steps_add_reqs: list[int],
                                checked_steps: list[dict[str, Any]]):
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
    for step in range(checked_steps[-1]['step'] + 1):
        # Add requests for this step
        while requests and requests[0][0] == step:
            engine_core.add_request(requests.popleft()[1])

        # Check step if it is in the provided list of steps to check
        if checked_steps and step == checked_steps[0]["step"]:
            step_ref = checked_steps.popleft()

            waiting = [r.request_id for r in scheduler.waiting]
            running = [r.request_id for r in scheduler.running]
            out_reqs_ids = [r.request_id for r in request_outputs]
            out_reqs_finished = [
                r.request_id for r in request_outputs if r.finished
            ]

            assert scheduler.tkv == step_ref["tkv"], f"Step {step}, tkv"
            assert waiting == step_ref["waiting"], f"Step {step}, num waiting"
            assert running == step_ref["running"], f"Step {step}, num running"
            assert out_reqs_ids == step_ref["request_outputs"], \
                f"Step {step}, request outputs"

            ref_finished_reqs = step_ref.get("finished_requests", [])
            assert out_reqs_finished == ref_finished_reqs, \
                f"Step {step}, finished request output"

        # Perform next step
        request_outputs = engine_core.step().outputs

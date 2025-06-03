"""Verification of continuous batching

Run `python -m pytest tests/test_spyre_cb.py`.
"""

import copy
import inspect
from collections import deque
from typing import Any

import pytest
from spyre_util import generate_cb_spyre_vllm_output, get_spyre_model_list
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
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


def create_random_request(
        request_id: int, num_tokens: int,
        sampling_params: SamplingParams) -> EngineCoreRequest:

    # Temporary until 'cache_salt' parameter makes it to a release version
    # in vllm
    if 'cache_salt' in [x[0] for x in inspect.getmembers(EngineCoreRequest)]:
        return EngineCoreRequest(request_id=str(request_id),
                                 prompt_token_ids=[request_id] * num_tokens,
                                 mm_inputs=None,
                                 mm_hashes=None,
                                 mm_placeholders=None,
                                 sampling_params=sampling_params,
                                 eos_token_id=None,
                                 arrival_time=0,
                                 lora_request=None,
                                 cache_salt=None)
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
    """ Scenario where it happens that all the sequences get scheduled in a 
    fashion where they are aligned with the block boundaries (i.e. tkv multiple 
    of 64 at the time of prefilling)."""

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
            "request_outputs": ["1", "0"]
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


def get_params_test_blocks_borders_misaligned_prompts():
    """ Scenario where it happens that some sequence gets scheduled in a way 
    that it is misaligned with the block boundary (i.e. tkv is not a multiple 
    of 64 at the time of prefilling). """

    seqs_max_tokens = [57, 67, 9]
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
            "request_outputs": ["1", "0"]
        },
        {
            # Sequence 0 finishes at step 58
            # (start step + 2 prefills + 56 decodes - 1) = 1 + 2 + 56 - 1 = 58
            "step": 58,
            "tkv": 120,
            "waiting": ["2"],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"]
        },
        {
            "step": 59,  # Prefill sequence 2
            "tkv": 120,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"]
        },
        {
            "step": 60,  # Decode sequences 1 and 2
            "tkv": 121,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2", "1"]
        },
        {
            # Sequence 2 finishes at step 68
            # (start step + 1 prefill + 8 decodes - 1) = 59 + 1 + 8 - 1 = 67
            "step": 67,
            "tkv": 128,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["2", "1"],
            "finished_requests": ["2"]
        },
        {
            "step": 68,  # Decode sequences 1
            "tkv": 129,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"]
        },
        {
            # Sequence 1 finishes at step 69
            # (start step + 2 prefills + 66 decodes - 1) = 2 + 2 + 66 - 1 = 69
            "step": 69,
            "tkv": 130,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"]
        },
        {
            # Tkv should be cleared one step later
            "step": 70,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": []
        },
    ]

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps)


def augment_checked_steps(
        checked_steps: list[dict[str, Any]]) -> deque[dict[str, Any]]:
    # Augment checked_steps: add in-between normal decode steps
    checked_steps = deque(checked_steps)
    all_checked_steps = deque()
    prev_step = None
    for step in range(checked_steps[-1]["step"] + 1):
        if checked_steps and step == checked_steps[0]["step"]:
            prev_step = checked_steps.popleft()
            all_checked_steps.append(prev_step)
        elif prev_step is not None:
            assert prev_step["step"] == step - 1
            new_step = copy.deepcopy(prev_step)
            new_step["step"] = step
            new_step["tkv"] += 1
            all_checked_steps.append(new_step)
            prev_step = new_step
    return all_checked_steps


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize(
    "seqs_max_tokens,prompts_lengths,steps_add_reqs,checked_steps",
    [
        get_params_test_blocks_borders_aligned_prompts(),
        get_params_test_blocks_borders_misaligned_prompts(),

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
    Test the scheduler execution by comparing the scheduler attributes at each 
    step with the provided reference values in 'checked_steps'.
    
    The missing steps from 'checked_steps' are automatically generated as decode
    steps, based on the existing elements in the list. For that to work, all the
    prefill steps and the first decode step after them needs be added to 
    'checked_steps'
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

    # In-between steps are added as normal decode steps
    checked_steps = augment_checked_steps(checked_steps)

    # Run steps, until last step from 'checked_steps' is reached
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
        step_output = engine_core.step()
        # backward compatibility
        if isinstance(step_output, tuple):
            engine_core_output = step_output[0].get(0)
            request_outputs = (engine_core_output.outputs
                               if engine_core_output is not None else [])
        else:
            request_outputs = step_output.outputs

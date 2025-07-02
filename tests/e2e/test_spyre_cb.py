"""Verification of continuous batching

Run `python -m pytest tests/e2e/test_spyre_cb.py`.
"""

import copy
from collections import deque
from typing import Any

import pytest
from spyre_util import (compare_results, create_random_request,
                        generate_hf_output, generate_spyre_vllm_output,
                        get_chicken_soup_prompts, get_spyre_backend_list,
                        get_spyre_model_list)
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import ContinuousBatchingSpyreScheduler


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("max_num_seqs", [2, 4],
                         ids=lambda val: f"max_num_seqs({val})")
def test_cb_output(
    model: str,
    backend: str,
    max_num_seqs: int,
    monkeypatch: pytest.MonkeyPatch,
    runtime_xfail,
):
    """Test that the spyre worker correctly outputs
    continuous batches of requests by comparing to HF"""

    if max_num_seqs > 2 and backend == "sendnn":
        runtime_xfail("CB failures expected for batch size > 2")

    max_tokens = 20
    prompts = get_chicken_soup_prompts(4)

    vllm_sampling_params = SamplingParams(max_tokens=max_tokens,
                                          temperature=0,
                                          ignore_eos=True,
                                          logprobs=0)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=256,
        block_size=256,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        max_num_seqs=max_num_seqs,
        use_cb=True,
        monkeypatch=monkeypatch)

    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=max_tokens)

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=[],
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_cb_max_tokens(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that continuous batches of requests that
    are longer than the max_model_len are correctly rejected"""

    max_model_len = 256
    max_tokens = 20

    overflow_prompt = " ".join(["a"] * max_model_len)

    vllm_sampling_params = SamplingParams(max_tokens=max_tokens,
                                          temperature=0,
                                          ignore_eos=True,
                                          logprobs=0)

    with pytest.raises(ValueError, match="max model context length"):
        generate_spyre_vllm_output(model=model,
                                   prompts=overflow_prompt,
                                   max_model_len=max_model_len,
                                   block_size=max_model_len,
                                   sampling_params=vllm_sampling_params,
                                   tensor_parallel_size=1,
                                   backend=backend,
                                   max_num_seqs=2,
                                   use_cb=True,
                                   monkeypatch=monkeypatch)


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
            "tkv": 67,  # tkv is reset by 64 due to removing the padded block
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"]
        },
        {
            # Sequence 2 finishes at step 73
            # (start step + 1 prefill + 6 decodes - 1) = 67 + 1 + 6 - 1 = 73
            "step": 73,
            "tkv": 70,
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


def get_params_test_special_finish():
    """ 2-cases-in-1: (1) Two sequences finish at the same time and (2) a new
    request arrives when another finishes. """

    seqs_max_tokens = [30, 30, 10]
    prompts_lengths = [49, 30, 20]
    steps_add_reqs = [0, 0, 31]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": []
        },
        {
            # Prefill sequence 0
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"]
        },
        {
            # Prefill sequence 1
            "step": 2,
            "tkv": 64,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"]
        },
        {
            # Decode sequences 0 and 1
            "step": 3,
            "tkv": 65,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"]
        },
        {
            # Sequences 0 and 1 finish at step 31
            # (start step + 2 prefills + 29 decodes - 1) = 1 + 2 + 29 - 1 = 31
            "step": 31,
            "tkv": 93,
            "waiting": ["2"],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"]
        },
        {
            # Prefill sequence 2
            "step": 32,
            "tkv": 64,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
        },
        {
            # Decode sequence 2
            "step": 33,
            "tkv": 65,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
        },
        {
            # Sequences 2 finishes at step 41
            # (start step + 1 prefill + 29 decodes - 1) = 32 + 1 + 9 - 1 = 41
            "step": 41,
            "tkv": 73,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"]
        },
        {
            # Tkv should be cleared one step later
            "step": 42,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
        },
    ]

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps)


def get_params_test_scheduler_constraints_tkv():
    """ Scenario where the requested prompt is too long for current tkv value"""

    seqs_max_tokens = [57, 67]
    prompts_lengths = [49, 70]
    steps_add_reqs = [0, 0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": []
        },
        {
            # Prefill sequence 0
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"]
        },
        {
            # Decode sequence 0
            # Cannot prefill sequence 1, because of tkv constraint
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"]
        },
        {
            # Prefill sequence 1, tkv large enough
            "step": 8,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"]
        },
        {
            # Decode sequences 0 and 1
            "step": 9,
            "tkv": 71,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"]
        },
        {
            # Sequence 0 finishes at step 58
            # (start step + 2 prefills + 56 decodes - 1) = 1 + 2 + 56 - 1 = 58
            "step": 58,
            "tkv": 120,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"]
        },
        {
            # Decode sequence 1
            "step": 59,
            "tkv": 121,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
        },
        {
            # Sequence 1 finishes at step 74
            # (start step + 1 prefill + 66 decodes - 1) = 8 + 1 + 66 - 1 = 74
            "step": 74,
            "tkv": 136,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"]
        },
        {
            # Tkv should be cleared one step later
            "step": 75,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": []
        },
    ]

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps)


def get_params_test_scheduler_constraints_max_prompt_len():
    """ Scenario where the request goes beyond max_model_len """

    seqs_max_tokens = [67, 57, 80]
    prompts_lengths = [70, 49, 41]
    steps_add_reqs = [0, 0, 0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
            "running": [],
            "request_outputs": []
        },
        {
            # Prefill sequence 0
            "step": 1,
            "tkv": 128,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"]
        },
        {
            # Prefill sequence 1
            "step": 2,
            "tkv": 128,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"]
        },
        {
            # Decode sequences 0 and 1
            "step": 3,
            "tkv": 129,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"]
        },
        {
            # Sequence 1 finishes at step 58
            # (start step + 1 prefills + 56 decodes - 1) = 2 + 1 + 56 - 1 = 58
            "step": 58,
            "tkv": 184,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"]
        },
        {
            # Decode sequence 0
            # Cannot prefill sequence 2: 185 + 80 = 265 > 256
            "step": 59,
            "tkv": 185,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["0"],
        },
        {
            # Sequence 0 finishes at step 68
            # (start step + 2 prefills + 66 decodes - 1) = 1 + 2 + 66 - 1 = 68
            "step": 68,
            "tkv": 194,
            "waiting": ["2"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"]
        },
        {
            # Prefill sequence 2
            "step": 69,
            "tkv": 64,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
        },
        {
            # Decode sequence 2
            "step": 70,
            "tkv": 65,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
        },
        {
            # Sequence 2 finishes at step 148
            # (start step + 1 prefill + 79 decodes - 1) = 69 + 1 + 79 - 1 = 148
            "step": 148,
            "tkv": 143,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"]
        },
        {
            # Tkv should be cleared one step later
            "step": 149,
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
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize(
    "seqs_max_tokens,prompts_lengths,steps_add_reqs,checked_steps", [
        get_params_test_blocks_borders_aligned_prompts(),
        get_params_test_blocks_borders_misaligned_prompts(),
        get_params_test_special_finish(),
        get_params_test_scheduler_constraints_tkv(),
        get_params_test_scheduler_constraints_max_prompt_len(),
    ])
def test_scheduler_cb_steps_tkv(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    steps_add_reqs: list[int],
    checked_steps: list[dict[str, Any]],
):
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

    max_model_len = 256

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
                             max_model_len=max_model_len,
                             block_size=max_model_len,
                             max_num_seqs=2)
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

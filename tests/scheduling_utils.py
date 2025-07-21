import copy
from collections import deque
from typing import Any

import pytest
from spyre_util import create_random_request
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import ContinuousBatchingSpyreScheduler


def augment_checked_steps(checked_steps: list[dict[str, Any]],
                          heterog_cb: bool) -> deque[dict[str, Any]]:
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
            if heterog_cb:
                n_tkvs = sum(k.startswith('tkv') for k in new_step)
                for i in range(n_tkvs):
                    new_step["tkv_" + str(i)] += 1
            else:
                new_step["tkv"] += 1
            all_checked_steps.append(new_step)
            prev_step = new_step
    return all_checked_steps


def check_scheduler_inference_steps(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    steps_add_reqs: list[int],
    checked_steps: list[dict[str, Any]],
    max_num_seqs: int,
    available_blocks: int,
    use_cb: bool = True,
    heterog_cb: bool = False,
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
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    if available_blocks > 0:
        monkeypatch.setenv("VLLM_SPYRE_N_BLOCKS", str(available_blocks))
    if use_cb:
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
    if heterog_cb:
        monkeypatch.setenv("VLLM_SPYRE_HETEROGEN_TKV", "1")
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
    checked_steps = augment_checked_steps(checked_steps, heterog_cb=heterog_cb)

    # Run steps, until last step from 'checked_steps' is reached
    request_outputs = []
    requested_blocks, reserved_blocks = {}, {}
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

            if heterog_cb:
                for i, tkv in enumerate(scheduler.tkvs):
                    assert tkv == step_ref["tkv_" +
                                           str(i)], f"Step {step}, tkv_{i}"
            else:
                assert len(set(scheduler.tkvs)) == 1  # assert homogeneous tkv
                assert scheduler.tkvs[0] == step_ref[
                    "tkv"], f"Step {step}, tkv"
            assert waiting == step_ref["waiting"], f"Step {step}, num waiting"
            assert running == step_ref["running"], f"Step {step}, num running"
            assert out_reqs_ids == step_ref["request_outputs"], \
                f"Step {step}, request outputs"

            ref_finished_reqs = step_ref.get("finished_requests", [])
            assert out_reqs_finished == ref_finished_reqs, \
                f"Step {step}, finished request output"

            # checking the scheduler handling of free and reserved blocks
            n_blocks = (engine_core.model_executor.driver_worker.worker.
                        model_runner.n_blocks)
            n_reserved_blocks = n_blocks - scheduler.n_free_blocks
            req_ids2blocks = (engine_core.model_executor.driver_worker.worker.
                              model_runner.req_ids2blocks)
            req_ids2reserved_blocks = (
                engine_core.model_executor.driver_worker.worker.model_runner.
                req_ids2reserved_blocks)
            n_used_blocks = sum(
                [len(blocks) for blocks in req_ids2blocks.values()])

            if step > 0:
                assert n_reserved_blocks == step_ref[
                    "n_reserved_blocks"], f"Step {step}, n_reserved_blocks"
                assert n_used_blocks == step_ref[
                    "n_used_blocks"], f"Step {step}, n_used_blocks"

            assert len(req_ids2blocks) == len(req_ids2reserved_blocks)
            for req_id in req_ids2blocks:
                # current number of used blocks should be less than reserved
                assert len(
                    req_ids2blocks[req_id]) <= req_ids2reserved_blocks[req_id]
                # update requested/reserved blocks to check in last step
                # Note: overwrite and not max because of reduce_left_padding()
                requested_blocks[req_id] = len(req_ids2blocks[req_id])
                reserved_blocks[req_id] = req_ids2reserved_blocks[req_id]

        # last step: check that sequences used all their reserved blocks
        # Note: no early stopping, all sequences produce max_num_tokens
        if len(checked_steps) == 0:
            for req_id in requested_blocks:
                assert requested_blocks[req_id] == reserved_blocks[req_id]

        # Perform next step
        step_output = engine_core.step()
        # backward compatibility
        if isinstance(step_output, tuple):
            engine_core_output = step_output[0].get(0)
            request_outputs = (engine_core_output.outputs
                               if engine_core_output is not None else [])
        else:
            request_outputs = step_output.outputs

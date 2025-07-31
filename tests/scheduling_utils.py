import copy
from collections import defaultdict, deque
from typing import Any

import pytest
from spyre_util import create_random_request
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import ContinuousBatchingSpyreScheduler


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


def check_scheduler_inference_steps(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    steps_add_reqs: list[int],
    checked_steps: list[dict[str, Any]],
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
    use_cb: bool = True,
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
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    if use_cb:
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")

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

    collected_outputs = defaultdict(lambda: {"token_ids": [], "logprobs": []})
    generated_prompts = []

    # Create random requests of specified lengths and max_tokens
    # Need to do before setting up the vLLM engine, otherwise test random seed
    # will be overridden
    sorted_reqs_params = zip(steps_add_reqs, seqs_max_tokens, prompts_lengths)
    requests: deque[tuple[int, EngineCoreRequest]] = deque()
    for i, (add_step, max_tokens,
            prompt_length) in enumerate(sorted_reqs_params):
        # ignoring eos because we want to force the decoding to finish
        # after max_tokens exactly
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         temperature=0.0,
                                         logprobs=0,
                                         ignore_eos=True)
        request = create_random_request(request_id=i,
                                        num_tokens=prompt_length,
                                        sampling_params=sampling_params,
                                        model=model)
        requests.append((add_step, request))
        generated_prompts.append(request.prompt_token_ids)

    # Setup the engine
    engine_args = EngineArgs(model=model,
                             tokenizer=model,
                             max_model_len=max_model_len,
                             block_size=max_model_len,
                             max_num_seqs=max_num_seqs,
                             num_gpu_blocks_override=available_blocks
                             if available_blocks > 0 else None)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(vllm_config=vllm_config,
                             executor_class=executor_class,
                             log_stats=False)
    scheduler: ContinuousBatchingSpyreScheduler = engine_core.scheduler

    # In-between steps are added as normal decode steps
    checked_steps = augment_checked_steps(checked_steps)

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

            assert (scheduler.tkv == step_ref["tkv"]
                    ), f"Step {step}, tkv: {scheduler.tkv}"
            assert waiting == step_ref[
                "waiting"], f"Step {step}, waiting: {waiting}"
            assert running == step_ref[
                "running"], f"Step {step}, running: {running}"
            assert (out_reqs_ids == step_ref["request_outputs"]
                    ), f"Step {step}, request outputs: {out_reqs_ids}"

            ref_finished_reqs = step_ref.get("finished_requests", [])
            assert (
                out_reqs_finished == ref_finished_reqs
            ), f"Step {step}, finished request output: {out_reqs_finished}"

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
                assert (
                    n_reserved_blocks == step_ref["n_reserved_blocks"]
                ), f"Step {step}, n_reserved_blocks: {n_reserved_blocks}"
                assert (n_used_blocks == step_ref["n_used_blocks"]
                        ), f"Step {step}, n_used_blocks: {n_used_blocks}"

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
        engine_core_output = step_output[0].get(0)
        request_outputs = (engine_core_output.outputs
                           if engine_core_output is not None else [])

        for output in request_outputs:
            new_token_ids = output.new_token_ids
            new_logprobs = output.new_logprobs.logprobs
            assert len(new_token_ids) == 1 and len(new_logprobs) == 1

            collected_outputs[output.request_id]["token_ids"].append(
                new_token_ids[0])
            collected_outputs[output.request_id]["logprobs"].append(
                new_logprobs[0][0])

    output_keys = sorted(int(k) for k in collected_outputs)
    assert output_keys[0] == 0 and output_keys[-1] == len(output_keys) - 1

    # convert dict of dicts to ordered list and make values immutable
    collected_outputs_new = []
    for k in output_keys:
        output = collected_outputs[str(k)]
        for k, list_values in output.items():
            if isinstance(list_values, list):
                output[k] = tuple(list_values)
        collected_outputs_new.append(output)

    return collected_outputs_new, generated_prompts

import copy
import os
from collections import defaultdict, deque
from typing import Any

import pytest
from llm_cache import get_cached_engine
from output_util import (ISCLOSE_ABS_TOL, ISCLOSE_ABS_TOL_QUANTIZATION,
                         compare_results, generate_hf_output)
from spyre_util import ModelInfo, create_random_request
from vllm import SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore

from vllm_spyre.v1.core.scheduler import ContinuousBatchingSpyreScheduler

DISABLE_ASSERTS = False  # used for debugging


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


def generate_prompts(
    model: ModelInfo,
    steps_add_reqs: list[int],
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
):
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
        # NOTE: It is going to be decoded later
        generated_prompts.append(request.prompt_token_ids)

    return generated_prompts, requests


def check_scheduler_inference_steps(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    steps_add_reqs: list[int],
    checked_steps: list[dict[str, Any]],
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
    max_batch_tkv_limit: int = -1,
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

    collected_outputs = defaultdict(lambda: {
        "token_ids": [],
        "logprobs": [],
        "text": "",
        "tokens": []
    })

    prompts, requests = generate_prompts(model, steps_add_reqs,
                                         seqs_max_tokens, prompts_lengths)

    hf_results = generate_hf_output(
        model=model,
        prompts=prompts,
        max_new_tokens=seqs_max_tokens,
        ignore_eos=True,
    )

    abs_tol = ISCLOSE_ABS_TOL_QUANTIZATION if model.is_quantized \
         else ISCLOSE_ABS_TOL
    # inject expectation.
    # json is fine to transfer between vllm subprocesses using pickle
    for idx, (req, hf) in enumerate(zip(requests, hf_results)):
        req[1].sampling_params.extra_args = {
            "golden_token_injector": {
                "expected_token_ids": hf['token_ids'],
                "expected_logprobs": hf['logprobs'],
                "error_threshold": abs_tol,
                "label": f"#{idx}"
            }
        }

    # Setup the engine
    engine_core: EngineCore = get_cached_engine(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        available_blocks=available_blocks,
        backend=backend,
        monkeypatch=monkeypatch)
    scheduler: ContinuousBatchingSpyreScheduler = engine_core.scheduler

    tokenizer = get_tokenizer(model.name, revision=model.revision)
    # clear the cache of function scheduler.check_batch_tkv_limit()
    scheduler._cache_check_batch_tkv_limit.clear()

    # Override the TKV limit in the scheduler if needed
    if max_batch_tkv_limit >= 0:
        scheduler.max_batch_tkv_limit = max_batch_tkv_limit
    else:
        # This default value is set by platform.py
        scheduler.max_batch_tkv_limit = int(
            os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"))

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

            assert DISABLE_ASSERTS or (scheduler.tkv == step_ref["tkv"]
                                       ), f"Step {step}, tkv: {scheduler.tkv}"
            assert (DISABLE_ASSERTS or waiting
                    == step_ref["waiting"]), f"Step {step}, waiting: {waiting}"
            assert (DISABLE_ASSERTS or running
                    == step_ref["running"]), f"Step {step}, running: {running}"
            assert DISABLE_ASSERTS or (
                out_reqs_ids == step_ref["request_outputs"]
            ), f"Step {step}, request outputs: {out_reqs_ids}"

            ref_finished_reqs = step_ref.get("finished_requests", [])
            assert DISABLE_ASSERTS or (
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
                assert DISABLE_ASSERTS or (
                    n_reserved_blocks == step_ref["n_reserved_blocks"]
                ), f"Step {step}, n_reserved_blocks: {n_reserved_blocks}"
                assert DISABLE_ASSERTS or (
                    n_used_blocks == step_ref["n_used_blocks"]
                ), f"Step {step}, n_used_blocks: {n_used_blocks}"

            assert DISABLE_ASSERTS or len(req_ids2blocks) == len(
                req_ids2reserved_blocks)
            for req_id in req_ids2blocks:
                # current number of used blocks should be less than reserved
                assert (DISABLE_ASSERTS or len(req_ids2blocks[req_id])
                        <= req_ids2reserved_blocks[req_id])
                # update requested/reserved blocks to check in last step
                # Note: overwrite and not max
                # because of reduce_left_padding()
                requested_blocks[req_id] = len(req_ids2blocks[req_id])
                reserved_blocks[req_id] = req_ids2reserved_blocks[req_id]

        # last step: check that sequences used all their reserved blocks
        # Note: no early stopping, all sequences produce max_num_tokens
        if len(checked_steps) == 0:
            for req_id in requested_blocks:
                assert (DISABLE_ASSERTS
                        or requested_blocks[req_id] == reserved_blocks[req_id])

        # Perform next step
        step_output = engine_core.step()
        engine_core_output = step_output[0].get(0)
        request_outputs = (engine_core_output.outputs
                           if engine_core_output is not None else [])

        for output in request_outputs:
            new_token_ids = output.new_token_ids
            new_logprobs = output.new_logprobs.logprobs
            assert DISABLE_ASSERTS or len(new_token_ids) == 1 and len(
                new_logprobs) == 1

            collected_outputs[output.request_id]["token_ids"].append(
                new_token_ids[0])
            collected_outputs[output.request_id]["logprobs"].append(
                new_logprobs[0][0])
            collected_outputs[output.request_id]["tokens"].append(
                tokenizer.decode(new_token_ids[0]))

    for k in collected_outputs:
        collected_outputs[k]['text'] = tokenizer.decode(
            collected_outputs[k]['token_ids'])
    output_keys = sorted(int(k) for k in collected_outputs)
    assert (DISABLE_ASSERTS
            or output_keys[0] == 0 and output_keys[-1] == len(output_keys) - 1)

    # convert dict of dicts to ordered list and make values immutable
    vllm_results = []
    for k in output_keys:
        output = collected_outputs[str(k)]
        for k, list_values in output.items():
            if isinstance(list_values, list):
                output[k] = tuple(list_values)
        vllm_results.append(output)

    compare_results(model=model,
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results,
                    prompts=prompts)

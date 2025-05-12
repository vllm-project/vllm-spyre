import os
import time
import pytest
from typing import Optional
from vllm.engine.arg_utils import EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor


def create_random_request(request_id: int, num_tokens: int, sampling_params: SamplingParams) -> EngineCoreRequest:
    return EngineCoreRequest(
        request_id=str(request_id),
        prompt_token_ids= [request_id] * num_tokens,
        mm_inputs=None,
        mm_hashes=None,
        mm_placeholders=None,
        sampling_params=sampling_params,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
    )


@pytest.mark.parametrize("model", ["/models/llama-194m"])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("seqs_max_tokens", [[65, 67, 7]])
@pytest.mark.parametrize("prompts_lengths", [[49, 41, 7]])
@pytest.mark.parametrize("checked_steps", [[1]])
@pytest.mark.parametrize("gt_num_waiting", [[2]])
@pytest.mark.parametrize("gt_num_running", [[1]])
@pytest.mark.parametrize("gt_tkv", [[64]])
@pytest.mark.parametrize("gt_last_step", [73])
def test_scheduler_queues_lengths_and_tkv(
    monkeypatch: pytest.MonkeyPatch,
    model: str, 
    max_num_seqs: int,
    seqs_max_tokens: list[int], 
    prompts_lengths: list[int],
    checked_steps: list[int], 
    gt_num_waiting: list[int],
    gt_num_running: list[int],
    gt_tkv: list[int],
    gt_last_step: int,
    add_requests_at_steps: Optional[list[int]] = None,
):
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "eager")
    monkeypatch.setenv("VLLM_USE_V1", "1")
    
    # ------ 
    # Input parameters sanity check, not actual testing
    if len(prompts_lengths) != len(seqs_max_tokens):
        raise ValueError("Number of prompts should be consistent with number of max tokens.")

    num_checked_steps = len(checked_steps)
    if len(gt_num_waiting) != num_checked_steps or len(gt_num_running) != num_checked_steps or len(gt_tkv) != num_checked_steps:
        raise ValueError("Number of elements in the ground truth list of values is inconsistent.")
    
    if not (checked_steps == sorted(checked_steps) and len(checked_steps) == len(set(checked_steps))):
        raise ValueError("List of checked steps needs to be strictly increasing.")

    if add_requests_at_steps is not None and add_requests_at_steps[0] != 0:
        raise ValueError("First requests should be necessarily added at step 0.")
    # ------

    # Engine initialization
    engine_args = EngineArgs(model=model, tokenizer=model, max_model_len=2048, block_size=2048, max_num_seqs=max_num_seqs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(vllm_config=vllm_config, executor_class=executor_class, log_stats=False)
    scheduler = engine_core.scheduler

    # Create requests
    requests = []
    for i, (max_tokens_i, prompt_length_i) in enumerate(zip(seqs_max_tokens, prompts_lengths)):
        sampling_params = SamplingParams(max_tokens=max_tokens_i, temperature=0.0, ignore_eos=True)
        request_i = create_random_request(request_id=i, num_tokens=prompt_length_i, sampling_params=sampling_params)
        requests.append(request_i)
    
    # Add requests to the vllm engine. 
    # If requests_added_steps is None, add all for step zero
    next_added_request_id = len(seqs_max_tokens)
    if add_requests_at_steps is None or all(req_step == 0 for req_step in add_requests_at_steps):
        for req in requests:
            engine_core.add_request(req)
    else:
        engine_core.add_request(req[0])
        next_added_request_id = 1
    
    # Initial state (step 0): ensure tkv is zero, and all scheduled requests are waiting
    assert scheduler.tkv == 0, "Step 0 (initialization step), tkv"
    assert len(scheduler.waiting) == next_added_request_id, "Step 0 (initialization step), num waiting"
    assert len(scheduler.running) == 0, "Step 0 (initialization step), num running"
    
    i = 1 if checked_steps[0] == 0 else 0  # We don't check step 0 again. Already done.
    next_step_to_check = checked_steps[i]
    
    for step in range(1, gt_last_step + 1, 1):
        # Do a step in the engine
        engine_core.step()
        if step == next_step_to_check:
            assert scheduler.tkv == gt_tkv[i], f"Step {step}, tkv"
            assert len(scheduler.waiting) == gt_num_waiting[i], f"Step {step}, num waiting"
            assert len(scheduler.running) == gt_num_running[i], f"Step {step}, num running"
        i += 1
    
    # Check last step and last + 1 step
    assert scheduler.tkv != 0, f"Step {step} (last step), tkv"
    assert len(scheduler.waiting) == 0, f"Step {step} (last step), num waiting"
    assert len(scheduler.running) == 0, f"Step {step} (last step), num running"
    engine_core.step()
    assert scheduler.tkv == 0, f"Step {step + 1} (step after last step), tkv"
    assert len(scheduler.waiting) == 0, f"Step {step + 1} (step after last step), num waiting"
    assert len(scheduler.running) == 0, f"Step {step + 1} (step after last step), num running"
    
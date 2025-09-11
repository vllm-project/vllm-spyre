"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

import pytest
from output_util import check_output_against_hf, generate_spyre_vllm_output
from spyre_util import (DecodeWarmupShapes, ModelInfo, create_random_request,
                        get_chicken_soup_prompts, patch_environment,
                        skip_unsupported_tp_size)
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import StaticBatchingSpyreScheduler


@pytest.mark.full_model
def test_output(model: ModelInfo, tp_size: int, backend: str, cb: int,
                max_num_seqs: int, max_model_len: int,
                warmup_shapes: DecodeWarmupShapes,
                monkeypatch: pytest.MonkeyPatch, use_llm_cache) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.
    
    Configuration for CB - parameters are combinatorial:
        * max_num_seqs: 4
        * tensor parallelism: 1, 2, 4, 8
        * number of prompts: 4 (Chicken soup prompts)
        * max tokens: 20 (same for all the prompts)
    '''

    skip_unsupported_tp_size(tp_size, backend)

    prompts = get_chicken_soup_prompts(4)

    kwargs = ({
        "max_num_seqs": max_num_seqs,
        "use_cb": True,
    } if cb == 1 else {
        "warmup_shapes": warmup_shapes,
    })

    max_new_tokens = warmup_shapes[0][1]

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend,
        monkeypatch=monkeypatch,
        max_model_len=max_model_len,
        **kwargs)
    check_output_against_hf(model, backend, max_new_tokens, vllm_results,
                            prompts)


@pytest.mark.parametrize("backend", [
    pytest.param(
        "sendnn_decoder", marks=pytest.mark.spyre, id="sendnn_decoder")
])
def test_output_sendnn_decoder(model: ModelInfo,
                               warmup_shapes: DecodeWarmupShapes, backend: str,
                               monkeypatch: pytest.MonkeyPatch,
                               use_llm_cache) -> None:
    '''
    Tests the deprecated sendnn_decoder backend, which should fall-back to
    sendnn
    '''

    max_new_tokens = warmup_shapes[0][1]
    prompts = get_chicken_soup_prompts(1)

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=warmup_shapes,
        max_model_len=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch)

    check_output_against_hf(model, backend, max_new_tokens, vllm_results,
                            prompts)


def test_batch_handling(model: ModelInfo, backend: str, cb: int, warmup_shapes,
                        max_num_seqs: int, max_model_len: int,
                        monkeypatch: pytest.MonkeyPatch, use_llm_cache):
    """Test that the spyre worker correctly handles
    continuous batches of requests that
    finish after different numbers of forward passes

    Configuration for CB - parameters are combinatorial:
        * max_num_seqs: 2
        * number of prompts: 4 (Chicken soup prompts)
        * max tokens: [5, 20, 10, 5]
    """

    prompts = get_chicken_soup_prompts(4)

    max_new_tokens = [5, 20, 10, 5]

    vllm_sampling_params = [
        SamplingParams(max_tokens=max_new_tokens[i],
                       min_tokens=max_new_tokens[i],
                       temperature=0,
                       ignore_eos=True,
                       logprobs=0) for i in range(len(max_new_tokens))
    ]

    kwargs = {
        "max_num_seqs": max_num_seqs,
        "use_cb": True
    } if cb == 1 else {
        "warmup_shapes": warmup_shapes
    }

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=256,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        **kwargs)

    check_output_against_hf(model, backend, max_new_tokens, vllm_results,
                            prompts)


def test_full_batch_scheduling(model: ModelInfo, backend: str, monkeypatch):
    """Test that we can schedule a full batch of prompts."""

    # We need to ensure here that the max number of tokens in a full batch
    # is greater than the value set for `--max-num-batched-tokens`.
    # This defaults to 2k in many cases for vllm.v1, which will cause problems
    # when trying to schedule a static batch with more than 2k tokens.
    # The plugin _should_ override this in config for the engine so that the
    # scheduler can properly schedule a full batch.

    # Here we set `--max-num-batched-tokens` to 64, and try to schedule a batch
    # of 4 x 64-token prompts
    max_batched_tokens = 64
    batch_size = 4

    # set batching config
    monkeypatch.setenv("VLLM_SPYRE_WARMUP_BATCH_SIZES", f"{batch_size}")
    monkeypatch.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS",
                       f"{max_batched_tokens}")
    monkeypatch.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS", "20")

    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

    # Setup the engine
    engine_args = EngineArgs(model=model.name,
                             tokenizer=model.name,
                             max_num_batched_tokens=max_batched_tokens,
                             max_num_seqs=batch_size,
                             revision=model.revision)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(vllm_config=vllm_config,
                             executor_class=executor_class,
                             log_stats=False)
    scheduler: StaticBatchingSpyreScheduler = engine_core.scheduler

    vllm_sampling_params = SamplingParams(max_tokens=20,
                                          temperature=0,
                                          logprobs=0)
    for i in range(batch_size):
        engine_core.add_request(
            create_random_request(
                request_id=i,
                num_tokens=max_batched_tokens,
                sampling_params=vllm_sampling_params,
                model=model.name,
            ))
    schedule = scheduler.schedule()

    assert len(schedule.scheduled_new_reqs) == batch_size


def test_max_model_len_override(model: ModelInfo, backend, warmup_shapes, cb,
                                monkeypatch):
    """Test that makes sure that --max-model-len
    doesn't affect SB, instead it is picked up from
    warmup shapes"""

    max_model_len = 64
    kwargs = ({
        "use_cb": True,
        "warmup_shapes": None
    } if cb == 1 else {
        "use_cb": False,
        "warmup_shapes": warmup_shapes,
    })

    patch_environment(**kwargs, backend=backend, monkeypatch=monkeypatch)
    vllm_config = EngineArgs(
        model=model.name, max_model_len=max_model_len).create_engine_config()
    model_config = vllm_config.model_config

    if not cb:
        assert model_config.max_model_len == max([
            prompt_length + new_tokens
            for prompt_length, new_tokens, _ in warmup_shapes
        ])
    else:
        assert model_config.max_model_len == max_model_len

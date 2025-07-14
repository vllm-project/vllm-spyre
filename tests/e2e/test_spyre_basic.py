"""Verification of vLLM output by comparing with HF

Run `python -m pytest tests/e2e/test_spyre_basic.py`.
"""

import pytest
from spyre_util import (compare_results, create_random_request,
                        generate_hf_output, generate_spyre_vllm_output,
                        get_chicken_soup_prompts, get_spyre_backend_list,
                        get_spyre_model_list, skip_unsupported_tp_size)
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import StaticBatchingSpyreScheduler


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "warmup_shape", [(64, 20, 4)])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("tp_size", [
    pytest.param(1),
    pytest.param(2, marks=pytest.mark.multi),
    pytest.param(4, marks=pytest.mark.multi),
    pytest.param(8, marks=pytest.mark.multi),
],
                         ids=lambda val: f"TP({val})")
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_output(
    model: str,
    warmup_shape: tuple[int, int, int],
    tp_size: int,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    '''
    The warmup is based on a single shape. After the warmup,
    one request with the provided prompts is input to vLLM.
    The same prompts are also input to HF. The generated output
    including text, token ids, and logprobs, is verified to be
    identical for vLLM and HF.

    If errors occur, these can be analyzed/debugged by setting
    'DISABLE_ASSERTS = True' in spyre_util.py and by rerunning the
    test using 'pytest --capture=no tests/spyre/test_spyre_basic.py'
    After debugging, DISABLE_ASSERTS should be reset to 'False'.
    '''

    skip_unsupported_tp_size(tp_size, backend)

    prompts = get_chicken_soup_prompts(4)

    max_new_tokens = warmup_shape[1]

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=[warmup_shape],
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend,
        monkeypatch=monkeypatch)

    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=max_new_tokens)

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=[warmup_shape],
                    tensor_parallel_size=tp_size,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "warmup_shape", [(64, 20, 4)])  # (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize(
    "backend",
    pytest.param("sendnn_decoder",
                 marks=pytest.mark.spyre,
                 id="sendnn_decoder"))
def test_output_sendnn_decoder(
    model: str,
    warmup_shape: tuple[int, int, int],
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    '''
    Tests the deprecated sendnn_decoder backend, which should fall-back to
    sendnn
    '''

    max_new_tokens = warmup_shape[1]
    prompts = get_chicken_soup_prompts(1)

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True)

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        warmup_shapes=[warmup_shape],
        max_model_len=2048,
        block_size=2048,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch)

    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=max_new_tokens)

    compare_results(model=model,
                    prompts=prompts,
                    warmup_shapes=[warmup_shape],
                    tensor_parallel_size=1,
                    backend=backend,
                    vllm_results=vllm_results,
                    hf_results=hf_results)


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("cb",
                         [pytest.param(1, marks=pytest.mark.cb, id="cb"), 0])
def test_batch_handling(model: str, backend: str, cb: int,
                        monkeypatch: pytest.MonkeyPatch):
    """Test that the spyre worker correctly handles
    continuous batches of requests that
    finish after different numbers of forward passes"""

    prompts = get_chicken_soup_prompts(4)

    sampling_params1 = SamplingParams(max_tokens=5,
                                      min_tokens=5,
                                      temperature=0,
                                      ignore_eos=True,
                                      logprobs=0)
    sampling_params2 = SamplingParams(max_tokens=20,
                                      min_tokens=20,
                                      temperature=0,
                                      ignore_eos=True,
                                      logprobs=0)
    sampling_params3 = SamplingParams(max_tokens=10,
                                      min_tokens=10,
                                      temperature=0,
                                      ignore_eos=True,
                                      logprobs=0)
    sampling_params4 = SamplingParams(max_tokens=5,
                                      min_tokens=5,
                                      temperature=0,
                                      ignore_eos=True,
                                      logprobs=0)

    vllm_sampling_params = [
        sampling_params1,
        sampling_params2,
        sampling_params3,
        sampling_params4,
    ]

    kwargs = {
        "max_num_seqs": 2,
        "use_cb": True
    } if cb == 1 else {
        "warmup_shapes": ((64, 20, 4), )
    }

    vllm_results = generate_spyre_vllm_output(
        model=model,
        prompts=prompts,
        max_model_len=256,
        block_size=256,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=1,
        backend=backend,
        monkeypatch=monkeypatch,
        **kwargs)
    hf_results = generate_hf_output(model=model,
                                    prompts=prompts,
                                    max_new_tokens=[5, 20, 10, 5])

    compare_results(
        model=model,
        prompts=prompts,
        warmup_shapes=[],
        tensor_parallel_size=1,
        backend=backend,
        vllm_results=vllm_results,
        hf_results=hf_results,
    )


@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_full_batch_scheduling(model: str, backend: str, monkeypatch):
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

    # So we can access the engine and scheduler in this process
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)

    # Setup the engine
    engine_args = EngineArgs(model=model,
                             tokenizer=model,
                             max_num_batched_tokens=max_batched_tokens,
                             max_num_seqs=batch_size)
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
            create_random_request(request_id=i,
                                  num_tokens=max_batched_tokens,
                                  sampling_params=vllm_sampling_params))
    schedule = scheduler.schedule()

    assert len(schedule.scheduled_new_reqs) == batch_size

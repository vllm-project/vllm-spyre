"""Tests the chunked prefill logic in the model runner

These tests do not involve the scheduler or engine, they isolate the testing to
the padding and chunking logic in the model runner itself. Theyare designed to
run on cpu only.

These tests all assume a chunk size of 128 to keep the test runtime overhead
low.
"""
from dataclasses import dataclass, field

import pytest
from llm_cache import get_cached_engine
from spyre_util import ModelInfo
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
from vllm.v1.engine.core import EngineCore
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, SamplingParams

from vllm_spyre.v1.worker.spyre_model_runner import SpyreModelRunner


########## Assuming that we have:
@dataclass
class ChunkedPrefillModelRunnerOutput(ModelRunnerOutput):
    # Current TKV for each request
    # request_tkvs: dict[str, int] = field(default_factory=dict)
    # JK! We will just return tkv = max(request_tkvs)

    # Number of computed tokens for each request (may be smaller than the number
    # of scheduled tokens)
    computed_tokens: dict[str, int] = field(default_factory=dict)
    # (not implemented yet- maybe will be?)

    # Free KV cache blocks left to use
    n_free_blocks: int = 0
    # Already part of CB model runner output


########## End assumptions


def default_test_params(test_func):
    """Sets params for the tests in this file"""
    test_func = pytest.mark.parametrize(
        "max_model_len", [512],
        ids=lambda val: f"max_model_len({val})")(test_func)
    test_func = pytest.mark.parametrize(
        "max_num_seqs", [2], ids=lambda val: f"max_num_seqs({val})")(test_func)

    test_func = pytest.mark.parametrize(
        "max_num_batched_tokens", [128],
        ids=lambda val: f"max_num_batched_tokens({val})")(test_func)

    return test_func


def get_cpu_model_runner(model: ModelInfo, max_model_len: int,
                         max_num_seqs: int, max_num_batched_tokens: int,
                         monkeypatch: pytest.MonkeyPatch) -> SpyreModelRunner:
    # TODO: Need to add chunked prefill mode + params to get_cached_engine
    engine_core: EngineCore = get_cached_engine(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        available_blocks=None,
        backend="eager",
        monkeypatch=monkeypatch)

    # NB: Only works because this engine is run with no multiprocessing and TP=1
    runner: SpyreModelRunner = \
        engine_core.model_executor.driver_worker.worker.model_runner

    # Clean things up, this isn't a fixture that cleans up after previous tests
    if runner.requests:
        for request_id in list(runner.requests):
            # aborting one-by-one should be less error prone for the runner
            abort_sched = make_scheduler_output(scheduled_new_reqs=[],
                                                finished_req_ids={request_id})
            runner.execute_model(abort_sched)

    return runner


def make_cached_request_data(req_id_to_computed_tokens) -> CachedRequestData:
    cached_request_data = CachedRequestData.make_empty()
    cached_request_data.req_ids = list(req_id_to_computed_tokens.keys())
    cached_request_data.num_computed_tokens = list(
        req_id_to_computed_tokens.values())
    return cached_request_data


def make_scheduler_output(
        scheduled_new_reqs: list[NewRequestData],
        scheduled_cached_reqs: CachedRequestData = None,
        num_scheduled_tokens: dict[str, int] = None,
        finished_req_ids: set[str] = None) -> SchedulerOutput:
    total_tokens = sum(
        num_scheduled_tokens.values()) if num_scheduled_tokens else 0
    if scheduled_cached_reqs is None:
        scheduled_cached_reqs = CachedRequestData.make_empty()
    if num_scheduled_tokens is None:
        num_scheduled_tokens = {}
    if finished_req_ids is None:
        finished_req_ids = set()

    return SchedulerOutput(scheduled_new_reqs=scheduled_new_reqs,
                           scheduled_cached_reqs=scheduled_cached_reqs,
                           num_scheduled_tokens=num_scheduled_tokens,
                           total_num_scheduled_tokens=total_tokens,
                           scheduled_spec_decode_tokens={},
                           scheduled_encoder_inputs={},
                           num_common_prefix_blocks=[],
                           finished_req_ids=finished_req_ids,
                           free_encoder_mm_hashes=[],
                           structured_output_request_ids={},
                           grammar_bitmask=None,
                           kv_connector_metadata=None)


def make_new_request_data(req_id, prompt_len):
    req = Request(request_id=req_id,
                  prompt_token_ids=[42] * prompt_len,
                  sampling_params=SamplingParams(),
                  pooling_params=None,
                  eos_token_id=None)
    return NewRequestData.from_request(req, block_ids=[])


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@default_test_params
def test_single_block_chunked_prefill(model: ModelInfo, max_model_len: int,
                                      max_num_seqs: int,
                                      max_num_batched_tokens: int,
                                      monkeypatch: pytest.MonkeyPatch):
    """A request that fits within a single block should be prefilled in one
    step and should not be padded out to the chunk boundary on decode"""
    runner: SpyreModelRunner = get_cpu_model_runner(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        monkeypatch=monkeypatch)

    req_id = "1"
    prompt_len = 48
    new_req_data = make_new_request_data(req_id, prompt_len)

    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[new_req_data],
        num_scheduled_tokens={req_id: prompt_len})

    output = runner.execute_model(scheduler_output)

    # one output token from prefill
    assert len(output.sampled_token_ids[0]) == 1
    # tkv is prompt_len + 1
    assert output.tkv == prompt_len + 1


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@default_test_params
def test_multi_chunk_padded_prefill(model: ModelInfo, max_model_len: int,
                                    max_num_seqs: int,
                                    max_num_batched_tokens: int,
                                    monkeypatch: pytest.MonkeyPatch):
    """A request that's longer than a chunk is split into multiple chunks"""
    runner: SpyreModelRunner = get_cpu_model_runner(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        monkeypatch=monkeypatch)

    req_id = "1"
    prompt_len = 148
    new_req_data = make_new_request_data(req_id, prompt_len)

    # Scheduler will give first 128 token chunk
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[new_req_data], num_scheduled_tokens={req_id: 128})
    output = runner.execute_model(scheduler_output)

    # no output tokens
    assert len(output.sampled_token_ids) == 0
    # 🌶️🌶️🌶️ We probably need to be able to pass back the number of tokens that
    # we actually processed so that the scheduler has an accurate count of
    # remaining prompt tokens.
    # Since this request needs to be padded to end in the page ending in 256,
    # it requires one full padding block on the left and will only process 64
    # tokens.
    # TODO: Uncomment when/if supported
    # assert output.computed_tokens[req_id] == 64
    computed_tokens = {req_id: 64}

    # Scheduler schedules remaining 148 - 64 = 84 tokens
    cached_req_data = make_cached_request_data(computed_tokens)
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 84})
    output = runner.execute_model(scheduler_output)

    # Should be one output token now
    assert len(output.sampled_token_ids[0]) == 1
    # TKV shouldn't reflect any prefill padding, it's just prompt + 1
    assert output.tkv == prompt_len + 1


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@default_test_params
def test_multi_chunk_unpadded_prefill(model: ModelInfo, max_model_len: int,
                                      max_num_seqs: int,
                                      max_num_batched_tokens: int,
                                      monkeypatch: pytest.MonkeyPatch):
    """A request that's longer than a chunk is split into multiple chunks"""
    runner: SpyreModelRunner = get_cpu_model_runner(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        monkeypatch=monkeypatch)

    req_id = "1"
    prompt_len = 200
    new_req_data = make_new_request_data(req_id, prompt_len)

    # Scheduler will give first 128 token chunk
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[new_req_data], num_scheduled_tokens={req_id: 128})
    output = runner.execute_model(scheduler_output)

    # no output tokens
    assert len(output.sampled_token_ids) == 0
    # No left padding block so all 128 tokens are computed
    # assert output.computed_tokens[req_id] == 128
    computed_tokens = {req_id: 128}

    # Scheduler schedules remaining 200 - 128 = 72 tokens
    cached_req_data = make_cached_request_data(computed_tokens)
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={req_id: 72})
    output = runner.execute_model(scheduler_output)

    # Should be one output token now
    assert len(output.sampled_token_ids[0]) == 1
    assert output.tkv == prompt_len + 1


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@default_test_params
def test_decode_padding_to_same_block(model: ModelInfo, max_model_len: int,
                                      max_num_seqs: int,
                                      max_num_batched_tokens: int,
                                      monkeypatch: pytest.MonkeyPatch):
    """A request that's longer than a chunk is split into multiple chunks"""
    runner: SpyreModelRunner = get_cpu_model_runner(
        model=model,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        monkeypatch=monkeypatch)

    short_req_id = "short"
    long_req_id = "long"
    short_prompt_len = 61
    long_prompt_len = 62

    steps = 1

    computed_tokens = lambda: {
        short_req_id: short_prompt_len + steps,
        long_req_id: long_prompt_len + steps
    }

    # Prefill both in one pass each
    new_req_data = make_new_request_data(short_req_id, short_prompt_len)
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[new_req_data],
        num_scheduled_tokens={short_req_id: short_prompt_len})
    output = runner.execute_model(scheduler_output)

    new_req_data = make_new_request_data(long_req_id, long_prompt_len)
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[new_req_data],
        num_scheduled_tokens={long_req_id: long_prompt_len})
    output = runner.execute_model(scheduler_output)

    # Both requests are still in the first block
    # Scheduler schedules 1 token each
    cached_req_data = make_cached_request_data(computed_tokens())
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={
            short_req_id: 1,
            long_req_id: 1
        })
    output = runner.execute_model(scheduler_output)
    steps += 1  # Step 2
    # TKV is the length of the long request since both are still in first block
    assert output.tkv == long_prompt_len + steps
    # Save the number of free blocks to compare once we allocate a new one
    initial_free_blocks = len(runner.block_pool)

    # Now the long request is at the first block boundary (tkv = 64)
    # We'll have to pad the short request to be within the same block by adding
    # a full block of left padding to it
    cached_req_data = make_cached_request_data(computed_tokens())
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={
            short_req_id: 1,
            long_req_id: 1
        })
    output = runner.execute_model(scheduler_output)
    steps += 1  # Step 3

    # 🌶️🌶️🌶️ short prompt gets padded, it's now the longest sequence
    assert output.tkv == short_prompt_len + steps + 64
    # We should have allocated only one new block for the long prompt entering
    assert len(runner.block_pool) == initial_free_blocks - 1

    # The shorter request is now at the second block boundary (tkv = 128), so we
    # can remove the extra block of padding (tkv = 64) since it will now be in
    # the second block along with the longer request
    cached_req_data = make_cached_request_data(computed_tokens())
    scheduler_output = make_scheduler_output(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=cached_req_data,
        num_scheduled_tokens={
            short_req_id: 1,
            long_req_id: 1
        })
    output = runner.execute_model(scheduler_output)
    steps += 1  # step 4

    # 🌶️🌶️🌶️ short prompt padding removed again, tkv is back to long + steps
    assert output.tkv == long_prompt_len + steps
    # One more real block was allocated for the short request
    assert len(runner.block_pool) == initial_free_blocks - 2

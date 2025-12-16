"""Verification of the correctness of the step-by-step execution of chunked
prefill with prefix caching. It does so by comparing, at every engine step 
(i.e. prefill or decode iteration), a bunch of attributes. 
This allows a finer testing of the padding and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_pc_inference_steps.py`.
"""

import pytest
from scheduling_utils import (create_request_for_scheduler_test, random_prompt,
                              validate_scheduler_steps)
from spyre_util import ModelInfo


def get_block_tables(engine_core):

    model_runner = (
        engine_core.model_executor.driver_worker.worker.model_runner)
    req_to_blocks = model_runner.kv_cache_manager.req_to_blocks

    block_tables = {
        req_id: [block.block_id for block in blocks]
        for req_id, blocks in req_to_blocks.items()
    }

    block_ref_count = {}

    for blocks in req_to_blocks.values():
        for block in blocks:
            block_ref_count[block.block_id] = block.ref_cnt

    return block_tables, block_ref_count


def verify_block_tables(engine_core, step_num, step_ref, disable_asserts):

    block_tables, block_ref_count = get_block_tables(engine_core)

    if not disable_asserts:
        if "block_tables" in step_ref:
            assert step_ref["block_tables"] == block_tables

        if "block_ref_count" in step_ref:
            assert step_ref["block_ref_count"] == block_ref_count
    else:
        print(f"{block_tables=}")
        print(f"{block_ref_count=}")


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_prefix_hit_within_batch(model: ModelInfo, backend: str,
                                 monkeypatch: pytest.MonkeyPatch,
                                 max_num_seqs: int, max_model_len: int,
                                 max_num_batched_tokens: int,
                                 available_blocks: int):
    """ Scenario where two equal sequences are scheduled. 
    While prefilling the second sequence we have a prefix cache
    hit and can reuse the first chunk. Note that the fetched prefix blocks 
    are still part of the existing decode batch. Hence we have duplicated 
    blocks in the block table for this example.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 192,  max tokens = 2, step joining = 0
            * 1: len = 192, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt = random_prompt(model=model, seed=0, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {  # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 192,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
            "block_tables": {
                '0': [1, 2, 3]
            },
            "block_ref_count": {
                1: 1,
                2: 1,
                3: 1
            }
        },
        {  # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 192,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
            "block_tables": {
                '0': [1, 2, 3]
            },
            "block_ref_count": {
                1: 1,
                2: 1,
                3: 1
            }
        },
        {  # prefill chunk 1 seq 1
            # prefix hit!
            "step": 3,
            "tkv": 192,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 8,
            "n_used_blocks": 3,
            "n_prefix_hits": 1,
            # each chunk has two blocks. Due to padding, the first chunk has
            # only one usable block
            "n_cached_blocks": 1,
            "block_tables": {
                '0': [1, 2, 3],
                '1': [1, 2, 3]
            },
            "block_ref_count": {
                1: 2,
                2: 2,
                3: 2
            }
        },
        {  # prefill chunk 2 seq 1
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 4,
            "tkv": 192,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
            "n_cached_blocks": 1,
            "block_tables": {
                '0': [1, 2, 3],
                '1': [1, 2, 3]
            },
            "block_ref_count": {
                1: 2,
                2: 2,
                3: 2
            }
        },
        {
            # Decode 1 of request 0.
            # Decode 1 of request 1.
            "step": 5,
            "tkv": 193,
            "waiting": [],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 5,
            "n_cached_blocks": 1,
            "block_tables": {
                '0': [1, 2, 3, 4],
                '1': [1, 2, 3, 5]
            },
            "block_ref_count": {
                1: 2,
                2: 2,
                3: 2,
                4: 1,
                5: 1
            }
        },
        {
            # Tkv should be cleared one step later
            "step": 6,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
            "block_tables": {},
            "block_ref_count": {}
        },
    ]

    validate_scheduler_steps(model=model,
                             backend=backend,
                             monkeypatch=monkeypatch,
                             requests=[request1, request2],
                             checked_steps=checked_steps,
                             max_num_seqs=max_num_seqs,
                             max_model_len=max_model_len,
                             available_blocks=available_blocks,
                             max_num_batched_tokens=max_num_batched_tokens,
                             prefix_caching=True,
                             extra_assert_func=verify_block_tables)


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_block_deduplication_within_batch(model: ModelInfo, backend: str,
                                          monkeypatch: pytest.MonkeyPatch,
                                          set_random_seed, max_num_seqs: int,
                                          max_model_len: int,
                                          max_num_batched_tokens: int,
                                          available_blocks: int):
    """ Scenario where two equal sequences are scheduled. As both sequences
    fit in a single chunk they have to be recomputed. However, we can write
    the KV cache into the same first block as the prompts are identical.
    Therefore we end up with a duplicated block in the block table despite
    not having a prefix hit for this example. 

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 70,  max tokens = 2, step joining = 0
            * 1: len = 70, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt = random_prompt(model=model, seed=0, length=70)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {  # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 70,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
            "n_prefix_hits": 0,
            "n_cached_blocks": 0,
            "block_tables": {
                '0': [1, 2]
            },
            "block_ref_count": {
                1: 1,
                2: 1
            }
        },
        {  # prefill chunk 1 seq 1
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 2,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
            "n_cached_blocks": 0,
            "block_tables": {
                '0': [1, 2],
                '1': [1, 3]
            },
            "block_ref_count": {
                1: 2,
                2: 1,
                3: 1
            }
        },
        {
            # Decode 1 of request 0.
            # Decode 1 of request 1.
            "step": 3,
            "tkv": 71,
            "waiting": [],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "block_tables": {
                '0': [1, 2],
                '1': [1, 3]
            },
            "block_ref_count": {
                1: 2,
                2: 1,
                3: 1
            }
        },
        {
            # Tkv should be cleared one step later
            "step": 4,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(model=model,
                             backend=backend,
                             monkeypatch=monkeypatch,
                             requests=[request1, request2],
                             checked_steps=checked_steps,
                             max_num_seqs=max_num_seqs,
                             max_model_len=max_model_len,
                             available_blocks=available_blocks,
                             max_num_batched_tokens=max_num_batched_tokens,
                             prefix_caching=True,
                             extra_assert_func=verify_block_tables)


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_prefix_hit_decoded_block_within_batch(model: ModelInfo, backend: str,
                                               monkeypatch: pytest.MonkeyPatch,
                                               max_num_seqs: int,
                                               max_model_len: int,
                                               max_num_batched_tokens: int,
                                               available_blocks: int):
    """ Scenario where two sequences are scheduled. We set the second
    sequence to be the entire first sequence plus some generated tokens.
    While prefilling the second sequence we have a prefix cache
    hit and can reuse the first chunk which consists of two blocks. The first
    block is entirely prompt while the second block is a mix of prompt and
    decoded tokens. Note that the fetched prefix blocks are still part of the 
    existing decode batch. Hence we have duplicated blocks in the block table 
    for this example.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 126,  max tokens = 68, step joining = 0
            * 1: len = 193, max tokens = 2, step joining = 67
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt = random_prompt(model=model, seed=0, length=126)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=68,
        prompt=prompt,
        use_golden_token_injection=True)

    # Next prompt uses part of the first request's output, matching 128 tokens
    # (2 blocks) in total
    prompt2 = prompt + request1.hf_output["token_ids"][:2] + \
        random_prompt(model=model, seed=0, length=65)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=67,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=True)


    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {   # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 126,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 2,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 0.
            "step": 2,
            "tkv": 127,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 2,
        },
        {
            # Decode 3 of request 0.
            # need an additional block
            "step": 4,
            "tkv": 129,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
        },
        {
            # Decode 66 of request 0.
            "step": 67,
            "tkv": 192,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
        },
        {   # prefill chunk 1 seq 1
            # prefix hit of the two blocks in the first chunk
            "step": 68,
            "tkv": 192, # why not 193
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 8,
            "n_used_blocks": 5,
            "n_prefix_hits": 1,
            # 1st block (prompt)
            # 2nd block (prompt + 2 decodes) <- what we want to test
            "n_cached_blocks": 2
        },
        {   # prefill chunk 2 seq 1
            # no prefix hit, always recompute last chunk
            "step": 69,
            "tkv": 192, # why not 193
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 5,
            "n_prefix_hits": 0,
            "n_cached_blocks": 2
        },
        {
            # Decode 1 of request 0.
            # Decode 1 of request 1.
            "step": 70,
            "tkv": 194,
            "waiting": [],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 6,
            "n_cached_blocks": 2
        },
        {
            # Tkv should be cleared one step later
            "step": 71,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_prefix_hit_not_in_batch(model: ModelInfo, backend: str,
                                 monkeypatch: pytest.MonkeyPatch,
                                 max_num_seqs: int, max_model_len: int,
                                 max_num_batched_tokens: int,
                                 available_blocks: int):
    """ Scenario where two equal sequences are scheduled. 
    While prefilling the second sequence we have a prefix cache
    hit and can reuse the first chunk. Note that the fetched prefix blocks 
    are not part of the existing decode batch as the sequence has already 
    left the batch at the time of prefilling the new sequence. Hence we have 
    no duplicated blocks in the block table for this example.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 192,  max tokens = 2, step joining = 0
            * 1: len = 192, max tokens = 2, step joining = 3
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt = random_prompt(model=model, seed=0, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=3,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {   # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 192,
            "waiting": [],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 192,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 0.
            # request 1 joined the waiting queue
            "step": 3,
            "tkv": 193,
            "waiting": ["1"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {   # prefill chunk 1 seq 1
            # prefix hit!
            "step": 4,
            "tkv": 192,
            "waiting": [],
            "running": ["1"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 1,
            "n_cached_blocks": 1
        },
        {   # prefill chunk 2 seq 1
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 5,
            "tkv": 192,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
            "n_cached_blocks": 1
        },
        {
            # Decode 1 of request 0.
            "step": 6,
            "tkv": 193,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
            "n_cached_blocks": 1
        },
        {
            # Tkv should be cleared one step later
            "step": 7,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [4])
def test_limit_blocks_no_prefix_hit(model: ModelInfo, backend: str,
                                    monkeypatch: pytest.MonkeyPatch,
                                    max_num_seqs: int, max_model_len: int,
                                    max_num_batched_tokens: int,
                                    available_blocks: int):
    """ Scenario where three sequences are scheduled with the 1st and 3rd
    sequences being identical. While prefilling the third sequence we don't 
    have a prefix cache hit for the first chunk as the KV cache has already 
    been overwritten. This is because we limit the number of available blocks
    to 4. Note: When increasing the number of available blocks to 8, see
    test_limit_blocks_prefix_hit, the same test results in a prefix hit. 

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 0: len = 192,  max tokens = 2, step joining = 0
            * 1: len = 192, max tokens = 2, step joining = 3
            * 2: len = 192, max tokens = 2, step joining = 6
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt1 = random_prompt(model=model, seed=0, length=192)
    prompt2 = random_prompt(model=model, seed=1, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=3,
        max_tokens=2,
        prompt=prompt2,  # 1st and 3rd sequence are the same
        use_golden_token_injection=True)

    request3 = create_request_for_scheduler_test(
        model=model,
        request_id=2,
        add_step=6,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {   # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 192,
            "waiting": [],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 192,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 0
            # request 1 joined the waiting queue
            "step": 3,
            "tkv": 193,
            "waiting": ["1"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {   # prefill chunk 1 seq 1
            "step": 4,
            "tkv": 192,
            "waiting": [],
            "running": ["1"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 1
            "step": 5,
            "tkv": 192,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 1
            # request 2 joined the waiting queue
            "step": 6,
            "tkv": 193,
            "waiting": ['2'],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {   # prefill chunk 1 seq 2
            # no prefix hit as KV cache is already overwritten!
            "step": 7,
            "tkv": 192,
            "waiting": [],
            "running": ["2"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 2
            "step": 8,
            "tkv": 192,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 2
            "step": 9,
            "tkv": 193,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {
            # Tkv should be cleared one step later
            "step": 10,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2, request3],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [4])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_double_prefix_hit_within_batch(model: ModelInfo, backend: str,
                                        monkeypatch: pytest.MonkeyPatch,
                                        max_num_seqs: int, max_model_len: int,
                                        max_num_batched_tokens: int,
                                        available_blocks: int):
    """ Scenario where three equal and one different sequences are scheduled.
    While prefilling the second and fourth sequence we have a prefix cache
    hit and can reuse the first chunk. Note that the fetched prefix blocks 
    are still part of the existing decode batch. Hence we have duplicated 
    blocks in the block table for this example. More specifically, three
    sequences in the decode batch share the same KV cache block.

    Configuration:
        * max_num_seqs: 4
        * number of prompts: 4
            * 0: len = 192,  max tokens = 2, step joining = 0
            * 1: len = 192, max tokens = 2, step joining = 0
            * 2: len = 192, max tokens = 2, step joining = 0
            * 3: len = 192, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt1 = random_prompt(model=model, seed=0, length=192)
    prompt2 = random_prompt(model=model, seed=1, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    request3 = create_request_for_scheduler_test(
        model=model,
        request_id=2,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,  # This request has a different prompt
        use_golden_token_injection=True)

    request4 = create_request_for_scheduler_test(
        model=model,
        request_id=3,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2", "3"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {   # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 192,
            "waiting": ["1", "2", "3"],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 192,
            "waiting": ["1", "2", "3"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 1 seq 1
            # prefix hit!
            "step": 3,
            "tkv": 192,
            "waiting": ["2", "3"],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 8,
            "n_used_blocks": 3,
            "n_prefix_hits": 1,
        },
        {   # prefill chunk 2 seq 1
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 4,
            "tkv": 192,
            "waiting": ["2", "3"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 1 seq 2
            "step": 5,
            "tkv": 192,
            "waiting": ["3"],
            "running": ["2", "1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 12,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 2
            "step": 6,
            "tkv": 192,
            "waiting": ["3"],
            "running": ["2", "1", "0"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 12,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 1 seq 3
            # prefix hit!
            "step": 7,
            "tkv": 192,
            "waiting": [],
            "running": ["3", "2", "1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 16,
            "n_used_blocks": 6,
            "n_prefix_hits": 1,
        },
        {   # prefill chunk 2 seq 3
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 8,
            "tkv": 192,
            "waiting": [],
            "running": ["3", "2", "1", "0"],
            "request_outputs": ["3"],
            "n_reserved_blocks": 16,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 0, 1, 2, 3
            "step": 9,
            "tkv": 193,
            "waiting": [],
            "running": [],
            "request_outputs": ["3", "2", "1", "0"],
            "finished_requests": ["3", "2", "1", "0"],
            "n_reserved_blocks": 16,
            "n_used_blocks": 10,
            "n_used_btests/e2e/test_spyre_pc_scheduler_steps.pylocks": 10
        },
        {
            # Tkv should be cleared one step later
            "step": 10,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2, request3, request4],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [8])
def test_limit_blocks_prefix_hit(model: ModelInfo, backend: str,
                                 monkeypatch: pytest.MonkeyPatch,
                                 max_num_seqs: int, max_model_len: int,
                                 max_num_batched_tokens: int,
                                 available_blocks: int):
    """ Scenario where three sequences are scheduled with the 1st and 3rd
    sequences being identical. While prefilling the third sequence we 
    have a prefix cache hit for the first chunk as the KV cache is still
    persistent. This is because the number of available blocks (8) is high
    enough. Note: When decreasing the number of available blocks to 4, see
    test_limit_blocks_no_prefix_hit, the same test results in a no prefix hit. 

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 0: len = 192,  max tokens = 2, step joining = 0
            * 1: len = 192, max tokens = 2, step joining = 3
            * 2: len = 192, max tokens = 2, step joining = 6
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt1 = random_prompt(model=model, seed=0, length=192)
    prompt2 = random_prompt(model=model, seed=1, length=192)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=3,
        max_tokens=2,
        prompt=prompt2,  # 1st and 3rd sequence are the same
        use_golden_token_injection=True)

    request3 = create_request_for_scheduler_test(
        model=model,
        request_id=2,
        add_step=6,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {   # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 192,
            "waiting": [],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 192,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 0
            # request 1 joined the waiting queue
            "step": 3,
            "tkv": 193,
            "waiting": ["1"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {   # prefill chunk 1 seq 1
            "step": 4,
            "tkv": 192,
            "waiting": [],
            "running": ["1"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 1
            "step": 5,
            "tkv": 192,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {
            # Decode 1 of request 1
            # request 2 joined the waiting queue
            "step": 6,
            "tkv": 193,
            "waiting": ['2'],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {   # prefill chunk 1 seq 2
            # prefix hit as KV cache is still persistent
            "step": 7,
            "tkv": 192,
            "waiting": [],
            "running": ["2"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 1,
            "n_cached_blocks": 1
        },
        {   # prefill chunk 2 seq 2
            "step": 8,
            "tkv": 192,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
            "n_cached_blocks": 1
        },
        {
            # Decode 1 of request 2
            "step": 9,
            "tkv": 193,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
            "n_cached_blocks": 1
        },
        {
            # Tkv should be cleared one step later
            "step": 10,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2, request3],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [512])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_multi_chunk_full_match(model: ModelInfo, backend: str,
                                monkeypatch: pytest.MonkeyPatch,
                                max_num_seqs: int, max_model_len: int,
                                max_num_batched_tokens: int,
                                available_blocks: int):
    """ Scenario where two equal sequences are scheduled.
    Both sequences have exactly 3 chunks worth of tokens, thus
    resulting in a 100% match up to the last token. This test
    makes sure that the last chunk is not reused.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 384, max tokens = 2, step joining = 0
            * 1: len = 384, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    prompt = random_prompt(model=model, seed=0, length=384)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {   # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 384,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 384,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 3 seq 0
            "step": 3,
            "tkv": 384,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
            # up until this point nothing interesting happened
            # with the block table
            "block_tables": {'0': [1, 2, 3, 4, 5, 6]},
            "block_ref_count": {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
        },
        {   # prefill chunk 1 seq 1
            # prefix hit!
            "step": 4,
            "tkv": 384,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 14,
            "n_used_blocks": 6,
            "n_prefix_hits": 1,
            # The number of cached blocks is determined up front
            "n_cached_blocks": 4, # can reuse the first two chunk (4 blocks)
            # Now, although the last chunk has to be recomputed,
            # the blocks are still shared.
            "block_tables": {'0': [1, 2, 3, 4, 5, 6], '1': [1, 2, 3, 4, 5, 6]},
            "block_ref_count": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2}
        },
        {   # prefill chunk 2 seq 1
            # prefix hit!
            "step": 5,
            "tkv": 384,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 14,
            "n_used_blocks": 6,
            "n_prefix_hits": 1,
            "n_cached_blocks": 4
        },
        {   # prefill chunk 3 seq 1
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 6,
            "tkv": 384,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 14,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
            "n_cached_blocks": 4
        },
        {
            # Decode 1 of request 0.
            # Decode 1 of request 1.
            "step": 7,
            "tkv": 385,
            "waiting": [],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 14,
            "n_used_blocks": 8,
            "n_cached_blocks": 4,
            # when decode starts, we see the tables diverge
            "block_tables": {
                '0': [1, 2, 3, 4, 5, 6, 7],
                '1': [1, 2, 3, 4, 5, 6, 8]
            },
            "block_ref_count": {1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 1, 8: 1}
        },
        {
            # Tkv should be cleared one step later
            "step": 8,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
    )


@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [512])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_multi_chunk_partial_match(model: ModelInfo, backend: str,
                                   monkeypatch: pytest.MonkeyPatch,
                                   max_num_seqs: int, max_model_len: int,
                                   max_num_batched_tokens: int,
                                   available_blocks: int):
    """ Scenario where two sequences are scheduled which share a common
    prefix. The second sequence shares 254 tokens with the first sequence,
    which is less than two chunks. We can therefore reuse only one chunk
    (254 < 2*128 = 256).

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 384,  max tokens = 2, step joining = 0
            * 1: len = 384, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    # twice the same seed for a sequence of length 384
    # the first sequence shares the same prefix of length 384 tokens
    # the second sequence shares the same prefix of length 254 tokens
    # hence sequence 1 shares the first 254 tokens with sequence 0

    prompt1 = random_prompt(model=model, seed=0, length=384)
    prompt2 = prompt1[0:254] + \
        random_prompt(model=model, seed=0, length=384 - 254)

    request1 = create_request_for_scheduler_test(
        model=model,
        request_id=0,
        add_step=0,
        max_tokens=2,
        prompt=prompt1,
        use_golden_token_injection=True)

    request2 = create_request_for_scheduler_test(
        model=model,
        request_id=1,
        add_step=0,
        max_tokens=2,
        prompt=prompt2,
        use_golden_token_injection=True)

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {   # prefill chunk 1 seq 0
            "step": 1,
            "tkv": 384,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 384,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 3 seq 0
            "step": 3,
            "tkv": 384,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 1 seq 1
            # prefix hit!
            "step": 4,
            "tkv": 384,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 14,
            "n_used_blocks": 9,
            "n_prefix_hits": 1,
            # The number of cached blocks is determined up front
            "n_cached_blocks": 2 # can only reuse the first chunk (2 blocks)
        },
        {   # prefill chunk 2 seq 1
            # cannot use prefix, as the prefix is less than 2 chunks
            "step": 5,
            "tkv": 384,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 14,
            "n_used_blocks": 9,
            "n_prefix_hits": 0,
            "n_cached_blocks": 2
        },
        {   # prefill chunk 3 seq 1
            "step": 6,
            "tkv": 384,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 14,
            "n_used_blocks": 9,
            "n_prefix_hits": 0,
            "n_cached_blocks": 2
        },
        {
            # Decode 1 of request 0.
            # Decode 1 of request 1.
            "step": 7,
            "tkv": 385,
            "waiting": [],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 14,
            "n_used_blocks": 11,
            "n_cached_blocks": 2
        },
        {
            # Tkv should be cleared one step later
            "step": 8,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    validate_scheduler_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        requests=[request1, request2],
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
        extra_assert_func=verify_block_tables,
    )

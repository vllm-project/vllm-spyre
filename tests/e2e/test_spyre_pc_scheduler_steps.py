"""Verification of the correctness of the step-by-step execution of chunked
prefill with prefix caching. It does so by comparing, at every engine step 
(i.e. prefill or decode iteration), a bunch of attributes. 
This allows a finer testing of the padding and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_pc_inference_steps.py`.
"""

import pytest
from scheduling_utils import check_scheduler_inference_steps
from spyre_util import ModelInfo


@pytest.mark.cpu
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
                                 set_random_seed, max_num_seqs: int,
                                 max_model_len: int,
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

    seqs_max_tokens = [2, 2]
    prompts_lengths = [192, 192]
    steps_add_reqs = [0, 0]
    seeds = [0, 0]  # twice the same sequence

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
            "tkv": 192,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": [],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3,
            "n_prefix_hits": 0,
        },
        {   # prefill chunk 2 seq 0
            "step": 2,
            "tkv": 192,
            "waiting": ["1"],
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
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 8,
            "n_used_blocks": 6,
            "n_prefix_hits": 1,
            # each chunk has two blocks. Due to padding, the first chunk has
            # only one usable block
            "n_cached_blocks": 1
        },
        {   # prefill chunk 2 seq 1
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 4,
            "tkv": 192,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 6,
            "n_prefix_hits": 0,
            "n_cached_blocks": 1
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
            "n_used_blocks": 8,
            "n_cached_blocks": 1
        },
        {
            # Tkv should be cleared one step later
            "step": 6,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=False,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
        seeds=seeds,
    )


@pytest.mark.cpu
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
                                 set_random_seed, max_num_seqs: int,
                                 max_model_len: int,
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

    seqs_max_tokens = [2, 2]
    prompts_lengths = [192, 192]
    # sequence 1 joins only at step 3, when seq 0 has already finished
    steps_add_reqs = [0, 3]
    seeds = [0, 0]  # twice the same sequence

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

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=False,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
        seeds=seeds,
    )


@pytest.mark.cpu
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
                                    set_random_seed, max_num_seqs: int,
                                    max_model_len: int,
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

    seqs_max_tokens = [2, 2, 2]
    prompts_lengths = [192, 192, 192]
    # sequence 1 joins only at step 3, when seq 0 has already finished
    # sequence 2 joins only at step 6, when seq 1 has already finished
    steps_add_reqs = [0, 3, 6]
    seeds = [0, 1, 0]  # 1st and 3rd sequence are the same

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

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=False,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
        seeds=seeds)


@pytest.mark.cpu
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
                                 set_random_seed, max_num_seqs: int,
                                 max_model_len: int,
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

    seqs_max_tokens = [2, 2, 2]
    prompts_lengths = [192, 192, 192]
    # sequence 1 joins only at step 3, when seq 0 has already finished
    # sequence 2 joins only at step 6, when seq 1 has already finished
    steps_add_reqs = [0, 3, 6]
    seeds = [0, 1, 0]  # 1st and 3rd sequence are the same

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

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=False,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
        seeds=seeds)


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.prefix_caching
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [512])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_full_match(model: ModelInfo, backend: str,
                    monkeypatch: pytest.MonkeyPatch, set_random_seed,
                    max_num_seqs: int, max_model_len: int,
                    max_num_batched_tokens: int, available_blocks: int):
    """ Scenario where two equal sequences are scheduled.
    Both sequences have exactly 3 chunks worth of tokens, thus
    resulting in a 100% match up to the last token. This test
    makes sure that the last chunk is not reused.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 384,  max tokens = 2, step joining = 0
            * 1: len = 384, max tokens = 2, step joining = 0
    """
    monkeypatch.setenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "0")

    seqs_max_tokens = [2, 2]
    prompts_lengths = [384, 384]
    steps_add_reqs = [0, 0]
    seeds = [0, 0]  # twice the same sequence

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
            "n_used_blocks": 12,
            "n_prefix_hits": 1,
            # The number of cached blocks is determined up front
            "n_cached_blocks": 4
        },
        {   # prefill chunk 2 seq 1
            # cannot use prefix, as the last chunk has to always be recomputed
            "step": 5,
            "tkv": 384,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": [],
            "n_reserved_blocks": 14,
            "n_used_blocks": 12,
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
            "n_used_blocks": 12,
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
            "n_used_blocks": 14,
            "n_cached_blocks": 4
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

    check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=False,
        random_prompts=True,
        max_num_batched_tokens=max_num_batched_tokens,
        prefix_caching=True,
        seeds=seeds,
    )

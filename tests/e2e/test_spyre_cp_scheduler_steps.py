"""Verification of the correctness of the step-by-step execution of chunked
prefill. It does so by comparing, at every engine step (i.e. prefill or decode 
iteration), a bunch of attributes. This allows a finer testing of the padding 
and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_cp_inference_steps.py`.
"""

import pytest
from scheduling_utils import check_scheduler_inference_steps
from spyre_util import ModelInfo


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len",
                         [128])  # restricted to violate scheduler condition
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_prefill_tkv_too_big(model: ModelInfo, backend: str,
                             monkeypatch: pytest.MonkeyPatch, set_random_seed,
                             max_num_seqs: int, max_model_len: int,
                             max_num_batched_tokens: int,
                             available_blocks: int):
    """ Scenario where the requested prompt is too long for current tkv value
   
    Note that as we could prefill the prompt straight away, however,
    in this test the max model length is decreased to a value where
    the tkv of the decode batch would be shifted beyond the max model length, 
    we therefore have to wait with scheduling.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 49, max tokens = 17, step joining = 0
            * 1: len = 70, max tokens = 17, step joining = 0
    """

    seqs_max_tokens = [17, 17]
    prompts_lengths = [49, 70]
    steps_add_reqs = [0, 0]

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
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 49,  # prompt len
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks":
            2,  # prefill (1 block) + 17 decodes (1 additional block)
            "n_used_blocks": 1
        },
        # Here we cannot schedule sequence 1. By shifting sequence 0 by
        # 1 block its max tkv would exceed the max model length
        {
            # Decode sequence 0
            # total blocks in use: 1 (writing into right pads)
            "step": 2,
            "tkv": 50,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 1
        },
        {
            # Prefill sequence 1, tkv large enough to prefill w/o tkv shift
            # total blocks in use: 1 + 2
            "step": 17,
            "tkv": 64,  # @Wallas, should be max(64, 70) here? 
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 2 + 2 (prefill (2 block) + 17 decodes in the last block)
            "n_reserved_blocks": 4,
            "n_used_blocks": 3
        },
        {
            # Decode sequences 0 and 1
            # Sequence 0 finishes
            "step": 18,
            "tkv": 71,  # tkv of seq 1 is now max
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4  # seq 0 needs another block for the last token 
        },
        {
            # Decode sequence 1
            # total blocks in use: 4 - 2 = 2
            "step": 19,
            "tkv": 72,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Sequence 1 finishes at step 33
            "step": 33,
            "tkv": 86,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Tkv should be cleared one step later
            "step": 34,
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
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
# provide only 2 blocks, but at least 3 blocks would be required to schedule
# the sequences together in a batch.
@pytest.mark.parametrize("available_blocks", [2])
def test_prefill_use_more_than_available_blocks(
        model: ModelInfo, backend: str, monkeypatch: pytest.MonkeyPatch,
        set_random_seed, max_num_seqs: int, max_model_len: int,
        max_num_batched_tokens: int, available_blocks: int):
    """ Scenario where the number of available KV cache blocks is decreased
    to a value where the the number of reserved blocks would exceed the number
    of available blocks, we therefore have to wait scheduling the request. 

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 49, max tokens = 3, step joining = 0
            * 1: len = 70, max tokens = 3, step joining = 0
        * available_blocks: 2
    """

    seqs_max_tokens = [3, 3]
    prompts_lengths = [49, 70]
    steps_add_reqs = [0, 0]

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
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 49,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 1,
            "n_used_blocks": 1
        },
        # We cannot schedule sequence 1 here. Needs 2 more blocks, but only
        # a total of 2 blocks available.
        {
            # Decode sequence 0
            "step": 2,
            "tkv": 50,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 1,
            "n_used_blocks": 1
        },
        {
            # Decode sequence 0
            # seq 0 finishes in this step
            "step": 3,
            "tkv": 51,
            "waiting": ["1"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 1,
            "n_used_blocks": 1
        },
        {
            # Prefill sequence 1
            "step": 4,
            "tkv": 70,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Decode sequence 1
            "step": 5,
            "tkv": 71,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Decode sequence 0
            # seq 0 finishes in this step
            "step": 6,
            "tkv": 72,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
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
        max_num_batched_tokens=max_num_batched_tokens,
    )


@pytest.mark.cpu
@pytest.mark.chunked_prefill
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_requests_exceed_batch_tkv_limit(model: ModelInfo, backend: str,
                                         monkeypatch: pytest.MonkeyPatch,
                                         set_random_seed, max_num_seqs: int,
                                         max_model_len: int,
                                         max_num_batched_tokens: int,
                                         available_blocks: int):
    """ Scenario where a request cannot be scheduled right away as the
    max batch x tkv limit, e.g the volumetric limit, is exceeded.
    
    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 64, max tokens = 2, step joining = 0
            * 2: len = 65, max tokens = 2, step joining = 0
    """

    seqs_max_tokens = [2, 2]
    prompts_lengths = [64, 65]
    steps_add_reqs = [0, 0]
    # total number of blocks needed if scheduled together: (1 + 1)+(1 + 1) = 4
    # note that as not scheduled together, we only needs 2 blocks here
    # needs 2 * (64 + 1) = 2 * 65 = 130
    max_batch_tkv_limit = 129  # not big enough

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
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 1 decode (1 block)
            "n_used_blocks": 1
        },
        # Note: we cannot prefill seq 1 as the volumetric limit
        # max_batch_tkv_limit is exceeded: 129 < 130
        # -> cond5 in can_schedule() is False
        {
            # Decode sequence 0
            # Sequence 0 finishes at step 2
            # total blocks in use: 2
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Prefill sequence 1
            # total blocks in use: 2
            "step": 3,
            "tkv": 65,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2  # 2 - 2 + 2
        },
        {
            # Decode sequence 1
            # Sequence 1 finishes at step 4
            # total blocks in use: 3
            "step": 4,
            "tkv": 66,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Tkv should be cleared one step later
            # total blocks in use: 2 - 2 = 0
            "step": 5,
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
        max_batch_tkv_limit=max_batch_tkv_limit,
        use_cb=False,
        max_num_batched_tokens=max_num_batched_tokens,
    )

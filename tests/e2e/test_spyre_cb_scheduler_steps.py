"""Verification of the correctness of the step-by-step execution of continuous
batching. It does so by comparing, at every engine step (i.e. prefill or decode
iteration), a bunch of attributes. This allows a finer testing of the padding
and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_cb_inference_steps.py`.
"""

import pytest
from scheduling_utils import check_scheduler_inference_steps
from spyre_util import ModelInfo


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("available_blocks", [None])
def test_prompts_aligned_with_tkv_boundaries(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed: None,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where it happens that all the sequences get scheduled in a
    fashion where they are aligned with the block boundaries (i.e. tkv multiple
    of 64 at the time of prefilling).

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 0: len = 49, max tokens = 65, step joining = 0
            * 1: len = 41, max tokens = 67, step joining = 0
            * 2: len = 47, max tokens = 4, step joining = 0
    """

    seqs_max_tokens = [65, 67, 4]
    prompts_lengths = [49, 41, 47]
    steps_add_reqs = [0, 0, 0]  # add all requests in the beginning

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 64 decodes (1 block)
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 1 + 1 = 2
            "step": 2,
            "tkv": 64,  # Still 64 because this step is also a prefill
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # prefill (1 block)  + 66 decodes (2 blocks)
            "n_reserved_blocks": 5,
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 0 and 1
            # total blocks in use: 2 + 2 = 4
            "step": 3,
            "tkv": 65,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4,
        },
        {
            # Sequence 0 finishes at step 66
            # (start step + 2 prefills + 64 decodes - 1) = 1 + 2 + 64 - 1 = 66
            "step": 66,
            "tkv": 128,
            "waiting": ["2"],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4,
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 2 + 1 = 3
            "step": 67,
            "tkv": 128,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"],
            # 5 - 2 (seq 0) + 2 (prefill (1 block) + decodes (1 block))
            "n_reserved_blocks": 5,
            "n_used_blocks": 3,
        },
        {
            # Decode sequences 1 and 2
            # total blocks in use: 3 + 2 = 5
            "step": 68,
            "tkv": 129,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2", "1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 5,
        },
        {
            # Sequence 1 finishes at step 69
            # (start step + 2 prefills + 66 decodes - 1) = 2 + 2 + 66 - 1 = 69
            "step": 69,
            "tkv": 130,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2", "1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 5,
        },
        {
            # Sequence 2 finishes at step 70
            # (start step + 1 prefill + 3 decodes - 1) = 67 + 1 + 3 - 1 = 70
            "step": 70,
            "tkv": 67,  # tkv is reset by 64 due to removing the padded block
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Tkv should be cleared one step later
            "step": 71,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("available_blocks", [None])
def test_prompts_misaligned_with_tkv_boundaries(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed: None,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where it happens that some sequence gets scheduled in a way
    that it is misaligned with the block boundary (i.e. tkv is not a multiple
    of 64 at the time of prefilling).

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 0: len = 49, max tokens = 10, step joining = 0
            * 1: len = 41, max tokens = 13, step joining = 0
            * 2: len = 5, max tokens = 2, step joining = 0
    """
    seqs_max_tokens = [10, 13, 2]
    prompts_lengths = [49, 41, 5]
    steps_add_reqs = [0, 0, 0]  # add all requests in the beginning

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 10 decodes (1 block)
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 1 + 1 = 2
            "step": 2,
            "tkv": 64,  # Still 64 because this step is also a prefill
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,  # prefill (1 block) + 12 decodes (1 block)
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 0 and 1
            # total blocks in use: 2 + 2 = 4
            "step": 3,
            "tkv": 65,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Sequence 0 finishes at step 11
            # (start step + 2 prefills + 9 decodes - 1) = 1 + 2 + 9 - 1 = 11
            "step": 11,
            "tkv": 73,
            "waiting": ["2"],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 2 + 1 = 3
            "step": 12,
            "tkv": 73,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"],
            # 4 - 2 (seq 0) + 1 (prefill (1 block) + 8 decodes in 1st block)
            "n_reserved_blocks": 3,
            "n_used_blocks": 3,
        },
        {
            # Sequence 2 finishes at step 13
            # (start step + 1 prefill + 1 decodes - 1) = 12 + 1 + 1 - 1 = 13
            "step": 13,
            "tkv": 74,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["2", "1"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3,
        },
        {
            # Decode sequences 1
            # total blocks in use: 3 - 1 + 1 = 3
            "step": 14,
            "tkv": 75,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 2,  # 3 - 1 (seq 2)
            "n_used_blocks": 2,
        },
        {
            # Sequence 1 finishes at step 15
            # (start step + 2 prefills + 12 decodes - 1) = 2 + 2 + 12 - 1 = 15
            "step": 15,
            "tkv": 76,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Tkv should be cleared one step later
            "step": 16,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_two_sequences_finish_same_time_as_new_arrive(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """2-cases-in-1: (1) Two sequences finish at the same time and (2) a new
    request arrives when another finishes.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 0: len = 49, max tokens = 4, step joining = 0
            * 1: len = 30, max tokens = 4, step joining = 0
            * 2: len = 20, max tokens = 3, step joining = 5
    """
    seqs_max_tokens = [4, 4, 3]
    prompts_lengths = [49, 30, 20]
    steps_add_reqs = [0, 0, 5]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 1 + 1 = 2
            "step": 2,
            "tkv": 64,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 0 and 1
            # total blocks in use: 2 + 2 = 4
            "step": 3,
            "tkv": 65,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Sequences 0 and 1 finish at step 5
            # (start step + 2 prefills + 3 decodes - 1) = 1 + 2 + 3 - 1 = 5
            # (start step + 1 prefills + 29 decodes - 1) = 2 + 1 + 3 - 1 = 5
            # Sequence 2 joins: one iteration in waiting queue
            "step": 5,
            "tkv": 67,
            "waiting": ["2"],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 4 + 2
            "step": 6,
            "tkv": 64,  # tkv is reset by 64 due to removing the padded block
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            # 4 - 4 + 2 (prefill (1 block) + 2 decodes (1 block))
            "n_reserved_blocks": 2,
            "n_used_blocks": 1,
        },
        {
            # Decode sequence 2
            # total blocks in use: 2
            "step": 7,
            "tkv": 65,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Sequences 2 finishes at step 8
            # (start step + 1 prefill + 2 decodes - 1) = 6 + 1 + 2 - 1 = 8
            "step": 8,
            "tkv": 66,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Tkv should be cleared one step later
            "step": 9,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [3])
@pytest.mark.parametrize("max_model_len", [192])
@pytest.mark.parametrize(
    "available_blocks", [12]
)  # specific value required to pass compilation with this config
def test_new_sequence_joins_during_decode(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where a new sequence joins while decoding other sequences.
    Sequence 1 joins when tkv is in the middle of a block (tkv=94), sequence 2
    joins when tkv is a the end of a block (tkv=128).

    Configuration:
        * max_num_seqs: 3
        * number of prompts: 4
            * 0: len = 49, max tokens = 60, step joining = 0
            * 1: len = 89, max tokens = 37, step joining = 32
            * 2: len = 9, max tokens = 3, step joining = 67
    """
    seqs_max_tokens = [60, 37, 3]
    prompts_lengths = [49, 89, 9]
    steps_add_reqs = [0, 31, 66]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            "step": 1,
            "tkv": 64,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 59 decode (1 block)
            "n_used_blocks": 1,
        },
        {
            # Decode sequences 0
            "step": 2,
            "tkv": 65,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Sequence 1 joins: one iteration in waiting queue
            "step": 31,
            "tkv": 94,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 1
            "step": 32,
            "tkv": 94,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 5,  # prefill (2 block) + 36 decode (1 block)
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0 and 1
            "step": 33,
            "tkv": 95,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4,
        },
        {
            # Sequence 0 finishes at step 61
            # (start step + 2 prefills + 59 decodes - 1) = 1 + 2 + 59 - 1 = 61
            "step": 61,
            "tkv": 123,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 1
            "step": 62,
            "tkv": 124,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 3,  # 2 blocks released
            "n_used_blocks": 2,  # 2 blocks released
        },
        {
            # Sequence 2 joins: one iteration in waiting queue
            "step": 66,
            "tkv": 128,
            "waiting": ["2"],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 2
            "step": 67,
            "tkv": 128,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"],
            # Note: here is where the optimization happens: we do the prefill
            # on a single block only instead of using 2 blocks
            "n_reserved_blocks": 5,  # prefill (1 block) + 2 decode (1 block)
            "n_used_blocks": 3,  # prefill (1 block)
        },
        {
            # Decode sequences 1 and 2, tkv expands to new block
            "step": 68,
            "tkv": 129,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2", "1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 5,  # 2 blocks extended, one for each sequence
        },
        {
            # Sequences 1 and 2 finish at step 69
            # (start step + 2 prefills + 36 decodes - 1) = 32 + 2 + 36 - 1 = 69
            # (start step + 1 prefills + 3 decodes - 1) = 67 + 1 + 2 - 1 = 69
            "step": 69,
            "tkv": 130,
            "waiting": [],
            "running": [],
            "request_outputs": ["2", "1"],
            "finished_requests": ["2", "1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 5,
        },
        {
            # Tkv should be cleared one step later
            "step": 70,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [192])
@pytest.mark.parametrize("available_blocks", [None])
def test_prompt_too_long_for_current_tkv(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where the requested prompt is too long for current tkv value

    Note that we can prefill the prompt straight away

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 49, max tokens = 10, step joining = 0
            * 1: len = 70, max tokens = 4, step joining = 0
    """

    seqs_max_tokens = [10, 4]
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
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 9 decodes (1 block)
            "n_used_blocks": 1,
        },
        # due to allowing sequences to join the current decode batch even if
        # prompt length > tkv, prefill of sequence 1 happens immediately
        {
            # Prefill sequence 1
            # total blocks in use: 1 + 2
            "step": 2,
            "tkv": 128,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 2 + 3 (prefill (2 block) + 3 decodes (1 block))
            "n_reserved_blocks": 5,
            "n_used_blocks": 3,
        },
        {
            # Decode sequences 0 and 1
            "step": 3,
            "tkv": 129,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 5,  # 3 + 2 = 5
        },
        {
            # Sequence 1 finishes at step 5
            # (start step + 1 prefill + 3 decodes - 1) = 2 + 1 + 3 - 1 = 5
            "step": 5,
            "tkv": 131,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 5,
        },
        {
            # Decode sequence 0
            # total blocks in use: 5 - 3 = 2
            "step": 6,
            "tkv": 68,  # tkv is reset by 64 due to removing the padded block
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # 5 - 3 (seq 1)
            "n_used_blocks": 2,
        },
        {
            # Sequence 0 finishes at step 11
            # (start step + 2 prefills + 9 decodes - 1) = 1 + 2 + 9 - 1 = 11
            "step": 11,
            "tkv": 73,
            "waiting": [],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Tkv should be cleared one step later
            "step": 12,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [192])  # restricted to violate scheduler condition
@pytest.mark.parametrize("available_blocks", [None])
def test_prefill_tkv_too_big(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where the requested prompt is too long for current tkv value

    Note that as we could prefill the prompt straight away. However,
    in this test the max model length is decreased to a value where
    the tkv of the decode batch would be shifted beyond the max model length,
    we therefore have to wait with scheduling.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 49, max tokens = 67, step joining = 0
            * 1: len = 70, max tokens = 50, step joining = 0
    """

    seqs_max_tokens = [67, 50]
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
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 3,  # prefill (1 block) + 66 decodes (2 blocks)
            "n_used_blocks": 1,
        },
        # Here we cannot schedule sequence 1. By shifting sequence 0 by
        # 1 block its max tkv would exceed the max model length:
        # 64 + 67 - 1 + 64 (shift) = 194 > 192 (max model length)
        {
            # Decode sequence 0
            # total blocks in use: 1 + 1
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 1, tkv large enough to prefill w/o tkv shift
            # total blocks in use: 2 + 2
            "step": 8,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 3 + 2 (prefill (2 block) + 49 decodes in the last block)
            "n_reserved_blocks": 5,
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0 and 1
            "step": 9,
            "tkv": 71,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4,  # seq 1 writes into the right pads
        },
        {
            # Sequence 1 finishes at step 57
            # (start step 8 + 1 prefills + 49 decodes - 1) = 8 + 1 + 49 - 1 = 57
            "step": 57,
            "tkv": 119,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4,
        },
        {
            # Decode sequence 0
            # total blocks in use: 4 - 2 = 2
            "step": 58,
            "tkv": 120,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 3,  # 5 - 2 (seq 1)
            "n_used_blocks": 2,
        },
        {
            # Decode sequence 0 needs another block
            # total blocks in use: 2 + 1 = 3
            "step": 67,
            "tkv": 129,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3,
        },
        {
            # Sequence 0 finishes at step 68
            # (start step + 2 prefill + 66 decodes - 1) = 1 + 2 + 66 - 1 = 68
            "step": 68,
            "tkv": 130,
            "waiting": [],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3,
        },
        {
            # Tkv should be cleared one step later
            "step": 69,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
# provide only 4 blocks, to prefill with tkv shift
# at least 5 blocks would be required
@pytest.mark.parametrize("available_blocks", [4])
def test_prefill_use_more_than_available_blocks(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where the requested prompt is too long for current tkv value

    Note that we could prefill the prompt straight away. However,
    in this test the number of available KV cache blocks is decreased
    to a value where the the number of reserved blocks would exceed the number
    of available blocks after the tkv shift, we therefore cannot schedule it.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 0: len = 49, max tokens = 10, step joining = 0
            * 1: len = 70, max tokens = 4, step joining = 0
        * available_blocks: 4
    """

    seqs_max_tokens = [10, 4]
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
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 9 decodes (1 block)
            "n_used_blocks": 1,
        },
        # We cannot schedule sequence 1 here. Prefill with tkv shift moves
        # sequence 0 by 1 block, so it still needs 2 blocks (not counting fully
        # padded blocks!) Aligning sequence 1 would then require 3 blocks. With
        # only 4 blocks available, scheduling sequence 1 is not possible.
        {
            # Decode sequence 0
            # total blocks in use: 1 + 1
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 1, tkv large enough to prefill w/o tkv shift
            # total blocks in use: 2 + 2
            "step": 8,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 2 + 2 (prefill (2 block) + 3 decodes in the last block)
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0 and 1
            "step": 9,
            "tkv": 71,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Sequences 0 and 1 finish at step 11
            # (start step + 2 prefills + 9 decodes - 1) = 1 + 2 + 9 - 1 = 11
            # (start step + 1 prefill + 3 decodes - 1) = 8 + 1 + 3 - 1 = 11
            "step": 11,
            "tkv": 73,
            "waiting": [],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Tkv should be cleared one step later
            "step": 12,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_requested_tokens_not_fitting_remaining_space(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where the request goes beyond max_model_len and needs to wait
    for a new batch.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 0: len = 49, max tokens = 18, step joining = 0
            * 1: len = 41, max tokens = 15, step joining = 0
            * 2: len = 30, max tokens = 55, step joining = 0
    """
    seqs_max_tokens = [18, 15, 55]
    prompts_lengths = [49, 41, 30]
    steps_add_reqs = [0, 0, 0]

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 2
            "step": 1,
            "tkv": 64,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            # prefill (1 block) + 17 decodes (1 block)
            "n_reserved_blocks": 2,
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 2 + 1
            "step": 2,
            "tkv": 64,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # prefill (1 block) + 14 decodes (1 block)
            "n_reserved_blocks": 4,
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 0 and 1
            # total blocks in use: 2 + 2 (decodes)
            "step": 3,
            "tkv": 65,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Sequence 1 finishes at step 16
            # (start step + 1 prefill + 14 decodes - 1) = 2 + 1 + 14 - 1 = 16
            "step": 16,
            "tkv": 78,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Decode sequence 0
            # Cannot prefill sequence 2: 78 + 54 = 132 > 128
            # total blocks in use: 4 - 2 = 2
            "step": 17,
            "tkv": 79,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # 4 - 2 (seq 1)
            "n_used_blocks": 2,
        },
        {
            # Sequence 0 finishes at step 19
            # (start step + 2 prefills + 17 decodes - 1) = 1 + 2 + 17 - 1 = 19
            "step": 19,
            "tkv": 81,
            "waiting": ["2"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 4 + 1 = 1
            "step": 20,
            "tkv": 64,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            # 2 - 2 (seq 0) + 2 (prefill (1 block) + 54 decodes (1 block))
            "n_reserved_blocks": 2,
            "n_used_blocks": 1,
        },
        {
            # Decode sequence 2
            # total blocks in use: 1 + 1 = 2
            "step": 21,
            "tkv": 65,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Sequence 2 finishes at step 64
            # (start step + 1 prefill + 54 decodes - 1) = 20 + 1 + 54 - 1 = 74
            "step": 74,
            "tkv": 118,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2,
        },
        {
            # Tkv should be cleared one step later
            "step": 75,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [4])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("available_blocks", [8])
def test_requests_use_all_available_blocks(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where the requests use all of the available blocks

    Configuration:
        * max_num_seqs: 4
        * number of prompts: 4
            * 0: len = 10, max tokens = 3, step joining = 0
            * 1: len = 10, max tokens = 3, step joining = 0
            * 2: len = 10, max tokens = 3, step joining = 0
            * 3: len = 10, max tokens = 3, step joining = 0
        * available_blocks: 8
    """
    seqs_max_tokens = [3, 3, 3, 3]  # 2 decodes into a new block per sequence
    prompts_lengths = [10, 10, 10, 10]  # 1 block for prefill per sequence
    steps_add_reqs = [0, 0, 0, 0]
    # total number of blocks needed if scheduled together : 4 * (1 + 1) = 8

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2", "3"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1", "2", "3"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 2
            "step": 2,
            "tkv": 64,
            "waiting": ["2", "3"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 2,
        },
        # requests 2 and 3 can be prefilled straight away
        {
            # Prefill sequence 2
            # note: needs two blocks, as crossing block boundary
            # total blocks in use: 3
            "step": 3,
            "tkv": 64,
            "waiting": ["3"],
            "running": ["2", "1", "0"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 6,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 3,
        },
        {
            # Prefill sequence 3
            # note: needs two blocks, as crossing block boundary
            # total blocks in use: 4
            "step": 4,
            "tkv": 64,
            "waiting": [],
            "running": ["3", "2", "1", "0"],
            "request_outputs": ["3"],
            "n_reserved_blocks": 8,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0, 1, 2, 3
            # total blocks in use: 8
            "step": 5,
            "tkv": 65,
            "waiting": [],
            "running": ["3", "2", "1", "0"],
            "request_outputs": ["3", "2", "1", "0"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 8,
        },
        {
            # Decode sequences 0, 1, 2, 3
            # all sequences finish at step 6
            # total blocks in use: 8
            "step": 6,
            "tkv": 66,
            "waiting": [],
            "running": [],
            "request_outputs": ["3", "2", "1", "0"],
            "finished_requests": ["3", "2", "1", "0"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 8,
        },
        {
            # Tkv should be cleared one step later
            # total blocks in use: 8 - 8 = 0
            "step": 7,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [4])
@pytest.mark.parametrize("max_model_len", [128])
@pytest.mark.parametrize("available_blocks", [4])
def test_requests_use_more_than_available_blocks(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where some request need to wait because of the number of
    available blocks.

    Configuration:
        * max_num_seqs: 4
        * number of prompts: 4
            * 0: len = 10, max tokens = 3, step joining = 0
            * 1: len = 10, max tokens = 3, step joining = 0
            * 2: len = 10, max tokens = 3, step joining = 0
            * 3: len = 10, max tokens = 3, step joining = 0
        * available_blocks: 4
    """

    seqs_max_tokens = [3, 3, 3, 3]  # 2 decodes into a new block per sequence
    prompts_lengths = [10, 10, 10, 10]  # 1 block for prefill per sequence
    steps_add_reqs = [0, 0, 0, 0]
    # total number of blocks needed if scheduled together : 4 * (1 + 1) = 8

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2", "3"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
        },
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1", "2", "3"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 2
            "step": 2,
            "tkv": 64,
            "waiting": ["2", "3"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 2,
        },
        # requests 2 and 3 cannot be prefilled as not enough blocks
        # thus decode 0 and 1 until they free the blocks again
        {
            # Decode sequences 0 and 1
            # total blocks in use: 4
            "step": 3,
            "tkv": 65,
            "waiting": ["2", "3"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 0 and 1
            # Sequence 0 and 1 finish at step 4
            # total blocks in use: 4
            "step": 4,
            "tkv": 66,
            "waiting": ["2", "3"],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        # now we have enough blocks to prefill sequence 2 and 3
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 4 + 1 = 1
            "step": 5,
            "tkv": 64,
            "waiting": ["3"],
            "running": ["2"],
            "request_outputs": ["2"],
            # 4 - 4 (seq 0 + 1) + 2 (prefill (1 block) + 3 decodes (1 block))
            "n_reserved_blocks": 2,
            "n_used_blocks": 1,
        },
        {
            # Prefill sequence 3
            # total blocks in use: 1 + 1 = 2
            "step": 6,
            "tkv": 64,
            "waiting": [],
            "running": ["3", "2"],
            "request_outputs": ["3"],
            "n_reserved_blocks": 4,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 2,
        },
        {
            # Decode sequences 2 and 3
            # total blocks in use: 2 + 2 = 4
            "step": 7,
            "tkv": 65,
            "waiting": [],
            "running": ["3", "2"],
            "request_outputs": ["3", "2"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Decode sequences 2 and 3
            # Sequence 2 and 3 finish at step 8
            # total blocks in use: 4
            "step": 8,
            "tkv": 66,
            "waiting": [],
            "running": [],
            "request_outputs": ["3", "2"],
            "finished_requests": ["3", "2"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4,
        },
        {
            # Tkv should be cleared one step later
            # total blocks in use: 4 - 4 = 0
            "step": 9,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [192])
@pytest.mark.parametrize("available_blocks", [None])
def test_requests_use_full_batch_tkv_limit(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where all requests can be scheduled right away as the
    max batch x tkv limit, e.g the volumetric limit, is just high enough.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 64, max tokens = 2, step joining = 0
            * 2: len = 65, max tokens = 2, step joining = 0
    """

    seqs_max_tokens = [2, 2]
    prompts_lengths = [64, 65]
    steps_add_reqs = [0, 0]
    # total number of blocks needed if scheduled together: (1 + 1)+(2 + 1) = 5
    # needs 2 * (64 + 64 + 1) = 2 * 129 = 258
    max_batch_tkv_limit = 258  # just big enough

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
            "n_used_blocks": 1,
        },
        # Note: we can prefill seq 1 here as the volumetric limit
        # max_batch_tkv_limit is just big enough (258)
        # -> cond5 in can_schedule() is True
        {
            # Prefill sequence 1
            # total blocks in use: 3
            "step": 2,
            "tkv": 128,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 5,  # prefill (2 block) + 1 decode (1 block)
            "n_used_blocks": 3,  # 1 + 2
        },
        {
            # Decode sequences 0 and 1
            # Sequence 0 and 1 finish at step 3
            # total blocks in use: 5
            "step": 3,
            "tkv": 129,
            "waiting": [],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 5,
        },
        {
            # Tkv should be cleared one step later
            # total blocks in use: 5 - 5 = 0
            "step": 4,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.full_model
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [192])
@pytest.mark.parametrize("available_blocks", [None])
def test_requests_exceed_batch_tkv_limit(
    model: ModelInfo,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    set_random_seed,
    max_num_seqs: int,
    max_model_len: int,
    available_blocks: int,
):
    """Scenario where a request cannot be scheduled right away as the
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
    # total number of blocks needed if scheduled together: (1 + 1)+(2 + 1) = 5
    # note that as not scheduled together, we only needs 3 blocks here
    # needs 2 * (64 + 64 + 1) = 2 * 129 = 258
    max_batch_tkv_limit = 257  # not big enough

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
            "n_used_blocks": 1,
        },
        # Note: we cannot prefill seq 1 as the volumetric limit
        # max_batch_tkv_limit is exceeded: 257 < 258
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
            "n_used_blocks": 2,
        },
        {
            # Prefill sequence 1
            # total blocks in use: 2
            "step": 3,
            "tkv": 128,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 3,  # prefill (2 block) + 1 decode (1 block)
            "n_used_blocks": 2,  # 2 - 2 + 2
        },
        {
            # Decode sequence 1
            # Sequence 1 finishes at step 4
            # total blocks in use: 3
            "step": 4,
            "tkv": 129,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3,
        },
        {
            # Tkv should be cleared one step later
            # total blocks in use: 3 - 3 = 0
            "step": 5,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0,
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
        use_cb=True,
    )

"""Verification of the correctness of the step-by-step execution of continuous 
batching. It does so by comparing, at every engine step (i.e. prefill or decode 
iteration), a bunch of attributes. This allows a finer testing of the padding 
and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_cb_inference_steps.py`.
"""

import pytest
from scheduling_utils import check_scheduler_inference_steps
from spyre_util import get_spyre_backend_list, get_spyre_model_list


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prompts_aligned_with_tkv_boundaries(model: str, backend: str,
                                             monkeypatch: pytest.MonkeyPatch):
    """ Scenario where it happens that all the sequences get scheduled in a 
    fashion where they are aligned with the block boundaries (i.e. tkv multiple 
    of 64 at the time of prefilling).
    
    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 1: len = 49, max tokens = 65, step joining = 0
            * 2: len = 41, max tokens = 67, step joining = 0
            * 3: len = 47, max tokens = 7, step joining = 0
    """

    seqs_max_tokens = [65, 67, 7]
    prompts_lengths = [49, 41, 47]
    steps_add_reqs = [0, 0, 0]  # add all requests in the beginning
    available_blocks = -1  # no restriction
    max_num_seqs = 2

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
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
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 64 decodes (1 block)
            "n_used_blocks": 1
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
            "n_used_blocks": 2
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
            "n_used_blocks": 4
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
            "n_used_blocks": 4
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 2 + 2 = 4
            "step": 67,
            "tkv": 128,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"],
            # 5 - 2 (seq 0) + 3 (prefill (2 blocks) + decodes (1 block))
            "n_reserved_blocks": 6,
            "n_used_blocks": 4
        },
        {
            # Decode sequences 1 and 2
            # total blocks in use: 4 + 2 = 6
            "step": 68,
            "tkv": 129,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2", "1"],
            "n_reserved_blocks": 6,
            "n_used_blocks": 6
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
            "n_reserved_blocks": 6,
            "n_used_blocks": 6
        },
        {
            # Decode sequence 2
            # total blocks in use: 6 - 3 - 1 (remove padded block) = 2
            "step": 70,
            "tkv": 67,  # tkv is reset by 64 due to removing the padded block
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            # 6 - 3 (seq 1 left) - 1 (removing the padded block)
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Sequence 2 finishes at step 73
            # (start step + 1 prefill + 6 decodes - 1) = 67 + 1 + 6 - 1 = 73
            "step": 73,
            "tkv": 70,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Tkv should be cleared one step later
            "step": 74,
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
        available_blocks=available_blocks,
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prompts_misaligned_with_tkv_boundaries(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch):
    """ Scenario where it happens that some sequence gets scheduled in a way 
    that it is misaligned with the block boundary (i.e. tkv is not a multiple 
    of 64 at the time of prefilling).
    
    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 1: len = 49, max tokens = 57, step joining = 0
            * 2: len = 41, max tokens = 67, step joining = 0
            * 3: len = 47, max tokens = 9, step joining = 0
    """

    seqs_max_tokens = [57, 67, 9]
    prompts_lengths = [49, 41, 47]
    steps_add_reqs = [0, 0, 0]  # add all requests in the beginning
    available_blocks = -1  # no restriction
    max_num_seqs = 2

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
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
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 56 decodes (1 block)
            "n_used_blocks": 1
        },
        {
            # Prefill sequence 1
            # total blocks in use: 1 + 1 = 2
            "step": 2,
            "tkv": 64,  # Still 64 because this step is also a prefill
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # prefill (1 block) + 66 decodes (2 blocks)
            "n_reserved_blocks": 5,
            "n_used_blocks": 2
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
            "n_used_blocks": 4
        },
        {
            # Sequence 0 finishes at step 58
            # (start step + 2 prefills + 56 decodes - 1) = 1 + 2 + 56 - 1 = 58
            "step": 58,
            "tkv": 120,
            "waiting": ["2"],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 2 + 2 = 4
            "step": 59,
            "tkv": 120,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"],
            # 5 - 2 (seq 0) + 2 (prefill (2 block) + 8 decodes in 2nd block)
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
        },
        {
            # Decode sequences 1 and 2
            "step": 60,
            "tkv": 121,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2", "1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
        },
        {
            # Sequence 2 finishes at step 68
            # (start step + 1 prefill + 8 decodes - 1) = 59 + 1 + 8 - 1 = 67
            "step": 67,
            "tkv": 128,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["2", "1"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
        },
        {
            # Decode sequences 1
            # total blocks in use: 4 - 2 + 1 = 3
            "step": 68,
            "tkv": 129,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 3,  # 5 - 2 (seq 2)
            "n_used_blocks": 3
        },
        {
            # Sequence 1 finishes at step 69
            # (start step + 2 prefills + 66 decodes - 1) = 2 + 2 + 66 - 1 = 69
            "step": 69,
            "tkv": 130,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3
        },
        {
            # Tkv should be cleared one step later
            "step": 70,
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
        available_blocks=available_blocks,
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_two_sequences_finish_same_time_as_new_arrive(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch):
    """ 2-cases-in-1: (1) Two sequences finish at the same time and (2) a new
    request arrives when another finishes.

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 1: len = 49, max tokens = 30, step joining = 0
            * 2: len = 30, max tokens = 30, step joining = 0
            * 3: len = 20, max tokens = 10, step joining = 31
    """

    seqs_max_tokens = [30, 30, 10]
    prompts_lengths = [49, 30, 20]
    steps_add_reqs = [0, 0, 31]
    available_blocks = -1  # no restriction
    max_num_seqs = 2

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
            "n_reserved_blocks": 2,  # prefill (1 block) + 29 decodes (1 block)
            "n_used_blocks": 1
        },
        {
            # Prefill sequence 1
            # total blocks in use: 1 + 1 = 2
            "step": 2,
            "tkv": 64,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,  # prefill (1 block) + 29 decodes (1 block)
            "n_used_blocks": 2
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
            "n_used_blocks": 4
        },
        {
            # Sequences 0 and 1 finish at step 31
            # (start step + 2 prefills + 29 decodes - 1) = 1 + 2 + 29 - 1 = 31
            "step": 31,
            "tkv": 93,
            "waiting": ["2"],
            "running": [],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1", "0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 4 + 1
            "step": 32,
            "tkv": 64,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            # 4 - 4 + 2 (prefill (1 block) + 9 decodes (1 block))
            "n_reserved_blocks": 2,
            "n_used_blocks": 1
        },
        {
            # Decode sequence 2
            # total blocks in use: 1 + 1
            "step": 33,
            "tkv": 65,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Sequences 2 finishes at step 41
            # (start step + 1 prefill + 29 decodes - 1) = 32 + 1 + 9 - 1 = 41
            "step": 41,
            "tkv": 73,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Tkv should be cleared one step later
            "step": 42,
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
        available_blocks=available_blocks,
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prompt_too_long_for_current_tkv(model: str, backend: str,
                                         monkeypatch: pytest.MonkeyPatch):
    """ Scenario where the requested prompt is too long for current tkv value

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 49, max tokens = 57, step joining = 0
            * 2: len = 70, max tokens = 67, step joining = 0
    """

    seqs_max_tokens = [57, 67]
    prompts_lengths = [49, 70]
    steps_add_reqs = [0, 0]
    available_blocks = -1  # no restriction
    max_num_seqs = 2

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
            "n_reserved_blocks": 2,  # prefill (1 block) + 56 decodes (1 block)
            "n_used_blocks": 1
        },
        {
            # Decode sequence 0
            # total blocks in use: 1 + 1
            # Cannot prefill sequence 1, because of tkv constraint
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Prefill sequence 1, tkv large enough
            # total blocks in use: 2 + 2
            "step": 8,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 2 + 3 (prefill (2 block) + 66 decodes (1 block))
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
        },
        {
            # Decode sequences 0 and 1
            "step": 9,
            "tkv": 71,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4  # seq 1 writes into the right pads
        },
        {
            # Sequence 0 finishes at step 58
            # (start step + 2 prefills + 56 decodes - 1) = 1 + 2 + 56 - 1 = 58
            "step": 58,
            "tkv": 120,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
        },
        {
            # Decode sequence 1
            # total blocks in use: 4 - 2 = 2
            "step": 59,
            "tkv": 121,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 3,  # 5 - 2 (seq 0)
            "n_used_blocks": 2
        },
        {
            # Decode sequence 1 needs another block
            # total blocks in use: 2 + 1 = 3
            "step": 67,
            "tkv": 129,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3
        },
        {
            # Sequence 1 finishes at step 74
            # (start step + 1 prefill + 66 decodes - 1) = 8 + 1 + 66 - 1 = 74
            "step": 74,
            "tkv": 136,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3
        },
        {
            # Tkv should be cleared one step later
            "step": 75,
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
        available_blocks=available_blocks,
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_requested_tokens_not_fitting_remaining_space(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch):
    """ Scenario where the request goes beyond max_model_len 

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 3
            * 1: len = 70, max tokens = 67, step joining = 0
            * 2: len = 49, max tokens = 57, step joining = 0
            * 3: len = 41, max tokens = 80, step joining = 0
    """

    seqs_max_tokens = [67, 57, 80]
    prompts_lengths = [70, 49, 41]
    steps_add_reqs = [0, 0, 0]
    available_blocks = -1  # no restriction
    max_num_seqs = 2

    checked_steps = [
        {
            "step": 0,
            "tkv": 0,
            "waiting": ["0", "1", "2"],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        {
            # Prefill sequence 0
            # total blocks in use: 2
            "step": 1,
            "tkv": 128,
            "waiting": ["1", "2"],
            "running": ["0"],
            "request_outputs": ["0"],
            # prefill (2 blocks) + 66 decodes (2 blocks)
            "n_reserved_blocks": 4,
            "n_used_blocks": 2
        },
        {
            # Prefill sequence 1
            # total blocks in use: 2 + 2
            "step": 2,
            "tkv": 128,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # prefill (2 blocks) + 56 decodes (1 block)
            "n_reserved_blocks": 7,
            "n_used_blocks": 4
        },
        {
            # Decode sequences 0 and 1
            # total blocks in use: 4 + 2 (decodes)
            "step": 3,
            "tkv": 129,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6
        },
        {
            # Sequence 1 finishes at step 58
            # (start step + 1 prefills + 56 decodes - 1) = 2 + 1 + 56 - 1 = 58
            "step": 58,
            "tkv": 184,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6
        },
        {
            # Decode sequence 0
            # Cannot prefill sequence 2: 185 + 80 = 265 > 256
            # total blocks in use: 6 - 3 = 3
            "step": 59,
            "tkv": 185,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,  # 7 - 3 (seq 1)
            "n_used_blocks": 3
        },
        {
            # Decode sequence 0 needs another block for decoding
            # total blocks in use: 3 + 1 = 4
            "step": 67,
            "tkv": 193,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {
            # Sequence 0 finishes at step 68
            # (start step + 2 prefills + 66 decodes - 1) = 1 + 2 + 66 - 1 = 68
            "step": 68,
            "tkv": 194,
            "waiting": ["2"],
            "running": [],
            "request_outputs": ["0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {
            # Prefill sequence 2
            # total blocks in use: 4 - 4 + 1 = 1
            "step": 69,
            "tkv": 64,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            # 4 - 4 (seq 0) + 3 (prefill (1 blocks) + 79 decodes (2 blocks))
            "n_reserved_blocks": 3,
            "n_used_blocks": 1
        },
        {
            # Decode sequence 2
            # total blocks in use: 1 + 1 = 2
            "step": 70,
            "tkv": 65,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 2
        },
        {
            # Decode sequence 2 needs another block
            # total blocks in use: 2 + 1 = 3
            "step": 134,
            "tkv": 129,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3
        },
        {
            # Sequence 2 finishes at step 148
            # (start step + 1 prefill + 79 decodes - 1) = 69 + 1 + 79 - 1 = 148
            "step": 148,
            "tkv": 143,
            "waiting": [],
            "running": [],
            "request_outputs": ["2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 3
        },
        {
            # Tkv should be cleared one step later
            "step": 149,
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
        available_blocks=available_blocks,
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_requests_use_all_available_blocks(model: str, backend: str,
                                           monkeypatch: pytest.MonkeyPatch):
    """ Scenario where the requests use all of the available blocks 
    
    Configuration:
        * max_num_seqs: 4
        * number of prompts: 4
            * 1: len = 10, max tokens = 3, step joining = 0
            * 2: len = 10, max tokens = 3, step joining = 0
            * 3: len = 10, max tokens = 3, step joining = 0
            * 4: len = 10, max tokens = 3, step joining = 0
        * available_blocks: 8
    """

    seqs_max_tokens = [3, 3, 3, 3]  # 2 decodes into a new block per sequence
    prompts_lengths = [10, 10, 10, 10]  # 1 block for prefil per sequence
    steps_add_reqs = [0, 0, 0, 0]
    # total number of blocks needed if scheduled together : 4 * (1 + 1) = 8
    available_blocks = 8
    max_num_seqs = 4
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
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1", "2", "3"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 1
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
            "n_used_blocks": 2
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
            "n_used_blocks": 3
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
            "n_used_blocks": 4
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
            "n_used_blocks": 8
        },
        {
            # Decode sequences 0, 1, 2, 3
            # all sequences finish at step 8
            # total blocks in use: 8
            "step": 6,
            "tkv": 66,
            "waiting": [],
            "running": [],
            "request_outputs": ["3", "2", "1", "0"],
            "finished_requests": ["3", "2", "1", "0"],
            "n_reserved_blocks": 8,
            "n_used_blocks": 8
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
        available_blocks=available_blocks,
        use_cb=True,
    )


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_requests_use_more_than_available_blocks(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch):
    """ Scenario where some request need to wait because of the number of 
    available blocks. 
    
    Configuration:
        * max_num_seqs: 4
        * number of prompts: 4
            * 1: len = 10, max tokens = 3, step joining = 0
            * 2: len = 10, max tokens = 3, step joining = 0
            * 3: len = 10, max tokens = 3, step joining = 0
            * 4: len = 10, max tokens = 3, step joining = 0
        * available_blocks: 8
    """

    seqs_max_tokens = [3, 3, 3, 3]  # 2 decodes into a new block per sequence
    prompts_lengths = [10, 10, 10, 10]  # 1 block for prefil per sequence
    steps_add_reqs = [0, 0, 0, 0]
    # total number of blocks needed if scheduled together : 4 * (1 + 1) = 8
    available_blocks = 4
    max_num_seqs = 4
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
        {
            # Prefill sequence 0
            # total blocks in use: 1
            "step": 1,
            "tkv": 64,
            "waiting": ["1", "2", "3"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 2,  # prefill (1 block) + 3 decodes (1 block)
            "n_used_blocks": 1
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
            "n_used_blocks": 2
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
            "n_used_blocks": 4
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
            "n_used_blocks": 4
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
            "n_used_blocks": 1
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
            "n_used_blocks": 2
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
            "n_used_blocks": 4
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
            "n_used_blocks": 4
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
        available_blocks=available_blocks,
        use_cb=True,
    )

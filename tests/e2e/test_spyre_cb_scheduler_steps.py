"""Verification of the correctness of the step-by-step execution of continuous 
batching. It does so by comparing, at every engine step (i.e. prefill or decode 
iteration), a bunch of attributes. This allows a finer testing of the padding 
and scheduling implementation.

Run `python -m pytest tests/e2e/test_spyre_cb_inference_steps.py`.
"""

import pytest
from scheduling_utils import check_scheduler_inference_steps
from spyre_util import (check_output_against_hf, get_spyre_backend_list,
                        get_spyre_model_list)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prompts_aligned_with_tkv_boundaries(model: str, backend: str,
                                             monkeypatch: pytest.MonkeyPatch,
                                             set_random_seed: None):
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
    available_blocks = 1000  # no restriction
    max_num_seqs = 2
    max_model_len = 256

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
            # total blocks in use: 4 - 2 + 1 = 3
            "step": 67,
            "tkv": 128,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"],
            # 5 - 2 (seq 0) + 2 (prefill (1 block) + decodes (1 block))
            "n_reserved_blocks": 5,
            "n_used_blocks": 3
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
            "n_used_blocks": 5
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
            "n_used_blocks": 5
        },
        {
            # Decode sequence 2
            # total blocks in use: 5 - 3  = 2
            "step": 70,
            "tkv": 67,  # tkv is reset by 64 due to removing the padded block
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            # 5 - 3 (seq 1 left)
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

    cb_outputs, prompts = check_scheduler_inference_steps(
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

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prompts_misaligned_with_tkv_boundaries(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch,
        set_random_seed: None):
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
    available_blocks = 1000  # no restriction
    max_num_seqs = 2
    max_model_len = 256

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
            # total blocks in use: 4 - 2 + 1 = 3
            "step": 59,
            "tkv": 120,  # Tkv doesn't increase because it is a prefill
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2"],
            # 5 - 2 (seq 0) + 1 (prefill (1 block) + 8 decodes in 1st block)
            "n_reserved_blocks": 4,
            "n_used_blocks": 3
        },
        {
            # Decode sequences 1 and 2
            "step": 60,
            "tkv": 121,
            "waiting": [],
            "running": ["2", "1"],
            "request_outputs": ["2", "1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3
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
            "n_reserved_blocks": 4,
            "n_used_blocks": 3
        },
        {
            # Decode sequences 1
            # total blocks in use: 3 - 1 + 1 = 3
            "step": 68,
            "tkv": 129,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 3,  # 4 - 1 (seq 2)
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

    cb_outputs, prompts = check_scheduler_inference_steps(
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

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_two_sequences_finish_same_time_as_new_arrive(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch,
        set_random_seed):
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
    available_blocks = 1000  # no restriction
    max_num_seqs = 2
    max_model_len = 256

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

    cb_outputs, prompts = check_scheduler_inference_steps(
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

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_new_sequence_joins_during_decode(model: str, backend: str,
                                          monkeypatch: pytest.MonkeyPatch,
                                          set_random_seed):
    """ Scenario where a new sequence joins while decoding other sequences

    Configuration:
        * max_num_seqs: 4
        * number of prompts: 4
            * 1: len = 49, max tokens = 119, step joining = 0
            * 2: len = 14, max tokens = 52, step joining = 0
            * 3: len = 89, max tokens = 104, step joining = 32
            * 4: len = 9, max tokens = 64, step joining = 131
    """
    # TODO change to 65 max_tokens for last prompt if ever possible

    seqs_max_tokens = [119, 52, 104, 64]
    prompts_lengths = [49, 14, 89, 9]
    steps_add_reqs = [0, 0, 32, 131]
    available_blocks = 1000  # no restriction
    max_num_seqs = 4
    max_model_len = 256

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
            "step": 1,
            "tkv": 64,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 3,  # prefill (1 block) + 119 decode (2 block)
            "n_used_blocks": 1
        },
        {
            # Prefill sequence 1
            "step": 2,
            "tkv": 64,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 5,  # prefill (1 block) + 51 decodes (1 block)
            "n_used_blocks": 2
        },
        {
            # Decode sequences 0 and 1
            "step": 3,
            "tkv": 65,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4  # 2 blocks extended, one for each sequence
        },
        {
            # Sequence 2 joins: one iteration in waiting queue
            "step": 32,
            "tkv": 94,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
        },
        {
            # Prefill sequence 2
            "step": 33,
            "tkv": 94,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 9,  # prefill (2 block) + 103 decode (2 block)
            "n_used_blocks": 6
        },
        {
            # Decode sequences 0, 1, and 2
            "step": 34,
            "tkv": 95,
            "waiting": [],
            "running": ["2", "1", "0"],
            "request_outputs": ["2", "1", "0"],
            "n_reserved_blocks": 9,
            "n_used_blocks": 6
        },
        {
            # Sequence 1 finishes at step 54
            # (start step + 2 prefills + 51 decodes - 1) = 2 + 2 + 51 - 1 = 54
            "step": 54,
            "tkv": 115,
            "waiting": [],
            "running": ["2", "0"],
            "request_outputs": ["2", "1", "0"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 9,
            "n_used_blocks": 6
        },
        {
            # Decode sequences 0 and 2
            "step": 55,
            "tkv": 116,
            "waiting": [],
            "running": ["2", "0"],
            "request_outputs": ["2", "0"],
            "n_reserved_blocks": 7,  # two blocks released
            "n_used_blocks": 4  # two blocks released
        },
        {
            # Decode sequences 0 and 2, tkv arrives to new block
            "step": 68,
            "tkv": 129,
            "waiting": [],
            "running": ["2", "0"],
            "request_outputs": ["2", "0"],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6  # 2 blocks extended, one for each sequence
        },
        {
            # Sequence 0 finishes at step 121
            # (start step + 3 prefills + 118 decode - 1) = 1 + 3 + 118 - 1 = 121
            "step": 121,
            "tkv": 182,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 7,
            "n_used_blocks": 6
        },
        {
            # Decode sequence 2
            "step": 122,
            "tkv": 183,
            "waiting": [],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 4,  # 3 blocks released
            "n_used_blocks": 3  # 3 blocks released
        },
        {
            # Sequence 3 joins: one iteration in waiting queue
            "step": 131,
            "tkv": 192,
            "waiting": ["3"],
            "running": ["2"],
            "request_outputs": ["2"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 3
        },
        {
            # Prefill sequence 3
            "step": 132,
            "tkv": 192,
            "waiting": [],
            "running": ["3", "2"],
            "request_outputs": ["3"],
            # Note: here is where the optimization happens: we do the prefill
            # on a single block only instead of using 3 blocks
            "n_reserved_blocks": 6,  # prefill (1 block) + 63 decode (1 block)
            "n_used_blocks": 4  # prefill (1 block)
        },
        {
            # Decode sequences 2 and 3
            "step": 133,
            "tkv": 193,
            "waiting": [],
            "running": ["3", "2"],
            "request_outputs": ["3", "2"],
            "n_reserved_blocks": 6,
            "n_used_blocks": 6  # 2 blocks extended, one for each sequence
        },
        {
            # Sequence 2 finishes at step 137
            # (start step + 2 prefills + 103 decodes) = 33 + 2 + 103 - 1 = 137
            "step": 137,
            "tkv": 197,
            "waiting": [],
            "running": ["3"],
            "request_outputs": ["3", "2"],
            "finished_requests": ["2"],
            "n_reserved_blocks": 6,
            "n_used_blocks": 6
        },
        {
            # Decode sequence 3
            "step": 138,
            "tkv": 70,
            "waiting": [],
            "running": ["3"],
            "request_outputs": ["3"],
            # 4 blocks freed due to finished sequence 2
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Sequence 3 finishes at step 196
            # (start step + 1 prefills + 103 decodes) = 132 + 1 + 63 - 1 = 196
            "step": 195,
            "tkv": 127,
            "waiting": [],
            "running": [],
            "request_outputs": ["3"],
            "finished_requests": ["3"],
            "n_reserved_blocks": 2,
            "n_used_blocks": 2
        },
        {
            # Tkv should be cleared one step later
            "step": 196,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
        # TODO this is when max_tokens = 65 for last prompt
        # {
        #     # Sequence 3 finishes at step 196
        #     # (start step + 1 prefills + 103 decodes) = 132 + 1 + 64 - 1 = 196
        #     "step": 196,
        #     "tkv": 128,
        #     "waiting": [],
        #     "running": [],
        #     "request_outputs": ["3"],
        #     "finished_requests": ["3"],
        #     "n_reserved_blocks": 2,
        #     "n_used_blocks": 2
        # },
        # {
        #     # Tkv should be cleared one step later
        #     "step": 197,
        #     "tkv": 0,
        #     "waiting": [],
        #     "running": [],
        #     "request_outputs": [],
        #     "n_reserved_blocks": 0,
        #     "n_used_blocks": 0
        # },
    ]

    cb_outputs, prompts = check_scheduler_inference_steps(
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

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("prefill_optimization", [True, False])
def test_prompt_too_long_for_current_tkv(model: str, backend: str,
                                         prefill_optimization: bool,
                                         monkeypatch: pytest.MonkeyPatch,
                                         set_random_seed):
    """ Scenario where the requested prompt is too long for current tkv value
   
    Note that with VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION enabled, we can 
    prefill the prompt straight away -> using checked_steps_with_optimization

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 49, max tokens = 57, step joining = 0
            * 2: len = 70, max tokens = 67, step joining = 0
    """

    if prefill_optimization:
        monkeypatch.setenv('VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION', '1')

    seqs_max_tokens = [57, 67]
    prompts_lengths = [49, 70]
    steps_add_reqs = [0, 0]
    available_blocks = 1000  # no restriction
    max_num_seqs = 2
    max_model_len = 256

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

    checked_steps_with_optimization = [
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
        # due to allowing sequences to join the current decode batch even if
        # prompt length > tkv, prefill of sequence 1 happens immediately
        {
            # Prefill sequence 1
            # total blocks in use: 2 + 2
            "step": 2,
            "tkv": 128,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 2 + 4 (prefill (2 block) + 66 decodes (2 blocks))
            "n_reserved_blocks": 6,
            "n_used_blocks": 3
        },
        {
            # Decode sequences 0 and 1
            "step": 3,
            "tkv": 129,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 6,
            "n_used_blocks": 5  # 3 + 2 = 5
        },
        {
            # Sequence 0 finishes at step 58
            # (start step + 2 prefills + 56 decodes - 1) = 1 + 2 + 56 - 1 = 58
            "step": 58,
            "tkv": 184,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["0"],
            "n_reserved_blocks": 6,
            "n_used_blocks": 5
        },
        {
            # Decode sequence 1
            # total blocks in use: 5 - 2 = 3
            "step": 59,
            "tkv": 185,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,  # 6 - 2 (seq 0)
            "n_used_blocks": 3
        },
        {
            # Decode sequence 1 needs another block
            # total blocks in use: 3 + 1 = 4
            "step": 67,
            "tkv": 193,
            "waiting": [],
            "running": ["1"],
            "request_outputs": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {
            # Sequence 1 finishes at step 74
            # (start step + 1 prefill + 66 decodes - 1) = 2 + 1 + 66 - 1 = 68
            "step": 68,
            "tkv": 194,
            "waiting": [],
            "running": [],
            "request_outputs": ["1"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 4,
            "n_used_blocks": 4
        },
        {
            # Tkv should be cleared one step later
            "step": 69,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    cb_outputs, prompts = check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps_with_optimization
        if prefill_optimization else checked_steps,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=True,
    )

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prefill_optimization_tkv_too_big(model: str, backend: str,
                                          monkeypatch: pytest.MonkeyPatch,
                                          set_random_seed):
    """ Scenario where the requested prompt is too long for current tkv value
   
    Note that as VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION is enabled, we could 
    prefill the prompt straight away -> using checked_steps_with_optimization

    However, in this test the max model length is decreased to a value where
    the tkv of the decode batch would be shifted beyond the max model length, 
    we therefore have to wait with scheduling it via the prefill optimization. 
    -> see cond4_updated in vllm_spyre/v1/core/scheduler.py

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 49, max tokens = 67, step joining = 0
            * 2: len = 70, max tokens = 50, step joining = 0
    """

    monkeypatch.setenv('VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION', '1')

    seqs_max_tokens = [67, 50]
    prompts_lengths = [49, 70]
    steps_add_reqs = [0, 0]
    available_blocks = 1000  # no restriction
    max_num_seqs = 2
    # restricting the max model length here to trigger the violated
    # scheduler condition
    max_model_len = 192

    checked_steps_with_optimization = [
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
            "n_reserved_blocks":
            3,  # prefill (1 block) + 66 decodes (2 blocks)
            "n_used_blocks": 1
        },
        # Here we cannot schedule sequence 1. By shifting sequence 0 by 1 block
        # due to the prefill optimization, its max tkv would exceed the max
        # model length: 64 + 67 - 1 + 64 (shift) = 194 > 192 (max model length)
        {
            # Decode sequence 0
            # total blocks in use: 1 + 1
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 2
        },
        {
            # Prefill sequence 1, tkv large enough to prefill w/o optimization
            # total blocks in use: 2 + 2
            "step": 8,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 3 + 2 (prefill (2 block) + 49 decodes in the last block)
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
            # Sequence 1 finishes at step 57
            # (start step 8 + 1 prefills + 49 decodes - 1) = 8 + 1 + 49 - 1 = 57
            "step": 57,
            "tkv": 119,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
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
            "n_used_blocks": 2
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
            "n_used_blocks": 3
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
            "n_used_blocks": 3
        },
        {
            # Tkv should be cleared one step later
            "step": 69,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    cb_outputs, prompts = check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps_with_optimization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=True,
    )

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_prefill_optimization_use_more_than_available_blocks(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch,
        set_random_seed):
    """ Scenario where the requested prompt is too long for current tkv value
   
    Note that as VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION is enabled, we could 
    prefill the prompt straight away -> using checked_steps_with_optimization

    However, in this test the number of available KV cache blocks is decreased
    to a value where the the number of reserved blocks would exceed the number
    of available blocks, we therefore have to wait with scheduling it via the 
    prefill optimization. 
    -> see cond5_updated in vllm_spyre/v1/core/scheduler.py

    Configuration:
        * max_num_seqs: 2
        * number of prompts: 2
            * 1: len = 49, max tokens = 67, step joining = 0
            * 2: len = 70, max tokens = 50, step joining = 0
    """

    monkeypatch.setenv('VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION', '1')

    seqs_max_tokens = [67, 50]
    prompts_lengths = [49, 70]
    steps_add_reqs = [0, 0]
    # provide only 5 blocks, to use the prefill optimization
    # at least 6 blocks would be required
    available_blocks = 5
    max_num_seqs = 2
    max_model_len = 256

    checked_steps_with_optimization = [
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
            "n_reserved_blocks":
            3,  # prefill (1 block) + 66 decodes (2 blocks)
            "n_used_blocks": 1
        },
        # We cannot schedule sequence 1 here. Prefill optimization shifts
        # sequence 0 by 1 block, so it still needs 3 blocks (not counting
        # fully padded blocks!) Aligning sequence 1 would then require
        # 3 blocks (instead of 2). With only 5 blocks available, scheduling
        # sequence 1 is not possible.
        {
            # Decode sequence 0
            # total blocks in use: 1 + 1
            "step": 2,
            "tkv": 65,
            "waiting": ["1"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 3,
            "n_used_blocks": 2
        },
        {
            # Prefill sequence 1, tkv large enough to prefill w/o optimization
            # total blocks in use: 2 + 2
            "step": 8,
            "tkv": 70,
            "waiting": [],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # 3 + 2 (prefill (2 block) + 49 decodes in the last block)
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
            # Sequence 1 finishes at step 57
            # (start step 8 + 1 prefills + 49 decodes - 1) = 8 + 1 + 49 - 1 = 57
            "step": 57,
            "tkv": 119,
            "waiting": [],
            "running": ["0"],
            "request_outputs": ["1", "0"],
            "finished_requests": ["1"],
            "n_reserved_blocks": 5,
            "n_used_blocks": 4
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
            "n_used_blocks": 2
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
            "n_used_blocks": 3
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
            "n_used_blocks": 3
        },
        {
            # Tkv should be cleared one step later
            "step": 69,
            "tkv": 0,
            "waiting": [],
            "running": [],
            "request_outputs": [],
            "n_reserved_blocks": 0,
            "n_used_blocks": 0
        },
    ]

    cb_outputs, prompts = check_scheduler_inference_steps(
        model=model,
        backend=backend,
        monkeypatch=monkeypatch,
        seqs_max_tokens=seqs_max_tokens,
        prompts_lengths=prompts_lengths,
        steps_add_reqs=steps_add_reqs,
        checked_steps=checked_steps_with_optimization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        available_blocks=available_blocks,
        use_cb=True,
    )

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_requested_tokens_not_fitting_remaining_space(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch,
        set_random_seed):
    """ Scenario where the request goes beyond max_model_len and needs to wait
    for a new batch.

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
    available_blocks = 1000  # no restriction
    max_num_seqs = 2
    max_model_len = 256

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
            # total blocks in use: 2 + 1
            "step": 2,
            "tkv": 128,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1"],
            # prefill (1 block) + 56 decodes (1 block)
            "n_reserved_blocks": 6,
            "n_used_blocks": 3
        },
        {
            # Decode sequences 0 and 1
            # total blocks in use: 3 + 2 (decodes)
            "step": 3,
            "tkv": 129,
            "waiting": ["2"],
            "running": ["1", "0"],
            "request_outputs": ["1", "0"],
            "n_reserved_blocks": 6,
            "n_used_blocks": 5
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
            "n_reserved_blocks": 6,
            "n_used_blocks": 5
        },
        {
            # Decode sequence 0
            # Cannot prefill sequence 2: 185 + 80 = 265 > 256
            # total blocks in use: 5 - 2 = 3
            "step": 59,
            "tkv": 185,
            "waiting": ["2"],
            "running": ["0"],
            "request_outputs": ["0"],
            "n_reserved_blocks": 4,  # 6 - 2 (seq 1)
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

    cb_outputs, prompts = check_scheduler_inference_steps(
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

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_requests_use_all_available_blocks(model: str, backend: str,
                                           monkeypatch: pytest.MonkeyPatch,
                                           set_random_seed):
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
    max_model_len = 256

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

    cb_outputs, prompts = check_scheduler_inference_steps(
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

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
def test_requests_use_more_than_available_blocks(
        model: str, backend: str, monkeypatch: pytest.MonkeyPatch,
        set_random_seed):
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
    max_model_len = 256

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

    cb_outputs, prompts = check_scheduler_inference_steps(
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

    check_output_against_hf(model, backend, seqs_max_tokens, cb_outputs,
                            prompts)

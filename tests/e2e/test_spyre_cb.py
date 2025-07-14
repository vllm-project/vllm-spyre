"""Verification of continuous batching

Run `python -m pytest tests/e2e/test_spyre_cb.py`.
"""

import copy
from collections import deque
from typing import Any

import pytest
from spyre_util import (create_random_request, generate_spyre_vllm_output,
                        get_spyre_backend_list, get_spyre_model_list)
from vllm import EngineArgs, SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.executor.abstract import Executor

from vllm_spyre.v1.core.scheduler import ContinuousBatchingSpyreScheduler


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
def test_cb_max_tokens(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that continuous batches of requests that
    are longer than the max_model_len are correctly rejected"""

    max_model_len = 256
    max_tokens = 20

    overflow_prompt = " ".join(["a"] * max_model_len)

    vllm_sampling_params = SamplingParams(max_tokens=max_tokens,
                                          temperature=0,
                                          ignore_eos=True,
                                          logprobs=0)

    with pytest.raises(ValueError, match="max model context length"):
        generate_spyre_vllm_output(model=model,
                                   prompts=overflow_prompt,
                                   max_model_len=max_model_len,
                                   block_size=max_model_len,
                                   sampling_params=vllm_sampling_params,
                                   tensor_parallel_size=1,
                                   backend=backend,
                                   max_num_seqs=2,
                                   use_cb=True,
                                   monkeypatch=monkeypatch)


def get_params_test_blocks_borders_aligned_prompts():
    """ Scenario where it happens that all the sequences get scheduled in a 
    fashion where they are aligned with the block boundaries (i.e. tkv multiple 
    of 64 at the time of prefilling)."""

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

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            max_num_seqs, available_blocks)


def get_params_test_blocks_borders_misaligned_prompts():
    """ Scenario where it happens that some sequence gets scheduled in a way 
    that it is misaligned with the block boundary (i.e. tkv is not a multiple 
    of 64 at the time of prefilling). """

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

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            max_num_seqs, available_blocks)


def get_params_test_special_finish():
    """ 2-cases-in-1: (1) Two sequences finish at the same time and (2) a new
    request arrives when another finishes. """

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

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            max_num_seqs, available_blocks)


def get_params_test_scheduler_constraints_tkv():
    """ Scenario where the requested prompt is too long for current tkv value"""

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

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            max_num_seqs, available_blocks)


def get_params_test_scheduler_constraints_max_prompt_len():
    """ Scenario where the request goes beyond max_model_len """

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

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            max_num_seqs, available_blocks)


def get_params_test_scheduler_constraints_max_available_blocks():
    """ Scenario where the requests use all of the available blocks """

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

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            max_num_seqs, available_blocks)


def get_params_test_scheduler_constraints_more_than_available_blocks():
    """ Scenario where some request need to wait because of the number of 
    available blocks. """

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

    return (seqs_max_tokens, prompts_lengths, steps_add_reqs, checked_steps,
            max_num_seqs, available_blocks)


def augment_checked_steps(
        checked_steps: list[dict[str, Any]]) -> deque[dict[str, Any]]:
    # Augment checked_steps: add in-between normal decode steps
    checked_steps = deque(checked_steps)
    all_checked_steps = deque()
    prev_step = None
    for step in range(checked_steps[-1]["step"] + 1):
        if checked_steps and step == checked_steps[0]["step"]:
            prev_step = checked_steps.popleft()
            all_checked_steps.append(prev_step)
        elif prev_step is not None:
            assert prev_step["step"] == step - 1
            new_step = copy.deepcopy(prev_step)
            new_step["step"] = step
            new_step["tkv"] += 1
            all_checked_steps.append(new_step)
            prev_step = new_step
    return all_checked_steps


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize(
    "seqs_max_tokens,prompts_lengths,steps_add_reqs,checked_steps,"
    "max_num_seqs,available_blocks", [
        get_params_test_blocks_borders_aligned_prompts(),
        get_params_test_blocks_borders_misaligned_prompts(),
        get_params_test_special_finish(),
        get_params_test_scheduler_constraints_tkv(),
        get_params_test_scheduler_constraints_max_prompt_len(),
        get_params_test_scheduler_constraints_max_available_blocks(),
        get_params_test_scheduler_constraints_more_than_available_blocks(),
    ])
def test_scheduler_cb_steps_tkv(
    model: str,
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
    seqs_max_tokens: list[int],
    prompts_lengths: list[int],
    steps_add_reqs: list[int],
    checked_steps: list[dict[str, Any]],
    max_num_seqs: int,
    available_blocks: int,
):
    """
    Test the scheduler execution by comparing the scheduler attributes at each 
    step with the provided reference values in 'checked_steps'.
    
    The missing steps from 'checked_steps' are automatically generated as decode
    steps, based on the existing elements in the list. For that to work, all the
    prefill steps and the first decode step after them needs be added to 
    'checked_steps'
    """

    # set env vars
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
    monkeypatch.setenv("VLLM_USE_V1", "1")
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", backend)
    if available_blocks > 0:
        monkeypatch.setenv("VLLM_SPYRE_N_BLOCKS", str(available_blocks))

    max_model_len = 256

    # Input parameters sanity check, not actual testing
    # ------
    if not (len(prompts_lengths) == len(seqs_max_tokens)
            and len(prompts_lengths) == len(steps_add_reqs)):
        raise ValueError(
            "Number of prompts should be consistent with number of max tokens."
        )

    if not (steps_add_reqs == sorted(steps_add_reqs)
            and steps_add_reqs[0] == 0):
        raise ValueError(
            "The list of steps where requests are added should be increasing "
            "start with 0")

    if not (checked_steps == sorted(checked_steps, key=lambda x: x["step"])
            and len(checked_steps) == len(set(x["step"]
                                              for x in checked_steps))):
        raise ValueError(
            "List of checked steps needs to be of increasing order of step")
    # ------

    # Setup the engine
    engine_args = EngineArgs(model=model,
                             tokenizer=model,
                             max_model_len=max_model_len,
                             block_size=max_model_len,
                             max_num_seqs=max_num_seqs)
    vllm_config = engine_args.create_engine_config()
    executor_class = Executor.get_class(vllm_config)
    engine_core = EngineCore(vllm_config=vllm_config,
                             executor_class=executor_class,
                             log_stats=False)
    scheduler: ContinuousBatchingSpyreScheduler = engine_core.scheduler

    # Create random requests of specified lengths and max_tokens
    sorted_reqs_params = zip(steps_add_reqs, seqs_max_tokens, prompts_lengths)
    requests: deque[tuple[int, EngineCoreRequest]] = deque()
    for i, (add_step, max_tokens,
            prompt_length) in enumerate(sorted_reqs_params):
        # ignoring eos because we want to force the decoding to finish
        # after max_tokens exactly
        sampling_params = SamplingParams(max_tokens=max_tokens,
                                         temperature=0.0,
                                         ignore_eos=True)
        request = create_random_request(request_id=i,
                                        num_tokens=prompt_length,
                                        sampling_params=sampling_params)
        requests.append((add_step, request))

    # In-between steps are added as normal decode steps
    checked_steps = augment_checked_steps(checked_steps)

    # Run steps, until last step from 'checked_steps' is reached
    request_outputs = []
    requested_blocks, reserved_blocks = {}, {}
    for step in range(checked_steps[-1]['step'] + 1):
        # Add requests for this step
        while requests and requests[0][0] == step:
            engine_core.add_request(requests.popleft()[1])

        # Check step if it is in the provided list of steps to check
        if checked_steps and step == checked_steps[0]["step"]:
            step_ref = checked_steps.popleft()

            waiting = [r.request_id for r in scheduler.waiting]
            running = [r.request_id for r in scheduler.running]
            out_reqs_ids = [r.request_id for r in request_outputs]
            out_reqs_finished = [
                r.request_id for r in request_outputs if r.finished
            ]

            assert scheduler.tkv == step_ref["tkv"], f"Step {step}, tkv"
            assert waiting == step_ref["waiting"], f"Step {step}, num waiting"
            assert running == step_ref["running"], f"Step {step}, num running"
            assert out_reqs_ids == step_ref["request_outputs"], \
                f"Step {step}, request outputs"

            ref_finished_reqs = step_ref.get("finished_requests", [])
            assert out_reqs_finished == ref_finished_reqs, \
                f"Step {step}, finished request output"

            # checking the scheduler handling of free and reserved blocks
            n_blocks = (engine_core.model_executor.driver_worker.worker.
                        model_runner.n_blocks)
            n_reserved_blocks = n_blocks - scheduler.n_free_blocks
            req_ids2blocks = (engine_core.model_executor.driver_worker.worker.
                              model_runner.req_ids2blocks)
            req_ids2reserved_blocks = (
                engine_core.model_executor.driver_worker.worker.model_runner.
                req_ids2reserved_blocks)
            n_used_blocks = sum(
                [len(blocks) for blocks in req_ids2blocks.values()])

            if step > 0:
                assert n_reserved_blocks == step_ref[
                    "n_reserved_blocks"], f"Step {step}, n_reserved_blocks"
                assert n_used_blocks == step_ref[
                    "n_used_blocks"], f"Step {step}, n_used_blocks"

            assert len(req_ids2blocks) == len(req_ids2reserved_blocks)
            for req_id in req_ids2blocks:
                # current number of used blocks should be less than reserved
                assert len(
                    req_ids2blocks[req_id]) <= req_ids2reserved_blocks[req_id]
                # update requested/reserved blocks to check in last step
                # Note: overwrite and not max because of reduce_left_padding()
                requested_blocks[req_id] = len(req_ids2blocks[req_id])
                reserved_blocks[req_id] = req_ids2reserved_blocks[req_id]

        # last step: check that sequences used all their reserved blocks
        # Note: no early stopping, all sequences produce max_num_tokens
        if len(checked_steps) == 0:
            for req_id in requested_blocks:
                assert requested_blocks[req_id] == reserved_blocks[req_id]

        # Perform next step
        step_output = engine_core.step()
        # backward compatibility
        if isinstance(step_output, tuple):
            engine_core_output = step_output[0].get(0)
            request_outputs = (engine_core_output.outputs
                               if engine_core_output is not None else [])
        else:
            request_outputs = step_output.outputs

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
# These values are all parameterized for test sorting
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("max_model_len", [256])
@pytest.mark.parametrize("max_num_batched_tokens", [128])
@pytest.mark.parametrize("available_blocks", [None])
def test_prefix_hit(model: ModelInfo, backend: str,
                    monkeypatch: pytest.MonkeyPatch, set_random_seed,
                    max_num_seqs: int, max_model_len: int,
                    max_num_batched_tokens: int, available_blocks: int):
    """ Scenario where two equal sequences are scheduled. 
    While prefilling the second sequence we have a prefix cache
    hit and can reuse the first chunk. 

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
            "n_used_blocks": 8
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
    )

"""Verification of vLLM output by comparing with HF
with VLLM_SPYRE_MAX_LOAD_PROCESSES enabled.

Run `python -m pytest tests/e2e/test_stagger_spyre_basic.py`.
"""

import pytest
from output_util import validate_vllm_vs_hf_output
from spyre_util import ModelInfo, get_chicken_soup_prompts, skip_unsupported_tp_size
from vllm import SamplingParams

sb_mark = pytest.param("sb", marks=pytest.mark.sb, id="sb")
cb_mark = pytest.param("cb", marks=pytest.mark.cb, id="cb")


@pytest.mark.parametrize("mode", [sb_mark, cb_mark])
def test_stagger_output(
    model: ModelInfo,
    tp_size: int,
    backend: str,
    mode: str,
    max_num_seqs: int,
    max_model_len: int,
    warmup_shapes,
    monkeypatch: pytest.MonkeyPatch,
    use_llm_cache,
) -> None:
    """
    This test verifies that generated output is still correct
    when stagger mode is enabled.
    VLLM_SPYRE_MAX_LOAD_PROCESSES is set to 1, allowing
    only a single worker to load or compile the model at
    a time.
    """

    skip_unsupported_tp_size(tp_size, backend)
    monkeypatch.setenv("VLLM_SPYRE_MAX_LOAD_PROCESSES", "1")

    prompts = get_chicken_soup_prompts(4)
    warmup_shape = (64, 20, 4)

    kwargs = (
        {
            "max_num_seqs": max_num_seqs,
            "use_cb": True,
        }
        if mode == "cb"
        else {"warmup_shapes": warmup_shapes}
    )

    max_new_tokens = warmup_shape[1]

    vllm_sampling_params = SamplingParams(
        max_tokens=max_new_tokens,
        temperature=0,
        logprobs=0,  # return logprobs of generated tokens only
        ignore_eos=True,
    )

    validate_vllm_vs_hf_output(
        model=model,
        prompts=prompts,
        sampling_params=vllm_sampling_params,
        tensor_parallel_size=tp_size,
        backend=backend,
        monkeypatch=monkeypatch,
        max_model_len=max_model_len,
        max_new_tokens=max_new_tokens,
        **kwargs,
    )

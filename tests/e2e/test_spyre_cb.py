
"""Verification of continuous batching
Run `python -m pytest tests/e2e/test_spyre_cb.py`.
"""

import pytest
from spyre_util import generate_spyre_vllm_output, get_spyre_model_list
from vllm import SamplingParams


@pytest.mark.cb
@pytest.mark.parametrize("model", get_spyre_model_list())
@pytest.mark.parametrize(
    "backend", [pytest.param("eager", marks=pytest.mark.cpu, id="eager")])
@pytest.mark.parametrize("heterog_tkv", [True, False])
def test_cb_max_tokens(
    model: str,
    backend: str,
    heterog_tkv: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that continuous batches of requests that
    are longer than the `max_model_len` are correctly rejected"""

    if heterog_tkv:
        monkeypatch.setenv("VLLM_SPYRE_HETEROGEN_TKV", "1")

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
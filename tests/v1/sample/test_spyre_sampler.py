# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

from types import SimpleNamespace

import pytest

from sendnn_inference.v1.sample.spyre_sampler import SpyreSampler


def _make_vllm_config(max_concurrency=1, vocab_size=128, use_text_config=False):
    if use_text_config:
        hf_config = SimpleNamespace(text_config=SimpleNamespace(vocab_size=vocab_size))
    else:
        hf_config = SimpleNamespace(vocab_size=vocab_size)
    return SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=max_concurrency),
        model_config=SimpleNamespace(hf_config=hf_config),
    )


@pytest.fixture(
    params=[
        _make_vllm_config(use_text_config=False),
        _make_vllm_config(use_text_config=True),
    ],
    ids=["hf_config_vocab_size", "text_config_vocab_size"],
)
def valid_vllm_config(request):
    """Return valid vLLM config variants expected to initialize SpyreSampler."""
    return request.param


@pytest.fixture(
    params=[
        SimpleNamespace(
            scheduler_config=SimpleNamespace(),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=128)),
        ),
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        ),
        SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_seqs=1),
            model_config=SimpleNamespace(hf_config=SimpleNamespace(text_config=SimpleNamespace())),
        ),
    ],
    ids=["missing_concurrency", "missing_vocab_size", "missing_nested_vocab_size"],
)
def invalid_vllm_config(request):
    """Return incomplete vLLM config objects that should fail validation."""
    return request.param


class TestSpyreSampler:
    """Test suite for SpyreSampler."""

    def test_initialization_rejects_fp64_gumbel(self, valid_vllm_config):
        """Test that SpyreSampler raises ValueError with use_fp64_gumbel=True."""
        with pytest.raises(ValueError, match="SpyreSampler does not support use_fp64_gumbel=True"):
            SpyreSampler(vllm_config=valid_vllm_config, use_fp64_gumbel=True)

    def test_initialization_accepts_supported_vllm_config_variants(self, valid_vllm_config):
        """SpyreSampler should initialize when the required vLLM config fields are present."""
        sampler = SpyreSampler(vllm_config=valid_vllm_config)

        assert sampler.topk_topp_sampler is not None
        assert sampler.topk_topp_sampler._noise_buffer is not None
        assert SpyreSampler.is_vllm_config_compatible(valid_vllm_config) is True

    def test_initialization_rejects_incomplete_vllm_config(self, invalid_vllm_config):
        """SpyreSampler should require both concurrency and vocabulary size metadata."""
        with pytest.raises(ValueError):
            SpyreSampler(vllm_config=invalid_vllm_config)

        assert SpyreSampler.is_vllm_config_compatible(invalid_vllm_config) is False

    def test_is_vllm_config_compatible(self, valid_vllm_config, invalid_vllm_config):
        """Compatibility checks should only pass when both required values are present."""
        assert SpyreSampler.is_vllm_config_compatible(valid_vllm_config) is True
        assert SpyreSampler.is_vllm_config_compatible(invalid_vllm_config) is False

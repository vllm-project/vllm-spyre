# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

import torch

from sendnn_inference.v1.sample.spyre_topk_topp_sampler import SpyreTopKTopPSampler


class TestSpyreTopKTopPSampler:
    """Test suite for SpyreTopKTopPSampler."""

    def test_initialization_with_valid_params(self):
        """Test that SpyreTopKTopPSampler initializes successfully with valid parameters."""
        vocab_size = 1000
        max_batch_size = 32

        sampler = SpyreTopKTopPSampler(
            vocab_size=vocab_size,
            max_batch_size=max_batch_size,
            logprobs_mode="raw_logprobs",
        )

        assert sampler is not None
        assert sampler._noise_buffer is not None
        sampler.shutdown()

    def test_forward_returns_valid_samples(self):
        """Test that forward pass returns valid sampled token indices."""
        vocab_size = 100
        max_batch_size = 8
        batch_size = 4

        sampler = SpyreTopKTopPSampler(
            vocab_size=vocab_size,
            max_batch_size=max_batch_size,
        )

        # Create dummy logits
        logits = torch.randn(batch_size, vocab_size)

        # Forward pass without top-k/top-p constraints
        samples, logprobs = sampler.forward(
            logits=logits,
            generators={},
            k=None,
            p=None,
        )

        # Verify output shapes and types
        assert samples.shape == (batch_size,), (
            f"Expected shape ({batch_size},), got {samples.shape}"
        )
        assert samples.dtype == torch.long, f"Expected dtype torch.long, got {samples.dtype}"
        assert logprobs is None, "Expected logprobs to be None with raw_logprobs mode"

        # Verify sampled tokens are within vocab range
        assert (samples >= 0).all() and (samples < vocab_size).all(), (
            "Sampled tokens should be within vocabulary range"
        )

        sampler.shutdown()

    def test_gumble_max_trick(self):
        """Test that sampling with log noise yields same tokens as regular sampling."""
        batch_size = 32
        vocab_size = 10000

        logits = torch.randn(batch_size, vocab_size)
        probs = logits.softmax(dim=-1, dtype=logits.dtype)
        noise = torch.empty_like(logits).exponential_()
        log_noise = torch.log(noise)

        expected_sampled_ids = SpyreTopKTopPSampler._sample_with_predrawn_noise(probs, noise)
        gumble_sampled_ids = SpyreTopKTopPSampler._sample_with_predrawn_log_noise(logits, log_noise)

        assert torch.all(expected_sampled_ids == gumble_sampled_ids)

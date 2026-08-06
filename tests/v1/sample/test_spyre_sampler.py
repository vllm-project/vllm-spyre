# # SPDX-License-Identifier: Apache-2.0
"""Tests for the pooled-noise Spyre sampler.

These validate that :class:`SpyreTopKTopPSampler` is a faithful, faster stand-in
for vLLM's ``random_sample``: it produces the same shaped output, samples from
the correct distribution, honors per-request seeded generators, and builds its
noise pool exactly once (the whole point — no per-step ``exponential_()``).
"""

import torch
import pytest

from sendnn_inference.v1.sample.spyre_sampler import (
    SpyreSampler,
    SpyreTopKTopPSampler,
)

VOCAB_SIZE = 512


def _make_logits(batch: int, peak_token: int | None = None) -> torch.Tensor:
    logits = torch.randn(batch, VOCAB_SIZE, dtype=torch.float32)
    if peak_token is not None:
        # Make one token dominate so argmax sampling is deterministic-ish.
        logits[:, peak_token] = 100.0
    return logits


@pytest.mark.cpu
def test_pool_built_once_and_lazily():
    sampler = SpyreTopKTopPSampler(pool_size=64)
    assert sampler._noise_pool is None

    logits = _make_logits(4)
    out, _ = sampler.forward_native(logits, generators={}, k=None, p=None)

    assert out.shape == (4,)
    pool = sampler._noise_pool
    assert pool is not None
    assert pool.shape == (64, VOCAB_SIZE)

    # A second call must reuse the same pool object, not rebuild it.
    sampler.forward_native(_make_logits(4), generators={}, k=None, p=None)
    assert sampler._noise_pool is pool


@pytest.mark.cpu
def test_prebuild_allocates_pool_up_front():
    sampler = SpyreTopKTopPSampler(pool_size=32)
    sampler.prebuild(vocab_size=VOCAB_SIZE, device=torch.device("cpu"))
    pool = sampler._noise_pool
    assert pool is not None
    assert pool.shape == (32, VOCAB_SIZE)

    # Sampling should reuse the prebuilt pool.
    sampler.forward_native(_make_logits(2), generators={}, k=None, p=None)
    assert sampler._noise_pool is pool


@pytest.mark.cpu
def test_rebuild_on_vocab_mismatch():
    sampler = SpyreTopKTopPSampler(pool_size=16)
    # Prebuild with a wrong (smaller) vocab guess.
    sampler.prebuild(vocab_size=VOCAB_SIZE // 2, device=torch.device("cpu"))
    assert sampler._noise_pool.shape[1] == VOCAB_SIZE // 2

    out, _ = sampler.forward_native(_make_logits(3), generators={}, k=None, p=None)
    assert out.shape == (3,)
    # Pool transparently rebuilt to the real width.
    assert sampler._noise_pool.shape[1] == VOCAB_SIZE


@pytest.mark.cpu
def test_offset_wraps_around_pool():
    sampler = SpyreTopKTopPSampler(pool_size=8)
    sampler.prebuild(vocab_size=VOCAB_SIZE, device=torch.device("cpu"))

    # batch=6 -> offset 6; next batch=6 would overflow (6+6 > 8) -> reset to 0.
    sampler.forward_native(_make_logits(6), generators={}, k=None, p=None)
    assert sampler._noise_offset == 6
    sampler.forward_native(_make_logits(6), generators={}, k=None, p=None)
    assert sampler._noise_offset == 6  # wrapped to 0, then advanced by 6


@pytest.mark.cpu
def test_batch_larger_than_pool_falls_back():
    sampler = SpyreTopKTopPSampler(pool_size=4)
    sampler.prebuild(vocab_size=VOCAB_SIZE, device=torch.device("cpu"))
    # batch (10) > pool (4): must still produce valid output via fresh noise.
    out, _ = sampler.forward_native(_make_logits(10), generators={}, k=None, p=None)
    assert out.shape == (10,)


@pytest.mark.cpu
def test_disabled_pool_uses_fresh_noise():
    sampler = SpyreTopKTopPSampler(pool_size=0)
    out, _ = sampler.forward_native(_make_logits(4), generators={}, k=None, p=None)
    assert out.shape == (4,)
    assert sampler._noise_pool is None


@pytest.mark.cpu
def test_samples_peaked_distribution():
    # With one token overwhelmingly likely, sampling should almost always
    # return it — confirms noise is applied to the right (softmax) axis.
    sampler = SpyreTopKTopPSampler(pool_size=64)
    peak = 123
    out, _ = sampler.forward_native(
        _make_logits(32, peak_token=peak), generators={}, k=None, p=None
    )
    assert (out == peak).float().mean() > 0.95


@pytest.mark.cpu
def test_pool_not_mutated_by_sampling():
    # sample_with_exponential_noise divides probs in place, not the noise. The
    # shared pool must survive unchanged so later steps get valid noise.
    sampler = SpyreTopKTopPSampler(pool_size=8)
    sampler.prebuild(vocab_size=VOCAB_SIZE, device=torch.device("cpu"))
    before = sampler._noise_pool.clone()
    sampler.forward_native(_make_logits(4), generators={}, k=None, p=None)
    torch.testing.assert_close(sampler._noise_pool, before)


@pytest.mark.cpu
def test_seeded_generator_is_reproducible_and_pool_safe():
    sampler = SpyreTopKTopPSampler(pool_size=8)
    sampler.prebuild(vocab_size=VOCAB_SIZE, device=torch.device("cpu"))
    pool_before = sampler._noise_pool.clone()

    logits = _make_logits(2)

    def run():
        g = torch.Generator()
        g.manual_seed(1234)
        # Seed every row so the whole batch is reproducible.
        gens = {0: g, 1: torch.Generator().manual_seed(1234)}
        return sampler.forward_native(logits.clone(), generators=gens, k=None, p=None)[0]

    out1 = run()
    out2 = run()
    torch.testing.assert_close(out1, out2)
    # Seeded rows are cloned before overwriting, so the pool is untouched.
    torch.testing.assert_close(sampler._noise_pool, pool_before)


@pytest.mark.cpu
def test_distribution_matches_reference():
    # Statistically compare sampled frequencies against the exact softmax
    # probabilities over many draws for a small vocab.
    torch.manual_seed(0)
    small_vocab = 8
    sampler = SpyreTopKTopPSampler(pool_size=1000)
    sampler.prebuild(vocab_size=small_vocab, device=torch.device("cpu"))

    base_logits = torch.tensor([3.0, 2.0, 1.0, 0.0, -1.0, -2.0, 0.5, 1.5])
    expected = torch.softmax(base_logits, dim=-1)

    counts = torch.zeros(small_vocab)
    draws = 200
    per = 100
    for _ in range(draws):
        logits = base_logits.unsqueeze(0).expand(per, -1).contiguous()
        out, _ = sampler.forward_native(logits, generators={}, k=None, p=None)
        counts += torch.bincount(out, minlength=small_vocab).float()

    freq = counts / counts.sum()
    # Loose tolerance: reused pool noise adds correlation, but marginals hold.
    assert torch.allclose(freq, expected, atol=0.05)


@pytest.mark.cpu
def test_spyre_sampler_installs_pooled_topk_topp():
    sampler = SpyreSampler()
    assert isinstance(sampler.topk_topp_sampler, SpyreTopKTopPSampler)
    sampler.prebuild_noise_pool(vocab_size=VOCAB_SIZE, device=torch.device("cpu"))
    assert sampler.topk_topp_sampler._noise_pool.shape == (
        sampler.topk_topp_sampler._pool_size,
        VOCAB_SIZE,
    )

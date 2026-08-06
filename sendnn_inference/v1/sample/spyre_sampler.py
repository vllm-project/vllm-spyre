"""Spyre-specific sampler with a pre-generated exponential noise pool.

vLLM's random sampler (``vllm.v1.sample.ops.topk_topp_sampler``) draws fresh
Gumbel/exponential noise on every step via ``q.exponential_()`` over a
``[batch, vocab]`` tensor. On the s390x host this call cannot be compiled and
costs ~50ms for a large batch on a real model, which dominates our per-step
latency (the sampler runs on the host CPU on Spyre, not on the accelerator).

To avoid paying that cost every step we generate a large pool of exponential
noise once, up front, and slice rows out of it at sample time instead of
calling ``exponential_()`` on the hot path. Statistically each pool row is an
i.i.d. exponential vector, exactly what ``random_sample`` would have produced;
the only difference is that noise vectors are reused over the lifetime of the
process. We advance a rolling offset through the pool so that a given batch
slot rarely sees the same row on consecutive steps, and the sampled logits
differ every step anyway, so the reuse is not observable in practice.

Requests that carry their own seeded ``torch.Generator`` still need
reproducible noise, so those rows are drawn with ``exponential_(generator=...)``
as upstream does — the pool only serves the common, unseeded case.
"""

import torch
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.v1.sample.ops.topk_topp_sampler import (
    TopKTopPSampler,
    apply_top_k_top_p,
    sample_with_exponential_noise,
)
from vllm.v1.sample.sampler import Sampler

import sendnn_inference.envs as envs_spyre

logger = init_logger(__name__)


class SpyreTopKTopPSampler(TopKTopPSampler):
    """``TopKTopPSampler`` that serves exponential noise from a fixed pool."""

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
        pool_size: int | None = None,
    ) -> None:
        super().__init__(logprobs_mode, use_fp64_gumbel)
        # Spyre is an out-of-tree platform, so the base class already binds
        # ``forward_native``. Pin it explicitly so we keep using our overridden
        # implementation even if upstream's platform dispatch changes.
        self.forward = self.forward_native

        if pool_size is None:
            pool_size = envs_spyre.SENDNN_INFERENCE_SAMPLER_NOISE_POOL_SIZE
        self._pool_size = pool_size
        # Lazily allocated on first use (or via ``prebuild``) once we know the
        # vocab width / dtype / device of the logits.
        self._noise_pool: torch.Tensor | None = None
        self._noise_offset = 0

    def prebuild(self, vocab_size: int, device: torch.device) -> None:
        """Eagerly build the noise pool, e.g. right after model load.

        Safe to call with a best-guess ``vocab_size``: if the actual logits
        width differs at sample time the pool is transparently rebuilt.
        """
        dtype = torch.float64 if self.use_fp64_gumbel else torch.float32
        self._build_pool(vocab_size, dtype, device)

    def _build_pool(
        self, vocab_size: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        if self._pool_size <= 0:
            # Pool disabled: callers fall back to fresh per-step noise.
            self._noise_pool = None
            return torch.empty(0, device=device, dtype=dtype)

        logger.info(
            "Building sampler noise pool: shape=[%d, %d], dtype=%s, device=%s (~%.0f MB)",
            self._pool_size,
            vocab_size,
            dtype,
            device,
            self._pool_size * vocab_size * torch.empty(0, dtype=dtype).element_size() / 1e6,
        )
        pool = torch.empty((self._pool_size, vocab_size), dtype=dtype, device=device)
        pool.exponential_()
        self._noise_pool = pool
        self._noise_offset = 0
        return pool

    def _get_pooled_noise(self, probs: torch.Tensor) -> torch.Tensor:
        """Return a ``[batch, vocab]`` slice of exponential noise.

        The returned tensor may be a view into the shared pool; callers must
        not mutate it in place (see ``forward_native``).
        """
        batch = probs.shape[0]
        vocab_size = probs.shape[1]
        dtype = torch.float64 if self.use_fp64_gumbel else probs.dtype

        pool = self._noise_pool
        if (
            pool is None
            or pool.shape[1] != vocab_size
            or pool.dtype != dtype
            or pool.device != probs.device
        ):
            pool = self._build_pool(vocab_size, dtype, probs.device)

        # A pool of size 0 (disabled) or a batch larger than the pool can't be
        # served from the pool — fall back to fresh noise for this step.
        if self._noise_pool is None or batch > pool.shape[0]:
            q = torch.empty((batch, vocab_size), dtype=dtype, device=probs.device)
            q.exponential_()
            return q

        if self._noise_offset + batch > pool.shape[0]:
            self._noise_offset = 0
        start = self._noise_offset
        self._noise_offset += batch
        return pool[start : start + batch]

    def forward_native(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Mirror of the upstream native path, but noise comes from the pool."""
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == "processed_logits":
            logits_to_return = logits
        elif self.logprobs_mode == "processed_logprobs":
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = self._get_pooled_noise(probs)

        # We must not mutate the shared pool. ``sample_with_exponential_noise``
        # only mutates ``q`` in place when its dtype differs from ``probs``
        # (the fp64-gumbel reciprocal path); seeded generators also require
        # overwriting rows. Clone in those cases; otherwise use the view
        # directly for a zero-copy hot path.
        if generators or q.dtype != probs.dtype:
            q = q.clone()
            for i, generator in generators.items():
                q[i].exponential_(generator=generator)

        return sample_with_exponential_noise(probs, q), logits_to_return


class SpyreSampler(Sampler):
    """``Sampler`` that swaps in :class:`SpyreTopKTopPSampler`.

    Drop-in replacement for ``vllm.v1.sample.sampler.Sampler`` used by
    ``SpyreCausalLM`` so random sampling avoids per-step ``exponential_()``.
    """

    def __init__(
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ) -> None:
        super().__init__(logprobs_mode, use_fp64_gumbel)
        self.topk_topp_sampler = SpyreTopKTopPSampler(logprobs_mode, use_fp64_gumbel)

    def prebuild_noise_pool(self, vocab_size: int, device: torch.device) -> None:
        """Build the noise pool up front (best-effort; rebuilt on mismatch)."""
        # nn.Module attribute access is typed as Module | Tensor; narrow it.
        sampler = self.topk_topp_sampler
        assert isinstance(sampler, SpyreTopKTopPSampler)
        sampler.prebuild(vocab_size, device)

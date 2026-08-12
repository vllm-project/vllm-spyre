# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

from vllm.config import VllmConfig
from vllm.config.model import LogprobsMode
from vllm.v1.sample.sampler import Sampler

from sendnn_inference.v1.sample.spyre_topk_topp_sampler import SpyreTopKTopPSampler


class SpyreSampler(Sampler):
    """A vLLM Sampler subclass that uses top-k/top-p sampling implementations optimized for Spyre platform."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ):
        """Initialize the SpyreSampler with Spyre-optimized sampling components.

        Initializes the parent Sampler and replaces the default top-k/top-p sampler
        with a Spyre-specific implementation. Configuration parameters are extracted
        from vllm_config to ensure the sampler is properly tuned for the target
        hardware and model.

        Args:
            vllm_config: The VLLMConfig instance containing model configuration,
                vocabulary size, and concurrency settings needed to initialize
                the Spyre sampler.
            logprobs_mode: See vllm.v1.sample.sampler.Sampler for details.
            use_fp64_gumbel: See vllm.v1.sample.sampler.Sampler for details.
                This parameter is not supported by SpyreSampler.
                Defaults to False.
        """
        if use_fp64_gumbel:
            raise ValueError("SpyreTopKTopPSampler does not support use_fp64_gumbel=True")

        super().__init__(logprobs_mode=logprobs_mode, use_fp64_gumbel=False)

        # read concurrency and vocab size from vllm_config
        max_concurrent_batches = vllm_config.max_concurrent_batches
        vocab_size = vllm_config.model_config.hf_config.vocab_size

        # override topk_topp_sampler with spyre-specific topk-topp-sampler
        self.topk_topp_sampler = SpyreTopKTopPSampler(
            max_batch_size=max_concurrent_batches,
            vocab_size=vocab_size,
            logprobs_mode=logprobs_mode,
        )

    def shutdown(self) -> None:
        """Shutdown the sampler and clean up resources."""
        self.topk_topp_sampler.shutdown()

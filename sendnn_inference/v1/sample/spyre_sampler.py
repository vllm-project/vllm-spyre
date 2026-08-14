# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

import torch
from vllm.config import VllmConfig
from vllm.config.model import LogprobsMode
from vllm.distributed import get_tp_group
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
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

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        if logits.device.type == "cpu":
            # if the sampler runs on CPU, use an optimized path that avoids redundant computations across ranks
            return self.forward_cpu(
                logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
            )
        else:
            # if the sampler does not run on CPU, fall back to the base class implementation
            return super().forward(
                logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
            )

    def forward_cpu(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        """Overrides the upstream sampler to run only on the first TP-rank and broadcast
        results to other TP ranks.

        This is a correctness fix, because independent sampling across ranks would diverge
        the computation across ranks over time. Further, this fix improves performance if
        the sampler runs on CPU by avoiding redundant computations.
        """

        tp_group = get_tp_group()
        if tp_group.is_first_rank:
            sampler_output = super().forward(
                logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
            )
        else:
            # Allocate placeholder; will be filled by the broadcast below.
            num_reqs = logits.shape[0]
            sampler_output = SamplerOutput(
                sampled_token_ids=torch.empty(
                    (num_reqs, 1), dtype=torch.int32, device=logits.device
                ),
                logprobs_tensors=None,
            )

        # Broadcast sampled token ids from TP rank 0 to all other TP ranks so
        # that every rank feeds identical tokens into the next forward pass.
        tp_group.broadcast(sampler_output.sampled_token_ids, src=0)

        # Broadcast the logprobs_tensors (broadcast_object handles None) and
        # update sampler outputs
        logprobs_tensors = tp_group.broadcast_object(sampler_output.logprobs_tensors, src=0)
        sampler_output.logprobs_tensors = logprobs_tensors

        return sampler_output

    def shutdown(self) -> None:
        """Shutdown the sampler and clean up resources."""
        self.topk_topp_sampler.shutdown()

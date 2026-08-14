# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the sendnn-inference project

import torch
from vllm.config import VllmConfig
from vllm.config.model import LogprobsMode
from vllm.distributed import get_tp_group
from vllm.distributed.parallel_state import _TP
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from sendnn_inference.v1.sample.spyre_topk_topp_sampler import SpyreTopKTopPSampler


class SpyreSampler(Sampler):
    """A vLLM Sampler subclass that uses top-k/top-p sampling implementations optimized for Spyre
    platform.
    """

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

        Raises:
            ValueError: If use_fp64_gumbel is True, as SpyreSampler does
                not support 64-bit Gumbel noise computation.
            ValueError: If vllm_config does not provide max_num_seqs or vocab_size,
                which are required for SpyreSampler initialization.
        """
        if use_fp64_gumbel:
            raise ValueError("SpyreSampler does not support use_fp64_gumbel=True")

        super().__init__(logprobs_mode=logprobs_mode, use_fp64_gumbel=False)

        # read concurrency and vocab size from vllm_config
        max_concurrency = SpyreSampler._try_get_concurrency(vllm_config)
        if max_concurrency is None:
            raise ValueError("SpyreSampler requires vllm_config to specify max_num_seqs")
        vocab_size = SpyreSampler._try_get_vocab_size(vllm_config)
        if vocab_size is None:
            raise ValueError("SpyreSampler requires vllm_config to specify vocab_size")

        # override topk_topp_sampler with spyre-specific topk-topp-sampler
        self.topk_topp_sampler: SpyreTopKTopPSampler = SpyreTopKTopPSampler(
            max_batch_size=max_concurrency,
            vocab_size=vocab_size,
            logprobs_mode=logprobs_mode,
        )

    @staticmethod
    def is_vllm_config_compatible(vllm_config: VllmConfig) -> bool:
        """Check if the provided VllmConfig provides all necessary parameters for SpyreSampler
        initialization.
        """
        has_concurrency = SpyreSampler._try_get_concurrency(vllm_config) is not None
        has_vocab_size = SpyreSampler._try_get_vocab_size(vllm_config) is not None
        return has_concurrency and has_vocab_size

    @staticmethod
    def _try_get_concurrency(vllm_config: VllmConfig) -> int | None:
        """Try to extract the max_num_seqs parameter from the VllmConfig.

        Returns:
            The max_num_seqs value if present, otherwise None.
        """
        return getattr(vllm_config.scheduler_config, "max_num_seqs", None)

    @staticmethod
    def _try_get_vocab_size(vllm_config: VllmConfig) -> int | None:
        """Try to extract the vocab_size parameter from the VllmConfig.

        Returns:
            The vocab_size value if present, otherwise None.
        """
        if hasattr(vllm_config, "model_config") and hasattr(vllm_config.model_config, "hf_config"):
            hf_cfg = vllm_config.model_config.hf_config
            if hasattr(hf_cfg, "vocab_size"):
                # convention: HuggingFace model configs have a vocab_size attribute
                return hf_cfg.vocab_size
            elif hasattr(hf_cfg, "text_config") and hasattr(hf_cfg.text_config, "vocab_size"):
                # fallback: some multi-modal HuggingFace model configs have a text_config
                # with a vocab_size attribute
                return hf_cfg.text_config.vocab_size
        return None

    def forward(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        if logits.device.type == "cpu" and _TP is not None:
            # if the sampler runs on CPU and is distributed across tensor parallel ranks,
            # use an optimized path on CPU that avoids redundant computations across ranks
            return self.forward_cpu_tp(
                logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
            )
        else:
            # if the sampler does not run on CPU, fall back to the base class implementation
            return super().forward(
                logits, sampling_metadata, predict_bonus_token, logprobs_mode_override
            )

    def forward_cpu_tp(
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

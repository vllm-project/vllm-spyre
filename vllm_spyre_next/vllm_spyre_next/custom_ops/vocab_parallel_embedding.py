# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre CPU fallback for VocabParallelEmbedding.

Remove this file once Spyre supports embedding natively.
"""

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)

from .cpu_fallback import SpyreCpuFallbackMixin

logger = init_logger(__name__)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(SpyreCpuFallbackMixin, VocabParallelEmbedding):
    """OOT VocabParallelEmbedding that falls back to CPU execution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_cpu_fallback("embedding")

    def forward_oot(self, input_):
        output = torch.empty(
            *input_.shape,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=self._output_device,
        )
        torch.ops.vllm.spyre_cpu_fallback(input_, output, self.prefix)
        return output

    def cpu_forward(self, cpu_input):
        return VocabParallelEmbedding.forward_native(self, cpu_input)


def register():
    # No per-layer op registration needed — uses generic spyre_cpu_fallback.
    # The OOT class registration happens at import time via the decorator.
    pass

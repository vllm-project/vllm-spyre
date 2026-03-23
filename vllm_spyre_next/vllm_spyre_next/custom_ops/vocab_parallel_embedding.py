# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific VocabParallelEmbedding implementation using out-of-tree (OOT) registration.

This module provides a custom VocabParallelEmbedding layer for IBM's Spyre device,
replacing the upstream vLLM implementation
(vllm/model_executor/layers/vocab_parallel_embedding.py) when instantiated.

Architecture:
    - OOT Registration: @VocabParallelEmbedding.register_oot() replaces upstream
      at instantiation
    - Custom Op Boundary: torch.ops.vllm.spyre_vocab_parallel_embedding is opaque
      to torch.compile, so forward_native runs eagerly outside the compiled graph

Spyre Device Constraints:
    - Algorithm: Standard F.embedding on CPU — torch-spyre has no native embedding
      kernel; aten.embedding.default is handled by a CPU fallback (spyre__embedding).
      Moving tensors to Spyre causes DtException (unsupported data format), so all
      embedding computation stays on CPU.

Limitations:
    - No Tensor Parallelism (TP) support: tp_size > 1 raises NotImplementedError.
    - No quantization support: quant_config != None raises NotImplementedError.

References:
    - Upstream VocabParallelEmbedding:
      vllm/model_executor/layers/vocab_parallel_embedding.py
"""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from functools import lru_cache

from .utils import register_layer, get_layer, _fake_impl

logger = init_logger(__name__)


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for IBM's Spyre device.

    This replaces the upstream vLLM VocabParallelEmbedding when instantiated,
    providing Spyre-specific device handling.

    No Tensor Parallelism (TP) is supported: tp_size > 1 raises NotImplementedError.
    No quantization is supported: quant_config != None raises NotImplementedError.
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyreVocabParallelEmbedding layer.

        Raises:
            NotImplementedError: If tp_size > 1 or quant_config is not None,
                as these are not supported in the current Spyre implementation.
        """
        super().__init__(*args, **kwargs)

        # Check for unsupported configurations after super().__init__
        # sets up the parallel config.
        quant_config = kwargs.get("quant_config")
        if quant_config is not None:
            raise NotImplementedError(
                "SpyreVocabParallelEmbedding does not support quantization "
                f"(quant_config={quant_config}). Only quant_config=None is supported."
            )

        if self.tp_size > 1:
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support Tensor Parallelism "
                f"(tp_size={self.tp_size}). TP masking and all_reduce are not implemented. "
                "Only tp_size=1 is supported."
            )

        logger.debug("Building custom VocabParallelEmbedding for Spyre")

        self._layer_name = register_layer(self, "spyre_vocab_parallel_embedding")

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom op to bypass torch.compile.

        Args:
            input_: Token index tensor [num_tokens] (int64)

        Returns:
            Embedding output [num_tokens, embedding_dim] in weight dtype
        """
        output = torch.empty(
            *input_.shape,
            self.embedding_dim,
            dtype=self.weight.dtype,
            device=input_.device,
        )
        torch.ops.vllm.spyre_vocab_parallel_embedding(input_, output, self._layer_name)
        return output

    def forward_native(self, input_: torch.Tensor) -> torch.Tensor:
        """Embedding execution on CPU.

        F.embedding runs on CPU via torch-spyre's spyre__embedding fallback.
        torch-spyre has no native embedding kernel; aten.embedding.default falls
        back to CPU (spyre__embedding). Moving tensors to Spyre would cause a
        DtException (unsupported data format), so no device transfer is performed.

        No TP masking or all_reduce is performed (tp_size > 1 is not supported).

        Args:
            input_: Token index tensor [num_tokens] on CPU (int64)

        Returns:
            Embedding output [num_tokens, embedding_dim] in weight dtype on CPU

        Raises:
            NotImplementedError: If input tensor is not on CPU.
        """
        if input_.device.type != "cpu":
            raise NotImplementedError(
                f"Expected input on CPU, got device={input_.device}. "
                "Spyre has no native embedding kernel; input must stay on CPU."
            )
        if input_.dtype not in (torch.int32, torch.int64):
            logger.warning(
                "SpyreVocabParallelEmbedding: expected integer index tensor "
                "(int32/int64), got dtype=%s. This may produce unexpected results.",
                input_.dtype,
            )
        return F.embedding(input_, self.weight)


def _op_func(
    input_: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op implementation — runs outside torch.compile graph."""
    layer = get_layer(layer_name)
    result = layer.forward_native(input_)
    output.copy_(result)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_vocab_parallel_embedding custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_vocab_parallel_embedding",
        op_func=_op_func,
        mutates_args=["output"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreVocabParallelEmbedding")

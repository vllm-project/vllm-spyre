# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific VocabParallelEmbedding implementation using out-of-tree (OOT) registration.

This module provides a custom VocabParallelEmbedding layer for IBM's Spyre device,
replacing the upstream vLLM implementation
(vllm/model_executor/layers/vocab_parallel_embedding.py) when instantiated.

Architecture Overview:
    1. OOT Registration: @VocabParallelEmbedding.register_oot() replaces upstream
       class at instantiation
    2. Custom Op Pattern: Uses torch.ops.vllm.spyre_vocab_parallel_embedding to
       bypass torch.compile
    3. Static Forward Context: Registers in compilation_config.static_forward_context
    4. No-Compile Execution: Retrieved via forward_context.no_compile_layers during
       forward

Key Components:
    - SpyreVocabParallelEmbedding: Main layer class with Spyre-specific optimizations
    - spyre_vocab_parallel_embedding: Custom op implementation (executes outside
      torch.compile)
    - spyre_vocab_parallel_embedding_fake: Fake implementation for shape inference
    - register(): Registers the custom op with vLLM

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

from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.utils.torch_utils import direct_register_custom_op


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

        Registers this instance in static_forward_context for retrieval during
        the custom op forward pass.

        Raises:
            NotImplementedError: If tp_size > 1 or quant_config is not None,
                as these are not supported in the current Spyre implementation.
        """
        # Check for unsupported configurations before calling super().__init__
        # to fail fast with a clear error message.
        quant_config = kwargs.get("quant_config")

        if quant_config is not None:
            raise NotImplementedError(
                "SpyreVocabParallelEmbedding does not support quantization "
                f"(quant_config={quant_config}). Only quant_config=None is supported."
            )

        super().__init__(*args, **kwargs)

        # Check TP size after super().__init__ sets up the parallel config.
        if self.tp_size > 1:
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support Tensor Parallelism "
                f"(tp_size={self.tp_size}). TP masking and all_reduce are not implemented. "
                "Only tp_size=1 is supported."
            )

        logger.debug("Building custom VocabParallelEmbedding for Spyre")

        # Register in static_forward_context for custom op access.
        # Each instance gets a unique name via counter to avoid collisions.
        compilation_config = get_current_vllm_config().compilation_config
        if not hasattr(SpyreVocabParallelEmbedding, "_instance_counter"):
            SpyreVocabParallelEmbedding._instance_counter = 0
        self.layer_name = (
            f"spyre_vocab_parallel_embedding_{SpyreVocabParallelEmbedding._instance_counter}"
        )
        SpyreVocabParallelEmbedding._instance_counter += 1

        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        compilation_config.static_forward_context[self.layer_name] = self

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_vocab_parallel_embedding which retrieves
        this layer from forward_context.no_compile_layers and calls forward_impl
        outside the compilation graph.

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
        torch.ops.vllm.spyre_vocab_parallel_embedding(input_, output, self.layer_name)
        return output

    def forward_impl(self, input_: torch.Tensor, output: torch.Tensor) -> None:
        """Implementation called by custom op, executes outside torch.compile.

        Args:
            input_: Token index tensor [num_tokens] (int64)
            output: Pre-allocated output tensor (modified in-place)
        """
        result = self.forward_native(input_)
        output.copy_(result)

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
        """
        return F.embedding(input_, self.weight)

    def forward_oot(self, input_: torch.Tensor) -> torch.Tensor:
        """OOT forward method — delegates to forward_native."""
        return self.forward_native(input_)


# Custom op implementation (executed outside torch.compile)


def spyre_vocab_parallel_embedding(
    input_: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op implementation — retrieves layer and executes outside compilation.

    Called by SpyreVocabParallelEmbedding.forward() via
    torch.ops.vllm.spyre_vocab_parallel_embedding. Retrieves the layer instance
    from forward_context.no_compile_layers using layer_name, then calls
    forward_impl to execute the actual computation.

    This pattern prevents torch.compile from inlining Spyre-specific operations.

    Args:
        input_: Token index tensor [num_tokens]
        output: Pre-allocated output tensor (modified in-place)
        layer_name: Unique layer identifier in static_forward_context
    """
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    layer.forward_impl(input_, output)


def spyre_vocab_parallel_embedding_fake(
    input_: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    """Fake implementation for shape/dtype inference during torch.compile.

    Provides metadata to torch.compile without executing actual computation.
    """
    return


def register():
    """Register the spyre_vocab_parallel_embedding custom op with vLLM.

    Registers torch.ops.vllm.spyre_vocab_parallel_embedding with:
        - op_func: Actual implementation (spyre_vocab_parallel_embedding)
        - fake_impl: Shape inference implementation
        - mutates_args: Indicates 'output' is modified in-place
    """
    direct_register_custom_op(
        op_name="spyre_vocab_parallel_embedding",
        op_func=spyre_vocab_parallel_embedding,
        mutates_args=["output"],
        fake_impl=spyre_vocab_parallel_embedding_fake,
    )
    logger.info("Registered custom op: SpyreVocabParallelEmbedding")

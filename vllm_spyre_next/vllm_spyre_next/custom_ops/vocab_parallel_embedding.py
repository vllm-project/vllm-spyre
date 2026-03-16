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
    - Minimum sequence length: 64 tokens (automatically padded, trimmed after)
    - Algorithm: Standard F.embedding on CPU — torch-spyre has no native embedding
      kernel; aten.embedding.default is handled by a CPU fallback (spyre__embedding).
      Moving tensors to Spyre causes DtException (unsupported data format), so all
      embedding computation stays on CPU.

Limitations:
    - No Tensor Parallelism (TP) support: TP masking and all_reduce are skipped.
      This is intentional for the current implementation scope.

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

# Minimum number of tokens required by Spyre device
_SPYRE_MIN_SEQ_LEN = 64

# Flip to True when torch-spyre adds a native embedding kernel.
# Tracked by: torch_spyre/ops/fallbacks.py
# When True: restores torch.compile + device transfer path in forward_native.
# When False: F.embedding runs directly on CPU via spyre__embedding fallback.
_SPYRE_NATIVE_EMBEDDING = False


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for IBM's Spyre device.

    This replaces the upstream vLLM VocabParallelEmbedding when instantiated,
    providing Spyre-specific device handling.

    No Tensor Parallelism (TP) is supported: the TP masking and all_reduce steps
    from forward_native are omitted. This is consistent with the current Spyre
    deployment model (single device per embedding layer).
    """

    def __init__(self, *args, **kwargs):
        """Initialize SpyreVocabParallelEmbedding layer.

        Registers this instance in static_forward_context for retrieval during
        the custom op forward pass. When _SPYRE_NATIVE_EMBEDDING is True,
        also compiles the Spyre embedding kernel via torch.compile.
        """
        super().__init__(*args, **kwargs)

        logger.debug("Building custom VocabParallelEmbedding for Spyre")

        if _SPYRE_NATIVE_EMBEDDING:
            self._fwd_spyre = torch.compile(self.forward_static, dynamic=False)

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

    @staticmethod
    def forward_static(input_: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Entry point for torch.compile when _SPYRE_NATIVE_EMBEDDING is True."""
        return F.embedding(input_, weight)

    def forward_native(self, input_: torch.Tensor) -> torch.Tensor:
        """Embedding execution with Spyre padding constraint.

        Two paths controlled by _SPYRE_NATIVE_EMBEDDING:

        False (current): F.embedding runs on CPU via torch-spyre's spyre__embedding
            fallback. Moving tensors to Spyre causes DtException (unsupported data
            format) — confirmed on Spyre pod. No device transfer performed.

        True (future): when torch-spyre adds a native gather/scatter kernel,
            flip _SPYRE_NATIVE_EMBEDDING = True. Restores device transfer and
            torch.compile path automatically.

        No TP masking or all_reduce is performed (no TP support).

        Args:
            input_: Token index tensor [num_tokens] on CPU (int64)

        Returns:
            Embedding output [num_tokens, embedding_dim] in weight dtype on CPU
        """
        original_len = input_.shape[0]
        padding = 0
        if original_len < _SPYRE_MIN_SEQ_LEN:
            padding = _SPYRE_MIN_SEQ_LEN - original_len
            input_ = F.pad(input_, (0, padding))

        if _SPYRE_NATIVE_EMBEDDING:
            input_spyre = input_.to(dtype=torch.int32).to(device="spyre")
            weight_spyre = self.weight.to(dtype=torch.float16).to(device="spyre")
            result = self._fwd_spyre(input_spyre, weight_spyre)
            result = result.to(device="cpu").to(dtype=self.weight.dtype)
        else:
            result = F.embedding(input_, self.weight)

        if padding > 0:
            result = result[:original_len]

        return result

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

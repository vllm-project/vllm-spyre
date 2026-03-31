# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spyre-specific RotaryEmbedding implementation using out-of-tree (OOT) registration.

This module provides a custom RoPE (Rotary Position Embedding) layer for IBM's Spyre device,
replacing the upstream vLLM implementation (vllm/model_executor/layers/rotary_embedding.py)
when instantiated.

Architecture:
    - OOT Registration: @RotaryEmbedding.register_oot() replaces upstream at instantiation
    - Custom Op Boundary: torch.ops.vllm.spyre_rotary_embedding is opaque to torch.compile,
      so forward_native runs eagerly outside the compiled graph
    - Separate Compilation: forward_static is compiled independently via maybe_compile

Spyre Device Constraints:
    - Device dtype: float16 (converted for Spyre)
    - Output dtype: matches input dtype (converted on CPU)
    - Cache management: cos/sin caches stored on Spyre device

Limitations:
    - No dtype promotion (torch-spyre limitation)
    - rope_scaling not yet implemented
    - Expect numerical differences from upstream vLLM

References:
    - Upstream RotaryEmbedding: vllm/model_executor/layers/rotary_embedding.py
"""

import torch
import torch.utils._pytree as pytree

from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from functools import lru_cache

from .utils import convert, register_layer, get_layer, _fake_impl

logger = init_logger(__name__)


@RotaryEmbedding.register_oot(name="RotaryEmbedding")
class SpyreRotaryEmbedding(RotaryEmbedding):
    """Out-of-tree (OOT) RotaryEmbedding implementation for IBM's Spyre device.

    This replaces the upstream vLLM RotaryEmbedding when instantiated,
    providing Spyre-specific optimizations and device handling.

    Implements RoPE (Rotary Position Embedding) which applies position-dependent
    rotations to query and key tensors in attention mechanisms.
    """

    _dynamic_arg_dims = {"query": [], "key": [], "positions": []}

    def __init__(self, *args, **kwargs):
        """Initialize SpyreRotaryEmbedding layer.

        Compiles the Spyre kernel and registers this instance in static_forward_context.
        Builds cos/sin cache for position embeddings.
        """
        super().__init__(*args, **kwargs)

        logger.debug("Building custom RotaryEmbedding")

        self._target_device = torch.device("spyre")
        self._target_dtype = torch.float16
        
        # Build cos/sin cache on initialization
        self._build_cos_sin_cache()
        
        # Compile the forward kernel
        self._fwd = self.maybe_compile(self.forward_static)

        self._layer_name = register_layer(self, "spyre_rotary_embedding")

        logger.warning(
            "SpyreRotaryEmbedding: no dtype promotion is performed, "
            "expect numerical differences to upstream vLLM."
        )

    def _build_cos_sin_cache(self):
        """Build cos/sin cache for position embeddings.

        Creates frequency-based position encodings:
        - frequencies = base^(-2i/rotary_dim) for i in [0, rotary_dim/2)
        - cos_cache[pos] = cos(pos * frequencies)
        - sin_cache[pos] = sin(pos * frequencies)
        
        Note: Uses float16 directly (no dtype promotion) to match Spyre constraints,
        similar to SpyreRMSNorm implementation.
        """
        # Use float16 directly - no dynamic dimensions (Spyre constraint)
        compute_dtype = torch.float16
        
        # Compute inverse frequencies: base^(-2i/rotary_dim)
        # Using negative exponent for numerical stability
        exponents = -torch.arange(0, self.rotary_dim, 2, dtype=compute_dtype) / self.rotary_dim
        inv_freq = torch.pow(self.base, exponents)
        
        # Create position indices [0, 1, 2, ..., max_position_embeddings-1]
        t = torch.arange(self.max_position_embeddings, dtype=compute_dtype)
        
        # Compute frequencies for each position: pos * inv_freq
        # Shape: [max_position_embeddings, rotary_dim // 2]
        freqs = torch.outer(t, inv_freq)
        
        # Duplicate frequencies for interleaved pattern
        # Shape: [max_position_embeddings, rotary_dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # Compute cos and sin directly in float16, then move to device
        self.cos_cache = emb.cos().to(device=self._target_device)
        self.sin_cache = emb.sin().to(device=self._target_device)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass using custom op to bypass torch.compile.

        Delegates to torch.ops.vllm.spyre_rotary_embedding which retrieves this layer
        from forward_context.no_compile_layers and calls forward_impl outside
        the compilation graph.

        Args:
            positions: Position indices [batch_size, seq_len] or [total_tokens]
            query: Query tensor [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor [batch_size, seq_len, num_kv_heads, head_dim]

        Returns:
            Tuple of (rotated_query, rotated_key) with same shapes as inputs
        """
        rotated_query = torch.empty_like(query)
        rotated_key = torch.empty_like(key)

        # Custom op call - executes outside torch.compile graph
        torch.ops.vllm.spyre_rotary_embedding(
            positions, query, key, rotated_query, rotated_key, self._layer_name
        )

        return rotated_query, rotated_key

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input.

        Splits the last dimension in half and rotates:
        [x1, x2] -> [-x2, x1]

        Args:
            x: Input tensor [..., rotary_dim]

        Returns:
            Rotated tensor [..., rotary_dim]
        """
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    @staticmethod
    def forward_static(
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        rotary_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spyre-optimized RoPE computation compiled via torch.compile.

        Applies rotary position embeddings to query and key tensors:
        output = x * cos + rotate_half(x) * sin

        Args:
            positions: Position indices [batch_size, seq_len] or [total_tokens]
            query: Query tensor [batch_size, seq_len, num_heads, head_dim]
            key: Key tensor [batch_size, seq_len, num_kv_heads, head_dim]
            cos_cache: Cosine cache [max_position_embeddings, rotary_dim]
            sin_cache: Sine cache [max_position_embeddings, rotary_dim]
            rotary_dim: Dimension to apply rotation

        Returns:
            Tuple of (rotated_query, rotated_key)
        """
        # Retrieve cos/sin for the given positions
        # positions shape: [batch_size, seq_len] or [total_tokens]
        cos = cos_cache[positions]  # [..., rotary_dim]
        sin = sin_cache[positions]  # [..., rotary_dim]

        # Reshape cos/sin to match query/key dimensions
        # Need to add head dimension: [..., 1, rotary_dim]
        if cos.dim() == 2:  # [batch_size, seq_len, rotary_dim]
            cos = cos.unsqueeze(-2)  # [batch_size, seq_len, 1, rotary_dim]
            sin = sin.unsqueeze(-2)  # [batch_size, seq_len, 1, rotary_dim]
        elif cos.dim() == 1:  # [total_tokens, rotary_dim]
            cos = cos.unsqueeze(-2)  # [total_tokens, 1, rotary_dim]
            sin = sin.unsqueeze(-2)  # [total_tokens, 1, rotary_dim]

        # Apply rotation to query and key
        # Only rotate the first rotary_dim dimensions
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]
        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]

        # Apply RoPE: x * cos + rotate_half(x) * sin
        query_rot = query_rot * cos + SpyreRotaryEmbedding._rotate_half(query_rot) * sin
        key_rot = key_rot * cos + SpyreRotaryEmbedding._rotate_half(key_rot) * sin

        # Concatenate rotated and pass-through parts
        query = torch.cat([query_rot, query_pass], dim=-1)
        key = torch.cat([key_rot, key_pass], dim=-1)

        return query, key

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Spyre device execution with device transfer and dtype conversion.

        Handles Spyre-specific constraints:
            1. Device transfer: CPU -> Spyre, convert to float16
            2. Kernel execution: Calls compiled _fwd
            3. Result transfer: Spyre -> CPU, restore original dtype

        Args:
            positions: Position indices on CPU
            query: Query tensor on CPU
            key: Key tensor on CPU

        Returns:
            Tuple of (rotated_query, rotated_key) on CPU with original dtype
        """
        query_dtype = query.dtype
        query_device = query.device
        key_dtype = key.dtype
        key_device = key.device

        # Transfer cos/sin cache to Spyre device if not already there
        if self.cos_cache.device != self._target_device:
            self.cos_cache = convert(self.cos_cache, self._target_device, self._target_dtype)
            self.sin_cache = convert(self.sin_cache, self._target_device, self._target_dtype)

        # Execute compiled kernel on Spyre device
        rotated_query, rotated_key = self._fwd(
            convert(positions, self._target_device, torch.long),
            convert(query, self._target_device, self._target_dtype),
            convert(key, self._target_device, self._target_dtype),
            self.cos_cache,
            self.sin_cache,
            self.rotary_dim,
        )

        # Transfer back to CPU and restore original dtype
        rotated_query = convert(rotated_query, query_device, query_dtype)
        rotated_key = convert(rotated_key, key_device, key_dtype)

        return rotated_query, rotated_key


def _op_func(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    rotated_query: torch.Tensor,
    rotated_key: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op implementation — runs outside torch.compile graph."""
    layer = get_layer(layer_name)
    result_query, result_key = layer.forward_native(positions, query, key)
    rotated_query.copy_(result_query)
    rotated_key.copy_(result_key)


@lru_cache(maxsize=1)
def register():
    """Register the spyre_rotary_embedding custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_rotary_embedding",
        op_func=_op_func,
        mutates_args=["rotated_query", "rotated_key"],
        fake_impl=_fake_impl,
    )
    logger.info("Registered custom op: SpyreRotaryEmbedding")


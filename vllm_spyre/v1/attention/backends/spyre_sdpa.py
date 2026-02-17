"""Attention layer with simple KV caching without paging.
Uses SDPA for attention
"""

from dataclasses import dataclass

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionMetadata,
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.registry import register_backend, AttentionBackendEnum
from typing import ClassVar
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.logger import init_logger

logger = init_logger(__name__)


@register_backend(AttentionBackendEnum.CUSTOM)
class SpyreSDPABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [64]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type["SpyreSDPABackendImpl"]:
        return SpyreSDPABackendImpl

    @staticmethod
    def get_builder_cls() -> type["SpyreSDPAMetadataBuilder"]:
        return SpyreSDPAMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        # TODO: copied from flash attn, need to verify
        return head_size % 8 == 0 and head_size <= 256

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.ENCODER_ONLY


@dataclass
class SpyreSDPAMetadata(AttentionMetadata):
    prompt_padding: torch.Tensor
    padded_num_seqs: int
    padded_seq_len: int


class SpyreSDPAMetadataBuilder(AttentionMetadataBuilder[SpyreSDPAMetadata]):
    def __init__(
        self,
        kv_cache_spec: "AttentionSpec",
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> SpyreSDPAMetadata:
        assert not common_attn_metadata.causal, "Causal attention is not supported"

        padded_num_seqs = common_attn_metadata.query_start_loc.shape[0]

        ret = SpyreSDPAMetadata(
            prompt_padding=common_attn_metadata.query_start_loc,
            padded_num_seqs=padded_num_seqs,
            padded_seq_len=common_attn_metadata.num_actual_tokens // padded_num_seqs,
        )
        return ret


class SpyreSDPABackendImpl(AttentionImpl[SpyreSDPAMetadata]):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        assert num_kv_heads is not None
        self.num_kv_heads = num_kv_heads

        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Check for supported head sizes
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported.")
        if logits_soft_cap is not None:
            raise NotImplementedError("Logits soft cap is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        self.attn_type = attn_type
        if attn_type != AttentionType.ENCODER_ONLY:
            raise NotImplementedError(
                "Only Encoder self-attention is implemented for SpyreSDPABackendImpl"
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: SpyreSDPAMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [batch_size * num_tokens, num_heads * head_size]
            key: shape = [batch_size * num_tokens, num_kv_heads, head_size]
            value: shape = [batch_size * num_tokens, num_kv_heads, head_size]
            kv_cache = [] # disabled
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size * num_tokens, num_heads * head_size]
        """
        assert output is None
        assert kv_cache.numel() == 0, "Only encoder attention is supported"
        assert key is not None and value is not None
        bsize = attn_metadata.padded_num_seqs
        seq_len = attn_metadata.padded_seq_len

        # Reshape the query, key, and value tensors.
        query = query.view(bsize, seq_len, self.num_heads, self.head_size)
        key = key.view(bsize, seq_len, self.num_kv_heads, self.head_size)
        value = value.view(bsize, seq_len, self.num_kv_heads, self.head_size)

        query = query.transpose(2, 1)
        key = key.transpose(2, 1)
        value = value.transpose(2, 1)

        attn_output = self._sdpa_forward(query, key, value, attn_metadata)

        return attn_output.view(bsize * seq_len, self.num_heads * self.head_size)

    def _sdpa_forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attn_metadata
    ) -> torch.Tensor:
        _, nheads, qlen, _ = query.shape
        kvlen = key.shape[2]
        assert self.num_kv_heads == key.shape[1]

        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        mask_list = []

        idx = torch.arange(kvlen, device=key.device)
        for prompt_padding in attn_metadata.prompt_padding:
            mask = idx >= prompt_padding
            mask = mask.unsqueeze(0).expand(qlen, kvlen)
            mask_list.append(mask)

        masks = torch.stack(mask_list)
        masks = masks.unsqueeze(1)
        masks = masks.expand(-1, nheads, -1, -1)

        out = scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=False,
            scale=self.scale,
            dropout_p=0.0,
            attn_mask=masks,
        )

        out = out.transpose(2, 1).contiguous()
        return out

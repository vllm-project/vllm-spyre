""" Attention layer with simple KV caching without paging.
Uses SDPA for attention
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Type

import torch
from torch.nn.functional import scaled_dot_product_attention

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType,
                                              is_quantized_kv_cache)

from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable
from vllm.attention.ops.paged_attn import PagedAttention, PagedAttentionMetadata
from vllm.logger import init_logger

logger = init_logger(__name__)

class SpyreSDPABackend(AttentionBackend):
    accept_output_buffer: bool = False
    @staticmethod
    def get_name() -> str:
        return "Spyre_SDPA"

    @staticmethod
    def get_impl_cls() -> Type["SpyreSDPABackendImpl"]:
        return SpyreSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["SpyreSDPAMetadata"]:
        return SpyreSDPAMetadata

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_builder_cls() -> Type["SpyreSDPAMetadataBuilder"]:
        return SpyreSDPAMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
            num_blocks: int,
            block_size: int,
            num_kv_heads: int,
            head_size: int,
    ) -> Tuple[int, ...]:
        # No paging, taking max seq len of 128
        max_seq_len = 128
        max_batch_size = 1
        return (2, max_batch_size, num_kv_heads, max_seq_len, head_size)

@dataclass
class SpyreSDPAMetadata():
    num_tokens = 0
    bsize = 1

    # For prefill
    is_prefill: bool

    # For decode
    past_token = 0

    def __init__(self, num_tokens, bsize, is_prefill, masks, past_token):
        self.num_tokens = num_tokens
        self.bsize = bsize
        self.is_prefill = is_prefill
        self.past_token = past_token
        self.masks = masks

class SpyreSDPAMetadataBuilder(AttentionMetadataBuilder[SpyreSDPAMetadata]):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(self, data) -> SpyreSDPAMetadata:
        num_tokens = len(data.input_tokens)
        is_prefill = data.is_prompt
        bsize = len(data.input_positions)
        past_token = 0
        # TODO: Handle batch sizes later
        if not is_prefill:
            past_token = data.input_masks.shape[2] - 1 # bsize x qsize x kvsize

        masks = data.input_masks

        attn_metadata = SpyreSDPAMetadata(
            num_tokens=num_tokens,
            bsize=bsize,
            is_prefill=is_prefill,
            past_token=past_token,
            masks=masks
        )

        return attn_metadata

class SpyreSDPABackendImpl(AttentionImpl[SpyreSDPAMetadata]):
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]] = None,
            logits_soft_cap: Optional[float] = None,
            attn_type: str = AttentionType.DECODER,
            use_irope: bool = False,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.sliding_window = sliding_window
        self.logits_soft_cap = logits_soft_cap
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # Check for supported head sizes
        if alibi_slopes is not None:
            raise NotImplementedError("Alibi slopes is not supported.")
        if kv_cache_dtype != "auto":
            raise NotImplementedError("FP8 KV cache dtype is not supported.")
        if blocksparse_params is not None:
            raise NotImplementedError("Blocksparse is not supported.")
        self.attn_type = attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "SpyreSDPABackendImpl")

    def _add_kv_cache(
            self,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: SpyreSDPAMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            keys: shape = [bsize, num_tokens, num_kv_heads, head_size]
            values: shape = [bsize, num_tokens, num_kv_heads, head_size]
            kv_cache = [2, bsize, num_tokens, num_kv_heads, head_size]
        Returns:
            shape = [bsize, num_tokens, num_kv_heads, head_size] * 2
        """
        if kv_cache.dtype != key.dtype:
            kv_cache = kv_cache.to(dtype=key.dtype)

        past_token = attn_metadata.past_token
        if attn_metadata.is_prefill:
            key_result, value_result = key, value
        else:
            keys = kv_cache[0,:,:,:past_token,:] # bsize x kv_heads x qlen x head_dim for fms
            values = kv_cache[1,:,:,:past_token,:]
            key_result = torch.cat((keys, key), dim=2) 
            value_result = torch.cat((values, value), dim=2)

        kv_len = key.shape[2] 
        kv_cache[0,:,:,past_token:past_token + kv_len,:] = key
        kv_cache[1,:,:,past_token:past_token + kv_len,:] = value

        return key_result, value_result, kv_cache

    def forward(
            self,
            layer: AttentionLayer,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: SpyreSDPAMetadata,  # type: ignore
            output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with torch SDPA and PagedAttention.

        Args:
            query: shape = [batch_size, num_tokens, num_heads * head_size]
            key: shape = [batch_size * num_tokens, num_kv_heads, head_size]
            value: shape = [batch_size * num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        bsize = attn_metadata.bsize
        # Reshape the query, key, and value tensors.
        query = query.view(bsize, -1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(bsize, -1, self.num_kv_heads, self.head_size)
            value = value.view(bsize, -1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        # For fms
        query = query.transpose(2, 1) # bsize x n_heads x qlen x kv_len
        key = key.transpose(2, 1) # bsize x num_kv_heads x qlen x kv_len
        value = value.transpose(2, 1) # bsize x num_kv_heads x qlen x kv_len

        keys, values, new_kv_cache = self._add_kv_cache(key, value, kv_cache, attn_metadata)
        kv_cache.copy_(new_kv_cache)
        attn_output = self._sdpa_forward(query, keys, values, attn_metadata)

        # Reshape the output tensor.
        attn_output = attn_output.view(-1, self.num_heads * self.head_size)
        return attn_output

    def _sdpa_forward(
            self,
            query: torch.Tensor, # bsize x nheads x qlen x head_dim
            key: torch.Tensor, # bsize x num_kv_heads x qlen x kv_len
            value: torch.Tensor, # bsize x num_kv_heads x qlen x kv_len
            attn_metadata
    ) -> torch.Tensor: # (bsize x qlen) x nheads x kv_len

        bsize, nheads, qlen, _ = query.shape
        assert self.num_kv_heads == key.shape[1]

        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Handle batchsize here
        query = query.squeeze(0)
        key = key.squeeze(0)
        value = value.squeeze(0)

        masks = attn_metadata.masks
        if masks.dtype != query.dtype:
            masks = masks.to(dtype=query.dtype)
        out = scaled_dot_product_attention(
            query[None,:,:,:],
            key[None,:,:,:],
            value[None,:,:,:],
            is_causal=False,
            scale=self.scale,
            dropout_p=0.0,
            attn_mask=masks[None,:,:,:],
        ) # bsize x nheads x qlen x kv_len

        # Need to return qlen x (nheads x kv_len)
        # TODO: change for bsize > 1
        out = out.transpose(2, 1).squeeze(0).contiguous()
        return out

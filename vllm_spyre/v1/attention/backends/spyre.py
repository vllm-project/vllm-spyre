""" Attention layer with torch scaled_dot_product_attention
and PagedAttention. Modified to work with spyre
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
    accept_output_buffer: bool = True
    @staticmethod
    def get_name() -> str:
        return "Spyre_SDPA"

    @staticmethod
    def get_impl_cls() -> Type["SpyreSDPABackendImpl"]:
        return SpyreSDPABackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
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
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

@dataclass
class SpyreSDPAMetadata(AttentionMetadata, PagedAttentionMetadata):
    seq_lens: Optional[List[int]] = None  # Non-chunked prefill
    seq_lens_tensor: Optional[List[int]]

    @property
    def prefill_metadata(self) -> Optional["SpyreSDPAMetadata"]:
        if self.num_prefill_tokens == 0:
            return None
        return self

    @property
    def decode_metadata(self) -> Optional["SpyreSDPAMetadata"]:
        if self.num_decode_tokens == 0:
            return None
        return self

    def set_prefill(num_prefills, num_prefill_tokens, slot_mapping):
        self.num_prefills = num_prefills
        self.num_prefill_tokens = num_prefill_tokens
        self.slot_mapping = slot_mapping

    def set_decode(num_decode_tokens, slot_mapping):
        self.num_decode_tokens = num_decode_tokens
        self.slot_mapping = slot_mapping

class SpyreSDPAMetadataBuilder(AttentionMetadataBuilder[SpyreSDPAMetadata]):
    def __init__(self):
        pass

    def prepare(self):
        pass

    def build(self, data) -> SpyreSDPAMetadata:
        seq_lens = [len(d) for d in data.input_tokens]
        num_prefills = len(data.input_tokens) if data.is_prompt else 0
        num_prefill_tokens = sum(seq_lens) if data.is_prompt else 0
        num_decode_tokens = sum(seq_lens) if not data.is_prompt else 0
        max_decode_seq_len = max(seq_lens) if not data.is_prompt else 0

        print("seq_lens: ", seq_lens)

        prefill_seq_lens = seq_lens[0:num_prefills] 

	# For paged attention
        if num_decode_tokens != 0:
            seq_lens_tensor = torch.tensor(
                seq_lens[num_prefills:],
                dtype=torch.int32,
                device="cpu",
            )
            block_tables = None # make_tensor_with_pad( None, pad=0, dtype=torch.int32, device="cpu",)
        else:
            block_tables = torch.tensor([])
            seq_lens_tensor = torch.tensor(
                seq_lens[:num_prefills],
                dtype=torch.int32,
                device="cpu",
            )

        attn_metadata = SpyreSDPAMetadata(
                seq_lens=prefill_seq_lens,
                seq_lens_tensor=seq_lens_tensor,
                num_prefills=num_prefills,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                block_tables=block_tables,
                slot_mapping=None,
                max_decode_seq_len=max_decode_seq_len,
                multi_modal_placeholder_index_maps=None,
                enable_kv_scales_calculation=False,
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
        supported_head_sizes = PagedAttention.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}.")

        if is_quantized_kv_cache(kv_cache_dtype) and not _use_ipex:
            raise NotImplementedError(
                "Spyre SDPA backend FP8 KV cache requires "
                "intel_extension_for_pytorch support.")
        self.attn_type = attn_type
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "SpyreSDPABackendImpl")

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
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        attn_type = self.attn_type

        assert output is not None, "Output tensor must be provided"
        if attn_metadata is None:
            # profiling run
            return output

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        if key is not None:
            assert value is not None
            key = key.view(-1, self.num_kv_heads, self.head_size)
            value = value.view(-1, self.num_kv_heads, self.head_size)
        else:
            assert value is None

        key_cache = kv_cache
        value_cache = kv_cache
        #print("kv cache: ", kv_cache.shape)
        if kv_cache.numel() > 0:
            key_cache, value_cache = PagedAttention.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size)

            if (key is not None) and (value is not None):
                PagedAttention.write_to_paged_cache(
                    key, value, key_cache, value_cache, attn_metadata.slot_mapping,
                    self.kv_cache_dtype, layer._k_scale, layer._v_scale)

        # Decoder self-attention supports chunked prefill.
        # Encoder/decoder cross-attention requires no chunked
        # prefill (100% prefill or 100% decode tokens, no mix)
        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens

        if attn_type == AttentionType.DECODER:
            # Only enforce this shape-constraint for decoder
            # self-attention
            assert key.shape[0] == num_prefill_tokens + num_decode_tokens
            assert value.shape[0] == num_prefill_tokens + num_decode_tokens

        output = torch.empty_like(query)
        if prefill_meta := attn_metadata.prefill_metadata:
            assert attn_metadata.seq_lens is not None
            self._run_sdpa_forward(output,
				   query,
				   key,
				   value,
				   prefill_meta,
				   attn_type=attn_type)

        if decode_meta := attn_metadata.decode_metadata:
            assert attn_type != AttentionType.ENCODER_ONLY, (
                "Encoder-only models should not have decode metadata.")
            # Decoding run.
            seq_lens_arg = attn_metadata.seq_lens_tensor
            max_seq_len_arg = attn_metadata.max_decode_seq_len
            block_tables_arg = attn_metadata.block_tables

            PagedAttention.forward_decode(
                output[attn_metadata.num_prefill_tokens:, :, :],
                query[attn_metadata.num_prefill_tokens:, :, :],
                key_cache,
                value_cache,
                block_tables_arg,
                seq_lens_arg,
                max_seq_len_arg,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
		None,
                layer._k_scale,
                layer._v_scale,
            )

        # Reshape the output tensor.
        return output.view(-1, self.num_heads * self.head_size)

    def _run_sdpa_forward(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: SpyreSDPAMetadata,
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        if self.num_kv_heads != self.num_heads:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        seq_lens = attn_metadata.seq_lens
        attn_masks = [None] * len(seq_lens)
        #attn_metadata.set_attn_bias(attn_masks, attn_type)

        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        causal_attn = (attn_type == AttentionType.DECODER)

        seq_lens_q = attn_metadata.seq_lens
        seq_lens_kv = attn_metadata.seq_lens
        start_q, start_kv = 0, 0
        for seq_len_q, seq_len_kv, mask in zip(seq_lens_q, seq_lens_kv,
                                               attn_masks):
            end_q = start_q + seq_len_q
            end_kv = start_kv + seq_len_kv
            sub_out = scaled_dot_product_attention(
                query[None, :, start_q:end_q, :],
                key[None, :, start_kv:end_kv, :],
                value[None, :, start_kv:end_kv, :],
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=causal_attn and mask is None,
                scale=self.scale).squeeze(0).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = sub_out
            start_q, start_kv = end_q, end_kv


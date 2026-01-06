"""Utilities for selecting and loading Spyre models."""

import os
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn as nn
from fms.models import get_model
from transformers import PretrainedConfig
from vllm.config import ModelConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

import vllm_spyre.envs as envs_spyre
import vllm_spyre.utils as utils_spyre
from vllm_spyre.platform import SpyrePlatform

try:
    import backends.dynamo_tracer  # noqa: F401
except ImportError:
    print("WARNING: Disabled: dynamo_tracer")
    pass

BACKEND_LIST = ["sendnn", "inductor"]

logger = init_logger(__name__)


@dataclass
class SpyreAttentionMetadata:
    slot_mapping: torch.Tensor = None
    current_tkv_mask: torch.Tensor = None
    left_padded_prompt_mask: torch.Tensor = None
    block_table: torch.Tensor = None
    is_prefill: bool = False
    # We need this indices because when requests are removed from the
    # persistent batch, we need to keep the reference of the remaining
    # requests, that is, this index must be the same from the prefill until
    # the end.
    scale_indices: torch.Tensor = None


class SpyreCausalLM(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        max_prompt_length: int,
        max_decode_length: int,
        rank: int,
    ) -> None:
        super().__init__()

        try:
            ## Temporary backwards compatibility for 0.10.2
            from vllm.model_executor.layers.sampler import get_sampler

            self.sampler = get_sampler()
        except (ImportError, ModuleNotFoundError):
            self.sampler = Sampler()

        # boolean tensor of length batch size with indices:
        # True for unfinished sequences and
        # False for finished or padded sequences
        self.indices = None

        # number of right pads (relevant for continuous batching only)
        self.n_pads_right = 0

        self._mask_dtype = (
            torch.float16 if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn" else torch.float32
        )

        # FMS Model
        if envs_spyre.VLLM_SPYRE_USE_CB:
            self.model = ContinuousBatchingFmsModel(vllm_config, rank)
        else:
            self.model = StaticBatchingFmsModel(
                vllm_config,
                max_prompt_length,
                max_decode_length,
                rank,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:
        if is_prompt and not envs_spyre.VLLM_SPYRE_USE_CB:
            self.model.past_key_value_states = None  # type: ignore

        extra_kwargs: dict[str, Any] = {}
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn":
            # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash
            # cpu impl when padding too much
            extra_kwargs["attn_algorithm"] = "math"

        # normal prefill or decoding step
        logits = self.model(
            input_ids,
            position_ids=positions,
            mask=masks,
            use_cache=True,
            is_prompt=is_prompt,
            **extra_kwargs,
        )

        if envs_spyre.VLLM_SPYRE_USE_CB:
            if is_prompt and self.n_pads_right > 0:
                # get last token before the right padding
                logits = logits[self.indices, -self.n_pads_right - 1, :]
            else:
                # just take last token if no right padding
                logits = logits[self.indices, -1, :]
        else:
            # removing finished or padded sequences
            logits = logits[self.indices]

        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput | None:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def get_mask_dtype(self) -> torch.dtype:
        return self._mask_dtype


class FmsModelBase(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        max_prompt_length: int,
        max_decode_length: int,
        rank: int,
        sendnn_dynamic: bool,
    ) -> None:
        super().__init__()

        self.config: PretrainedConfig = vllm_config.model_config.hf_config

        # Actual FMS model
        self.model: nn.Module
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.dtype = self.get_dtype()

        # Load the weights from the cached or downloaded files.
        self.load_weights(
            model_config=self.model_config,
            max_prompt_length=max_prompt_length,
            max_decode_length=max_decode_length,
            distributed_strategy="tp" if self.parallel_config.world_size > 1 else None,
            sendnn_dynamic=sendnn_dynamic,
            rank=rank,
            world_size=self.parallel_config.world_size,
        )

    def load_weights(
        self,
        model_config: ModelConfig,
        max_prompt_length: int,
        max_decode_length: int,
        distributed_strategy: str | None,
        sendnn_dynamic: bool,
        **kwargs,
    ) -> None:
        logger.debug("Loading model weights for model %s", model_config.model)
        logger.debug("Model config has dtype: %s", model_config.dtype)

        # When using quantized models, we might not be using the
        # model_config's dtype, hence we don't log the msg below
        # since it might confuse the user
        if model_config.quantization:
            logger.debug("Quantized model found with quantization : %s", model_config.quantization)
        else:
            if self.dtype is not model_config.dtype:
                logger.info(
                    "Ignoring user-provided dtype=%s (provided either through"
                    " --dtype CLI arg or model_config.dtype) and using"
                    " dtype=%s instead.",
                    model_config.dtype,
                    self.dtype,
                )

        is_local = os.path.isdir(model_config.model)
        model_path = model_config.model
        # Get location of model from HF cache.
        if not is_local:
            model_path = download_weights_from_hf(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=["*.safetensors", "*.bin", "*.pt"],
                revision=model_config.revision,
            )

        with utils_spyre.stagger_region(
            envs_spyre.VLLM_SPYRE_MAX_LOAD_PROCESSES,
            kwargs["world_size"],
            kwargs["rank"],
        ):
            self.model = get_model(
                architecture="hf_pretrained",
                model_path=model_path,
                distributed_strategy=distributed_strategy,
                group=dist.group.WORLD,
                fused_weights=False,
            )

        self.model.eval()
        torch.set_grad_enabled(False)

        _target_cache_size = max(int(max_decode_length * 2), int(max_prompt_length * 2.5))
        if (
            hasattr(torch._dynamo.config, "accumulated_cache_size_limit")
            and _target_cache_size > torch._dynamo.config.accumulated_cache_size_limit
        ):
            _prev = torch._dynamo.config.accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = _target_cache_size
            logger.info(
                "NOTICE: Adjusting "
                "torch._dynamo.config.accumulated_cache_size_limit "
                "from %s to %s "
                "to accommodate prompt size of %d "
                "and decode tokens of %d",
                _prev,
                torch._dynamo.config.accumulated_cache_size_limit,
                max_prompt_length,
                max_decode_length,
            )

        if _target_cache_size > torch._dynamo.config.cache_size_limit:
            _prev = torch._dynamo.config.cache_size_limit
            torch._dynamo.config.cache_size_limit = _target_cache_size
            logger.info(
                "NOTICE: Adjusting torch._dynamo.config.cache_size_limit "
                "from %s to %s "
                "to accommodate prompt size of %d "
                "and decode tokens of %d",
                _prev,
                torch._dynamo.config.accumulated_cache_size_limit,
                max_prompt_length,
                max_decode_length,
            )

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            # When running on Spyre cards for either non-quantized (bf16) models
            # or quantized (fp8) models, we cast any bf16 params down
            self._cast_bf16_to_f16()
            options = {"sendnn.dynamic": True} if sendnn_dynamic else {}

            # Lazy import to avoid load torch_sendnn runtime before it is really
            # necessary. This solve issues of running forked tests that share
            # some resources from parent to children which can have problems
            # of caching even though the test run in isolated subprocesses.

            if SpyrePlatform.sendnn_configured():
                from torch_sendnn import torch_sendnn  # noqa: F401

            self.model = torch.compile(
                self.model,
                backend=envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND,
                options=options,
            )
        else:
            # CPU execution
            # For continuous batching w/ paged attention, we only support either
            # fp32 or fp8, not f16 or bf16.
            if not model_config.quantization:
                assert self.dtype == torch.float32
                self._cast_to_f32()

        logger.debug("Model weights loaded successfully.")

    def _cast_bf16_to_f16(self):
        """Cast all bf16 params in the model to f16."""
        for name, param in self.model.named_parameters():
            if param.dtype == torch.bfloat16:
                logger.debug(
                    "You are casting param %s to fp16, which"
                    " will cause loss of accuracy. This is required for"
                    " spyre cards that don't support bf16. You can ignore"
                    " this warning if this is intended.",
                    name,
                )
                param.data = param.data.to(dtype=torch.float16)

    def _cast_to_f32(self):
        """Cast model parameters to f32."""
        for name, param in self.model.named_parameters():
            logger.debug(
                "Casting param %s to fp32. This is required"
                " for attention implementations that only support"
                " full precision.",
                name,
            )
            param.data = param.data.to(dtype=torch.float32)


class ContinuousBatchingFmsModel(FmsModelBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        rank: int,
    ) -> None:
        BLOCK_SIZE = SpyrePlatform.get_block_size()
        max_model_len = vllm_config.model_config.max_model_len

        # edge case: prompt fills model length: can produce 1 token with prefill
        max_prompt_length = max_model_len
        # edge case: prompt will be padded to first block:
        # can produce 1 token with prefill plus rest of model length
        max_decode_length = max_model_len - BLOCK_SIZE + 1

        super().__init__(
            vllm_config, max_prompt_length, max_decode_length, rank, sendnn_dynamic=True
        )

        self.prefill_past_key_values = None

        # physical KV cache on AIU Spyre: will eventually not live in this class
        self.kv_cache_specs = {}
        self.kv_cache_specs["block_size"] = BLOCK_SIZE
        self.kv_cache_specs["num_kv_heads"] = self.model_config.get_num_kv_heads(
            self.parallel_config
        )

        if self.config.model_type in {"llama", "granite", "granitemoehybrid"}:
            self.kv_cache_specs["num_layers"] = self.config.num_hidden_layers
            self.kv_cache_specs["head_dim"] = getattr(
                self.model.config,
                "head_dim",
                self.config.hidden_size // self.config.num_attention_heads,
            )
        elif self.config.model_type == "gpt_bigcode":
            self.kv_cache_specs["num_layers"] = self.config.n_layer
            self.kv_cache_specs["head_dim"] = self.config.n_embd // self.config.n_head
        else:
            raise NotImplementedError(
                f"[SpyreCausalLM] model type {self.config.model_type} "
                f"not supported in ContinuousBatchingFmsModel"
            )

        if self.model_config.quantization:
            self.attention_name = "spyre_paged_attn_fp8"
            self.is_fp8_model = True
        else:
            self.attention_name = "spyre_paged_attn"
            self.is_fp8_model = False

        self.current_scale: list[tuple] | None = None

    def set_past_key_value_states(self, num_blocks) -> None:
        # List[layers] of Tuple[k,v] of
        # Tensor[num_blocks, block_size, num_kv_heads, head_dim]
        if not self.model_config.quantization:
            self.past_key_value_states = [
                (
                    torch.zeros(
                        num_blocks,
                        self.kv_cache_specs["block_size"],
                        self.kv_cache_specs["num_kv_heads"],
                        self.kv_cache_specs["head_dim"],
                        dtype=self.dtype,
                    ),
                    torch.zeros(
                        num_blocks,
                        self.kv_cache_specs["block_size"],
                        self.kv_cache_specs["num_kv_heads"],
                        self.kv_cache_specs["head_dim"],
                        dtype=self.dtype,
                    ),
                )
                for _ in range(self.kv_cache_specs["num_layers"])
            ]
        else:
            from fms_mo.aiu_addons.fp8.fp8_utils import ScaledTensor

            batch_size = max(2, self.scheduler_config.max_num_seqs)
            self.past_key_value_states = [
                (
                    ScaledTensor(
                        torch.zeros(
                            num_blocks,
                            self.kv_cache_specs["block_size"],
                            self.kv_cache_specs["num_kv_heads"],
                            self.kv_cache_specs["head_dim"],
                            dtype=self.dtype,
                        ),
                        scale=torch.tensor([1.0] * batch_size, dtype=torch.float32),
                        scaled=False,
                    ),
                    ScaledTensor(
                        torch.zeros(
                            num_blocks,
                            self.kv_cache_specs["block_size"],
                            self.kv_cache_specs["num_kv_heads"],
                            self.kv_cache_specs["head_dim"],
                            dtype=self.dtype,
                        ),
                        scale=torch.tensor([1.0] * batch_size, dtype=torch.float32),
                        scaled=False,
                    ),
                )
                for _ in range(self.kv_cache_specs["num_layers"])
            ]
            # This list keep the reference of scales of the quantized weights
            # that will be updated after model execution
            self.current_kv_scales = [
                (k_cache._scale, v_cache._scale) for k_cache, v_cache in self.past_key_value_states
            ]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool,
        is_prompt: bool,
        **extra_kwargs,
    ) -> torch.Tensor:
        forward_context = get_forward_context()

        attn_metadata = cast(SpyreAttentionMetadata, forward_context.attn_metadata)
        assert attn_metadata is not None
        # import will be not be needed/ handled by FMS soon
        import fms.utils.spyre.paged  # noqa # pylint: disable=unused-import

        # specify attention type for continuous batching
        extra_kwargs["attn_name"] = self.attention_name

        if self.is_fp8_model:
            # set scale for kv_cache
            self._set_scale_for_fp8(attn_metadata)

            # Adjust decode for bs=1 if needed
            input_ids, position_ids, attn_metadata = self._adjust_input_for_fp8(
                input_ids=input_ids, position_ids=position_ids, attn_metadata=attn_metadata
            )

        # Run the model
        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=self.past_key_value_states,
            use_cache=use_cache,
            last_n_tokens=SpyrePlatform.get_block_size() if is_prompt else 1,
            current_tkv_mask=attn_metadata.current_tkv_mask,
            left_padded_prompt_mask=attn_metadata.left_padded_prompt_mask,
            block_table=attn_metadata.block_table,
            slot_mapping=attn_metadata.slot_mapping,
            **extra_kwargs,
        )

        logits, self.past_key_value_states = output

        if is_prompt:
            # assert that indeed received the last block of logits
            assert logits.shape[1] == SpyrePlatform.get_block_size()

        if self.is_fp8_model:
            # update scale for kv_cache after execute model
            self._update_scale_for_fp8(attn_metadata)

            logits = self._adjust_output_for_fp8(logits, attn_metadata)

        return logits

    def _set_scale_for_fp8(self, attn_metadata: SpyreAttentionMetadata):
        for layer_idx, (k, v) in enumerate(self.past_key_value_states):
            if attn_metadata.is_prefill:
                # NOTE: Currently, prefill is only for a single prompt
                # In prefill, we restore the scale (no scale) and
                # reset to 1.
                assert len(attn_metadata.scale_indices) == 1
                prefill_index = attn_metadata.scale_indices[0]
                k._scale = self.current_kv_scales[layer_idx][0][prefill_index] = torch.ones(
                    1, dtype=torch.float32
                )
                v._scale = self.current_kv_scales[layer_idx][1][prefill_index] = torch.ones(
                    1, dtype=torch.float32
                )
                k._scaled = False
                v._scaled = False
            elif len(attn_metadata.scale_indices) == 1:
                # Decode
                # Special case for decode of bs=1, pad the batch to be bs=2
                dec_index = attn_metadata.scale_indices[0]
                k._scale = self.current_kv_scales[layer_idx][0][dec_index].repeat(2)
                v._scale = self.current_kv_scales[layer_idx][1][dec_index].repeat(2)

            else:
                # Set scale only for the requests of the batch
                k._scale = self.current_kv_scales[layer_idx][0][
                    attn_metadata.scale_indices
                ].reshape(-1)
                v._scale = self.current_kv_scales[layer_idx][1][
                    attn_metadata.scale_indices
                ].reshape(-1)

            # We set dynamic only for the first dimension of scale
            # during decoding
            is_dynamic_flag = 0 if attn_metadata.is_prefill else 1

            torch._dynamo.mark_dynamic(v._scale, is_dynamic_flag)
            torch._dynamo.mark_dynamic(k._scale, is_dynamic_flag)

    def _update_scale_for_fp8(self, attn_metadata: SpyreAttentionMetadata):
        for layer_idx, (k, v) in enumerate(self.past_key_value_states):
            if attn_metadata.is_prefill or len(attn_metadata.scale_indices) > 1:
                self.current_kv_scales[layer_idx][0][attn_metadata.scale_indices] = k._scale
                self.current_kv_scales[layer_idx][1][attn_metadata.scale_indices] = v._scale
            else:
                # if we did the padding, then we need to update only the scale
                # for the decoding index
                self.current_kv_scales[layer_idx][0][attn_metadata.scale_indices[0]] = k._scale[0]
                self.current_kv_scales[layer_idx][1][attn_metadata.scale_indices[0]] = v._scale[0]

    def get_dtype(self) -> torch.dtype:
        # Get the model's data type
        # This should be:
        # FP32 for un-quantized models on cpu
        # FP16 for un-quantized models on spyre
        # FP8 (float8_e4m3fn) for quantized models
        # (only fp8 quantization is supported)
        if self.model_config.quantization:
            return torch.float8_e4m3fn
        else:
            if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
                return torch.float16
            else:
                return torch.float32

    # TODO: this is not the best place to do. But we expect this to
    # be temporary and here should be easy to remove later
    def _adjust_input_for_fp8(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: SpyreAttentionMetadata,
    ):
        # NOTE: We only need to adjust the inputs for decode with
        # batch_size=2
        if attn_metadata.is_prefill or input_ids.shape[0] > 1:
            return input_ids, position_ids, attn_metadata

        input_ids = input_ids.repeat(2, 1)
        position_ids = position_ids.repeat(2, 1)
        attn_metadata = SpyreAttentionMetadata(
            slot_mapping=attn_metadata.slot_mapping.repeat(2, 1),
            current_tkv_mask=attn_metadata.current_tkv_mask.repeat(2),
            left_padded_prompt_mask=attn_metadata.left_padded_prompt_mask.repeat(2),
            block_table=attn_metadata.block_table.repeat(2, 1),
            is_prefill=attn_metadata.is_prefill,
            # NOTE: we don't change here, because we'll need this untouched
            # when we update the the scale after run the model
            scale_indices=attn_metadata.scale_indices,
        )
        return input_ids, position_ids, attn_metadata

    def _adjust_output_for_fp8(self, logits: torch.Tensor, attn_metadata: SpyreAttentionMetadata):
        if attn_metadata.is_prefill or len(attn_metadata.scale_indices) > 1:
            # skip for prefill or decode for bs>1
            return logits

        return logits[0].unsqueeze(0)


class StaticBatchingFmsModel(FmsModelBase):
    def __init__(
        self,
        vllm_config: VllmConfig,
        max_prompt_length: int,
        max_decode_length: int,
        rank: int,
    ) -> None:
        super().__init__(
            vllm_config, max_prompt_length, max_decode_length, rank, sendnn_dynamic=False
        )

        # dynamic KV cache
        self.past_key_value_states = None

        if self.model_config.quantization:
            self.attention_name = "math_fp8"
        else:
            self.attention_name = "sdpa_causal"

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool,
        **extra_kwargs,
    ) -> torch.Tensor:
        # specify attention type for static batching
        extra_kwargs["attn_name"] = self.attention_name

        # In order to calculate prompt logprobs, we have to return the
        # hidden states from the whole prompt. The static graphs need to be
        # compiled with this set one way or the other.
        last_n_tokens = 0 if envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS else 1

        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=self.past_key_value_states,
            use_cache=use_cache,
            last_n_tokens=last_n_tokens,
            **extra_kwargs,
        )

        logits, self.past_key_value_states = output

        if not envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS:
            logits = logits.squeeze(1)

        return logits

    def get_dtype(self) -> torch.dtype:
        # For static batching, we set fp16 on spyre and fp32 on cpu
        # (This applies even when running fp8 quantized models)
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            return torch.float16
        else:
            return torch.float32

"""Utilities for selecting and loading Spyre models."""
import os
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn as nn
from fms.models import get_model
from transformers import PretrainedConfig
from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf)
from vllm.model_executor.sampling_metadata import SamplingMetadata

import vllm_spyre.envs as envs_spyre
import vllm_spyre.utils as utils_spyre
from vllm_spyre.platform import SpyrePlatform

try:
    import backends.dynamo_tracer  # noqa: F401
except ImportError:
    print("WARNING: Disabled: dynamo_tracer")
    pass

BACKEND_LIST = ['sendnn', 'inductor']

logger = init_logger(__name__)


@dataclass
class SpyreAttentionMetadata:
    slot_mapping: torch.Tensor = None
    current_tkv_mask: torch.Tensor = None
    left_padded_prompt_mask: torch.Tensor = None
    block_table: torch.Tensor = None


class SpyreCausalLM(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        max_prompt_length: int,
        max_decode_length: int,
        rank: int,
    ) -> None:
        super().__init__()

        self.logits_processor = LogitsProcessor(
            model_config.hf_config.vocab_size, logits_as_input=True)
        self.sampler = get_sampler()

        # boolean tensor of length batch size with indices:
        # True for unfinished sequences and
        # False for finished or padded sequences
        self.indices = None

        # number of right pads (relevant for continuous batching only)
        self.n_pads_right = 0

        # FMS Model
        if envs_spyre.VLLM_SPYRE_USE_CB:
            self.model = ContinuousBatchingFmsModel(model_config,
                                                    parallel_config,
                                                    scheduler_config, rank)
        else:
            self.model = StaticBatchingFmsModel(
                model_config,
                parallel_config,
                scheduler_config,
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

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens


class FmsModelBase(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_prompt_length: int,
        max_decode_length: int,
        rank: int,
        sendnn_dynamic: bool,
    ) -> None:
        super().__init__()

        self.config: PretrainedConfig = model_config.hf_config

        # Actual FMS model
        self.model: nn.Module
        self.model_config = model_config
        self.dtype = self.get_dtype()

        # Load the weights from the cached or downloaded files.
        self.load_weights(
            model_config=model_config,
            max_prompt_length=max_prompt_length,
            max_decode_length=max_decode_length,
            distributed_strategy="tp"
            if parallel_config.world_size > 1 else None,
            sendnn_dynamic=sendnn_dynamic,
            rank=rank,
            world_size=parallel_config.world_size,
        )

    def load_weights(
        self,
        model_config: ModelConfig,
        max_prompt_length: int,
        max_decode_length: int,
        distributed_strategy: Optional[str],
        sendnn_dynamic: bool,
        **kwargs,
    ) -> None:

        if self.dtype is not model_config.dtype:
            logger.info(
                "Ignoring user-provided dtype=%s and using dtype=%s instead.",
                model_config.dtype, self.dtype)

        is_local = os.path.isdir(model_config.model)
        model_path = model_config.model
        # Get location of model from HF cache.
        if not is_local:
            model_path = download_weights_from_hf(
                model_name_or_path=model_path,
                cache_dir=None,
                allow_patterns=["*.safetensors", "*.bin", "*.pt"],
                revision=model_config.revision)

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

        _target_cache_size = max(int(max_decode_length * 2),
                                 int(max_prompt_length * 2.5))
        if hasattr(torch._dynamo.config, "accumulated_cache_size_limit") and \
            _target_cache_size > torch._dynamo.config.\
            accumulated_cache_size_limit:
            _prev = torch._dynamo.config.accumulated_cache_size_limit
            torch._dynamo.config.accumulated_cache_size_limit = \
                _target_cache_size
            logger.info(
                "NOTICE: Adjusting "
                "torch._dynamo.config.accumulated_cache_size_limit "
                "from %s to %s "
                "to accommodate prompt size of %d "
                "and decode tokens of %d", _prev,
                torch._dynamo.config.accumulated_cache_size_limit,
                max_prompt_length, max_decode_length)

        if _target_cache_size > torch._dynamo.config.cache_size_limit:
            _prev = torch._dynamo.config.cache_size_limit
            torch._dynamo.config.cache_size_limit = _target_cache_size
            logger.info(
                "NOTICE: Adjusting torch._dynamo.config.cache_size_limit "
                "from %s to %s "
                "to accommodate prompt size of %d "
                "and decode tokens of %d", _prev,
                torch._dynamo.config.accumulated_cache_size_limit,
                max_prompt_length, max_decode_length)

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            # When running on Spyre cards for either non-quantized (bf16) models
            # or quantized (fp8) models, we cast any bf16 params down
            self._cast_bf16_to_f16()
            options = {"sendnn.dynamic": True} if sendnn_dynamic else {}

            # Lazy import to avoid load torch_sendnn runtime before it is really
            # necessary. This solve issues of running forked tests that share
            # some resources from parent to children which can have problems
            # of caching even though the test run in isolated subprocesses.
            try:
                from torch_sendnn import torch_sendnn  # noqa: F401
            except ImportError:
                print("WARNING: Disabled: torch_sendnn")

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

    def _cast_bf16_to_f16(self):
        """Cast all bf16 params in the model to f16.
        This is required for spyre cards that don't support bf16."""
        for name, param in self.model.named_parameters():
            if param.dtype == torch.bfloat16:
                logger.debug(
                    "You are casting param %s to fp16, which"
                    " will cause loss of accuracy. You can ignore"
                    " this warning if this is intended.", name)
                param.data = param.data.to(dtype=torch.float16)

    def _cast_to_f32(self):
        """Cast model parameters to f32.
        This is required for attention implementations that only support full
        precision."""
        for name, param in self.model.named_parameters():
            logger.debug("Casting param %s to fp32", name)
            param.data = param.data.to(dtype=torch.float32)


class ContinuousBatchingFmsModel(FmsModelBase):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: int,
    ) -> None:

        if model_config.quantization:
            raise ValueError("FP8 is not supported with continuous batching")

        BLOCK_SIZE = SpyrePlatform.get_block_size()
        max_model_len = scheduler_config.max_model_len

        # edge case: prompt fills model length: can produce 1 token with prefill
        max_prompt_length = max_model_len
        # edge case: prompt will be padded to first block:
        # can produce 1 token with prefill plus rest of model length
        max_decode_length = max_model_len - BLOCK_SIZE + 1

        super().__init__(model_config,
                         parallel_config,
                         max_prompt_length,
                         max_decode_length,
                         rank,
                         sendnn_dynamic=True)

        self.scheduler_config = scheduler_config
        self.parallel_config = parallel_config

        # physical KV cache on AIU Spyre: will eventually not live in this class
        self.kv_cache_specs = {}
        self.kv_cache_specs['block_size'] = BLOCK_SIZE
        self.kv_cache_specs['num_kv_heads'] = model_config.get_num_kv_heads(
            parallel_config)

        if self.config.model_type in {'llama', 'granite'}:
            self.kv_cache_specs['num_layers'] = self.config.num_hidden_layers
            self.kv_cache_specs['head_dim'] = self.config.hidden_size // \
                self.config.num_attention_heads
        elif self.config.model_type == 'gpt_bigcode':
            self.kv_cache_specs['num_layers'] = self.config.n_layer
            self.kv_cache_specs[
                'head_dim'] = self.config.n_embd // self.config.n_head
        else:
            raise NotImplementedError(
                f"[SpyreCausalLM] model type {self.config.model_type} "
                f"not supported in ContinuousBatchingFmsModel")

        if self.model_config.quantization:
            self.attention_name = "spyre_paged_attn_fp8"
        else:
            self.attention_name = "spyre_paged_attn"

    def get_num_blocks_available(self) -> int:
        """Function returns the number of available blocks/pages.
        Will eventually contain a function in torch_sendnn which reads 
        the actual value provided by the compiler for backend sendnn"""

        max_batch_size = self.scheduler_config.max_num_seqs
        max_model_len = self.scheduler_config.max_model_len
        block_size = self.kv_cache_specs['block_size']

        min_req_num_blocks = max_model_len // block_size

        # TODO: replace the hard coded NUM_BLOCKS_SPYRE by calling a function
        # in torch_sendnn which returns the value set by the Spyre compiler.
        if ('granite-3.3-8b-instruct' in self.model_config.model
                and self.parallel_config.world_size == 4):
            # hard coded value for tensor parallel size 4 with the below model
            # https://huggingface.co/ibm-granite/granite-3.3-8b-instruct
            NUM_BLOCKS_SPYRE = 2080
            logger.info(
                "Model %s and tensor parallel "
                "size %d detected. Using NUM_BLOCKS_SPYRE = %d",
                self.model_config.model,
                self.parallel_config.world_size,
                NUM_BLOCKS_SPYRE,
            )
        else:
            # default value for any other model/ tensor parallel size
            NUM_BLOCKS_SPYRE = max_batch_size * min_req_num_blocks
            logger.info("No model / tensor parallel size specific value for " \
            "the number of KV cache blocks available on Spyre found. Using " \
            "default value (max_batch_size * max_model_len / block_size): %d",
              NUM_BLOCKS_SPYRE)

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == 'sendnn':
            num_blocks_spyre = NUM_BLOCKS_SPYRE
            assert num_blocks_spyre >= min_req_num_blocks, (
                "Number of pages available on Spyre (%d) is not enough to "
                "serve the current model (need at least %d pages)." %
                (num_blocks_spyre, min_req_num_blocks))
            max_concurrency_spyre = num_blocks_spyre * block_size \
                / max_model_len
            logger.info("Spyre KV cache size: %s tokens",
                        num_blocks_spyre * block_size)
            logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                        str(max_model_len), max_concurrency_spyre)
            return num_blocks_spyre
        else:  # dynamo backend 'eager'
            # for debugging purposes we also put the spyre value here for cpu
            num_blocks_cpu = NUM_BLOCKS_SPYRE
            assert num_blocks_cpu >= min_req_num_blocks, (
                "Number of pages available on CPU (%d) is not enough to "
                "serve the current model (need at least %d pages)." %
                (num_blocks_cpu, min_req_num_blocks))
            max_concurrency_cpu = num_blocks_cpu * block_size \
                / max_model_len
            logger.info("CPU KV cache size: %s tokens",
                        num_blocks_cpu * block_size)
            logger.info("Maximum concurrency for %s tokens per request: %.2fx",
                        str(max_model_len), max_concurrency_cpu)
            return num_blocks_cpu

    def _set_past_key_value_states(self, num_blocks) -> None:
        # overwrite num_blocks for testing scheduler constraints
        num_blocks_override = SpyrePlatform.get_num_spyre_blocks_override()
        if num_blocks_override > 0:
            num_blocks = num_blocks_override

        # List[layers] of Tuple[k,v] of
        # Tensor[num_blocks, block_size, num_kv_heads, head_dim]

        if not self.model_config.quantization:
            self.past_key_value_states = [
                (torch.zeros(num_blocks,
                             self.kv_cache_specs['block_size'],
                             self.kv_cache_specs['num_kv_heads'],
                             self.kv_cache_specs['head_dim'],
                             dtype=self.dtype),
                 torch.zeros(num_blocks,
                             self.kv_cache_specs['block_size'],
                             self.kv_cache_specs['num_kv_heads'],
                             self.kv_cache_specs['head_dim'],
                             dtype=self.dtype))
                for _ in range(self.kv_cache_specs['num_layers'])
            ]
        else:
            # TODO: This does not work yet. The scale needs to be handled, see:
            # https://github.com/foundation-model-stack/aiu-fms-testing-utils/blob/v0.1.0rc3/aiu_fms_testing_utils/utils/paged.py#L306-L319
            from fms_mo.aiu_addons.fp8.fp8_utils import ScaledTensor
            self.past_key_value_states = [
                (ScaledTensor(torch.zeros(num_blocks,
                                          self.kv_cache_specs['block_size'],
                                          self.kv_cache_specs['num_kv_heads'],
                                          self.kv_cache_specs['head_dim'],
                                          dtype=self.dtype),
                              scale=torch.tensor([1.0] * 1,
                                                 dtype=torch.float32),
                              scaled=False),
                 ScaledTensor(torch.zeros(num_blocks,
                                          self.kv_cache_specs['block_size'],
                                          self.kv_cache_specs['num_kv_heads'],
                                          self.kv_cache_specs['head_dim'],
                                          dtype=self.dtype),
                              scale=torch.tensor([1.0] * 1,
                                                 dtype=torch.float32),
                              scaled=False))
                for _ in range(self.kv_cache_specs['num_layers'])
            ]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool,
        **extra_kwargs,
    ) -> torch.Tensor:

        forward_context = get_forward_context()

        attn_metadata = cast(SpyreAttentionMetadata,
                             forward_context.attn_metadata)
        # import will be not be needed/ handled by FMS soon
        import fms.utils.spyre.paged  # noqa # pylint: disable=unused-import

        # specify attention type for continuous batching
        extra_kwargs['attn_name'] = self.attention_name

        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=self.past_key_value_states,
            use_cache=use_cache,
            only_last_token=False,
            current_tkv_mask=attn_metadata.current_tkv_mask,
            left_padded_prompt_mask=attn_metadata.left_padded_prompt_mask,
            block_table=attn_metadata.block_table,
            slot_mapping=attn_metadata.slot_mapping,
            **extra_kwargs,
        )

        logits, self.past_key_value_states = output

        return logits

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


class StaticBatchingFmsModel(FmsModelBase):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        _: SchedulerConfig,
        max_prompt_length: int,
        max_decode_length: int,
        rank: int,
    ) -> None:
        super().__init__(model_config,
                         parallel_config,
                         max_prompt_length,
                         max_decode_length,
                         rank,
                         sendnn_dynamic=False)

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
        extra_kwargs['attn_name'] = "sdpa_causal"

        if envs_spyre.VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS:
            # In order to calculate prompt logprobs, we have to return the
            # hidden states from the whole prompt. The static graphs need to be
            # compiled with this set one way or the other.
            only_last_token = False
        else:
            only_last_token = True

        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=self.past_key_value_states,
            use_cache=use_cache,
            only_last_token=only_last_token,
            **extra_kwargs,
        )

        logits, self.past_key_value_states = output

        return logits

    def get_dtype(self) -> torch.dtype:
        # For static batching, we set fp16 on spyre and fp32 on cpu
        # (This applies even when running fp8 quantized models)
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            return torch.float16
        else:
            return torch.float32

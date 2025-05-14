"""Utilities for selecting and loading Spyre models."""
import os
from typing import Any, Optional

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn as nn
from fms.models import get_model
from transformers import PretrainedConfig
from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf)
from vllm.model_executor.sampling_metadata import SamplingMetadata

import vllm_spyre.envs as envs_spyre

try:
    from torch_sendnn import torch_sendnn  # noqa: F401
except ImportError:
    print("WARNING: Disabled: torch_sendnn")
    pass
try:
    import backends.dynamo_tracer  # noqa: F401
except ImportError:
    print("WARNING: Disabled: dynamo_tracer")
    pass

BACKEND_LIST = ['sendnn_decoder', 'inductor']

logger = init_logger(__name__)


class SpyreCausalLM(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        max_prompt_length: int,
        max_decode_length: int,
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
                                                    scheduler_config)
        else:
            self.model = StaticBatchingFmsModel(
                model_config,
                parallel_config,
                scheduler_config,
                max_prompt_length,
                max_decode_length,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
        current_tkv_mask: Optional[torch.Tensor] = None,
        left_padded_prompt_mask: Optional[torch.Tensor] = None,
        block_table: Optional[torch.Tensor] = None,
        slot_mapping: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if is_prompt and not envs_spyre.VLLM_SPYRE_USE_CB:
            self.model.past_key_value_states = None  # type: ignore

        extra_kwargs: dict[str, Any] = {}
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn_decoder":
            # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash
            # cpu impl when padding too much
            extra_kwargs["attn_algorithm"] = "math"

        if envs_spyre.VLLM_SPYRE_USE_CB:
            extra_kwargs["current_tkv_mask"] = current_tkv_mask
            extra_kwargs["left_padded_prompt_mask"] = left_padded_prompt_mask
            extra_kwargs["block_table"] = block_table
            extra_kwargs["slot_mapping"] = slot_mapping

        # normal prefill or decoding step
        logits = self.model(
            input_ids,
            position_ids=positions,
            mask=masks,
            use_cache=True,
            only_last_token=not envs_spyre.VLLM_SPYRE_USE_CB,
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
        sendnn_dynamic: bool,
    ) -> None:
        super().__init__()

        self.config: PretrainedConfig = model_config.hf_config
        self.dtype = torch.float16 if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == \
            'sendnn_decoder' else torch.float32

        # Actual FMS model
        self.model: nn.Module

        # Load the weights from the cached or downloaded files.
        self.load_weights(model_config=model_config,
                          max_prompt_length=max_prompt_length,
                          max_decode_length=max_decode_length,
                          distributed_strategy="tp"
                          if parallel_config.world_size > 1 else None,
                          sendnn_dynamic=sendnn_dynamic)

    def load_weights(
        self,
        model_config: ModelConfig,
        max_prompt_length: int,
        max_decode_length: int,
        distributed_strategy: Optional[str],
        sendnn_dynamic: bool,
        **kwargs,
    ) -> None:

        if model_config.quantization == "gptq":
            if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
                from fms_mo.aiu_addons.gptq import (  # noqa: F401
                    gptq_aiu_adapter, gptq_aiu_linear)
                linear_type = "gptq_aiu"
                logger.info("Loaded `aiu_addons` functionalities")
            else:
                linear_type = "gptq_cpu"
                logger.warning("GPTQ is not expected to work on CPU.")

            quant_cfg = model_config._parse_quant_hf_config()

            linear_config = {
                "linear_type": linear_type,
                "group_size": quant_cfg['group_size'],
                "desc_act": quant_cfg['desc_act'],
            }
            self.dtype = None
            model_source = "hf_gptq_aiu"
        else:
            linear_config = {"linear_type": "torch_linear"}
            model_source = "hf"

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

        # we can use fused weights unless running on Spyre
        fused_weights = envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn_decoder"

        self.model = get_model(architecture="hf_configured",
                               variant=model_config.model,
                               model_path=model_path,
                               source=model_source,
                               data_type=self.dtype,
                               distributed_strategy=distributed_strategy,
                               group=dist.group.WORLD,
                               fused_weights=fused_weights,
                               linear_config=linear_config)

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

            options = {"sendnn.dynamic": True} if sendnn_dynamic else {}

            self.model = torch.compile(
                self.model,
                backend=envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND,
                options=options,
            )


class ContinuousBatchingFmsModel(FmsModelBase):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:

        BLOCK_SIZE = 64
        max_batch = scheduler_config.max_num_seqs
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
                         sendnn_dynamic=True)

        # physical KV cache on AIU Spyre: will eventually not live in this class
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        if self.config.model_type in {'llama', 'granite'}:
            num_layers = self.config.num_hidden_layers
            head_dim = self.config.hidden_size // \
                self.config.num_attention_heads
        elif self.config.model_type == 'gpt_bigcode':
            num_layers = self.config.n_layer
            head_dim = self.config.n_embd // self.config.n_head
        else:
            raise NotImplementedError(
                f"[SpyreCausalLM] model type {self.config.model_type} "
                f"not supported in ContinuousBatchingFmsModel")

        num_blocks = max_batch * max_model_len // BLOCK_SIZE  # 64

        # List[layers] of Tuple[k,v] of
        # Tensor[num_blocks, BLOCK_SIZE, num_kv_heads, head_dim]
        self.past_key_value_states = [(torch.zeros(num_blocks,
                                                   BLOCK_SIZE,
                                                   num_kv_heads,
                                                   head_dim,
                                                   dtype=self.dtype),
                                       torch.zeros(num_blocks,
                                                   BLOCK_SIZE,
                                                   num_kv_heads,
                                                   head_dim,
                                                   dtype=self.dtype))
                                      for _ in range(num_layers)]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool,
        only_last_token: bool,
        current_tkv_mask: torch.Tensor,
        left_padded_prompt_mask: torch.Tensor,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        **extra_kwargs,
    ) -> torch.Tensor:

        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=self.past_key_value_states,
            use_cache=use_cache,
            only_last_token=only_last_token,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            **extra_kwargs,
        )

        logits, self.past_key_value_states = output

        return logits


class StaticBatchingFmsModel(FmsModelBase):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        _: SchedulerConfig,
        max_prompt_length: int,
        max_decode_length: int,
    ) -> None:
        super().__init__(model_config,
                         parallel_config,
                         max_prompt_length,
                         max_decode_length,
                         sendnn_dynamic=False)

        # dynamic KV cache
        self.past_key_value_states = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool,
        only_last_token: bool,
        **extra_kwargs,
    ) -> torch.Tensor:

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

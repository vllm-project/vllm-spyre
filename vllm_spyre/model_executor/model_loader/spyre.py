"""Utilities for selecting and loading Spyre models."""
import os
from dataclasses import dataclass
from typing import Any, Optional, cast

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn as nn
import vllm.envs as envs
from fms.models import get_model
from transformers import PretrainedConfig
from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import Sampler, SamplerOutput
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf)
from vllm.model_executor.sampling_metadata import SamplingMetadata

import vllm_spyre.envs as envs_spyre

try:
    import backends.dynamo_tracer  # noqa: F401
except ImportError:
    print("WARNING: Disabled: dynamo_tracer")
    pass

BACKEND_LIST = ['sendnn', 'inductor']

logger = init_logger(__name__)


def get_sampler() -> torch.nn.Module:
    if envs.VLLM_USE_V1:
        # Lazy import: the v1 package isn't distributed
        from vllm_spyre.v1.sample.sampler import Sampler as V1Sampler
        return V1Sampler()
    return Sampler()


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
        sendnn_dynamic: bool,
    ) -> None:
        super().__init__()

        self.config: PretrainedConfig = model_config.hf_config
        self.dtype = torch.float16 if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == \
            'sendnn' else torch.float32

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
            if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn":
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
        fused_weights = envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn"

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


class ContinuousBatchingFmsModel(FmsModelBase):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
    ) -> None:

        BLOCK_SIZE = 64  # hardcoded Spyre constraint for now
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

        # set num_blocks to the minimal value of 4 required for warmup
        # is reset to the value returned by the Spyre compiler after warmup
        # self._set_past_key_value_states(num_blocks=4)
        num_blocks = scheduler_config.max_num_seqs * max_model_len // BLOCK_SIZE
        self._set_past_key_value_states(num_blocks=num_blocks)

        # mark the num_blocks dimension dynamic for Spyre compiler for warmup
        # only, compiler will return the number of blocks it can accommodate.
        # (This is not yet supported by the compiler)
        # for layer in self.past_key_value_states:
        #     for tensor in layer:
        #         torch._dynamo.mark_dynamic(tensor, 0)

    def _set_past_key_value_states(self, num_blocks) -> None:
        # List[layers] of Tuple[k,v] of
        # Tensor[num_blocks, block_size, num_kv_heads, head_dim]
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
        extra_kwargs['attn_name'] = "spyre_paged_attn"

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
        **extra_kwargs,
    ) -> torch.Tensor:
        # specify attention type for static batching
        extra_kwargs['attn_name'] = "sdpa_bidirectional"

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

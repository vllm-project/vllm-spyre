"""Utilities for selecting and loading Spyre models."""
import os
import sys
from typing import Optional

import torch
import torch._inductor.config
import torch.distributed as dist
import torch.nn as nn
from fms.models import get_model
from transformers import PretrainedConfig
from vllm.config import ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.model_loader.weight_utils import (
    download_weights_from_hf)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.platforms import current_platform

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

# for testing use offline_inference_spyre_cb.py
TESTING_CB = False
PRINTS_CB = False


class SpyreCausalLM(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
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

        # FMS Wrapper Model
        fms_wrapper = FmsModelWrapper if envs_spyre.VLLM_SPYRE_USE_CB \
            else FmsModelPseudoWrapper
        self.model = fms_wrapper(
            model_config,
            parallel_config,
            max_prompt_length,
            max_decode_length,
        )

        # horizontal offset in physical KV cache memory block
        self.tkv: int = 0

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:

        if is_prompt:
            self.tkv = 0
            if not envs_spyre.VLLM_SPYRE_USE_CB:
                self.model.past_key_value_states = None

        extra_kwargs = {}
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn_decoder":
            # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash
            # cpu impl when padding too much
            extra_kwargs["attn_algorithm"] = "math"

        # testing only: prefil after 5 decodes
        if TESTING_CB and self.tkv == (5 + 64):
            # define sample prompt
            input_ids_list = [
                128000, 39314, 374, 459, 7754, 430, 16964, 264, 3465, 13, 9842,
                264, 2077, 430, 36001, 45695, 279, 1715, 13, 2893, 48887, 304,
                701, 2077, 311, 279, 1217, 382, 14711, 30151, 512, 61524, 264,
                1160, 315, 11470, 369, 20646, 16553, 19724, 369, 264, 3070,
                315, 3116, 382, 14711, 6075, 25
            ]
            prompt_len = len(input_ids_list)
            tkv_insert = self.tkv - 1
            padding_len = tkv_insert - prompt_len

            # construct token and position ids for the sample prompt
            self.model.sample_token_id = torch.tensor(
                [0] * padding_len + input_ids_list).unsqueeze(0)
            self.model.sample_position = torch.tensor(
                [0] * padding_len + [i
                                     for i in range(prompt_len)]).unsqueeze(0)

            # Construct attention mask for the sample prompt
            n = tkv_insert
            m = prompt_len

            top = torch.cat(
                (torch.tril(torch.ones(n - m, n - m)), torch.zeros(n - m, m)),
                dim=1)
            bottom = torch.cat(
                (torch.zeros(m, n - m), torch.tril(torch.ones(m, m))), dim=1)
            matrix = torch.cat((top, bottom), dim=0)

            matrix = matrix.masked_fill(matrix == 1,
                                        0).masked_fill(matrix == 0,
                                                       float('-inf'))
            self.model.sample_mask = matrix.unsqueeze(0)

            # prefil of batch size 1
            logits, self.tkv = self.model(
                self.model.sample_token_id,
                position_ids=self.model.sample_position,
                mask=self.model.sample_mask,
                use_cache=True,
                only_last_token=True,
                tkv=0,
                active_pages=[0],
                **extra_kwargs,
            )

            if PRINTS_CB:
                print('inserted sequence token id: ',
                      torch.argmax(logits[0, :]))

            # update sample_token_id, sample_position and sample_mask
            self.model.update_sample_inputs(logits=logits[0, :])

        if TESTING_CB and self.tkv >= (5 + 64):
            # set input_ids, positions, masks for inserted sequence
            input_ids[0, :] = self.model.sample_token_id
            positions[0, :] = self.model.sample_position
            masks[0, :, :] = self.model.sample_mask

        # normal prefil or decoding step
        logits, self.tkv = self.model(
            input_ids,
            position_ids=positions,
            mask=masks,
            use_cache=True,
            only_last_token=True,
            tkv=self.tkv,
            active_pages=[i for i in range(input_ids.shape[0])],
            **extra_kwargs,
        )
        if TESTING_CB and self.tkv >= (6 + 64):
            # update sample_token_id, sample_position and sample_mask
            self.model.update_sample_inputs(logits=logits[0, :])
            if PRINTS_CB:
                print('inserted sequence token id: ',
                      torch.argmax(logits[0, :]))

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


def get_spyre_model(
    model_config: ModelConfig,
    parallel_config: ParallelConfig,
    max_prompt_length,
    max_decode_length,
) -> nn.Module:

    # Create a model instance.
    model = SpyreCausalLM(model_config, parallel_config, max_prompt_length,
                          max_decode_length)

    return model


class FmsModelBaseWrapper(nn.Module):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_prompt_length: int,
        max_decode_length: int,
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
                          if parallel_config.world_size > 1 else None)

    def load_weights(
        self,
        model_config: ModelConfig,
        max_prompt_length: int,
        max_decode_length: int,
        distributed_strategy: Optional[str],
        **kwargs,
    ) -> None:

        if self.dtype is not model_config.dtype:
            logger.info(
                "Ignoring user-provided dtype=%s and using dtype=%s instead.",
                model_config.dtype, self.dtype)

        if model_config.quantization == "gptq":

            # note, we have to find a better way to package this
            # shouldn't it be part of FMS?
            sys.path.append("/home/senuser/aiu-fms")

            if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == "sendnn_decoder":
                from aiu_as_addon import aiu_adapter, aiu_linear  # noqa: F401
                linear_type = "gptq_aiu"
                print("Loaded `aiu_as_addon` functionalities")
            else:
                from cpu_addon import cpu_linear  # noqa: F401
                linear_type = "gptq_cpu"
                print("Loaded `cpu_addon` functionalities")

            quant_cfg = model_config._parse_quant_hf_config()

            linear_config = {
                "linear_type": linear_type,
                "group_size": quant_cfg['group_size'],
                "desc_act": quant_cfg['desc_act'],
            }
            data_type = None
            model_source = "hf_gptq_aiu"
        else:
            linear_config = {"linear_type": "torch_linear"}
            data_type = self.dtype
            model_source = "hf"

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
                               data_type=data_type,
                               distributed_strategy=distributed_strategy,
                               group=dist.group.WORLD,
                               fused_weights=fused_weights,
                               linear_config=linear_config)

        compile_mode = "default"

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
            print("NOTICE: Adjusting "
                  "torch._dynamo.config.accumulated_cache_size_limit"
                  f" from {_prev} to "
                  f"{torch._dynamo.config.accumulated_cache_size_limit} "
                  f"to accommodate prompt size of {max_prompt_length} "
                  f"and decode tokens of {max_decode_length}")

        if _target_cache_size > torch._dynamo.config.cache_size_limit:
            _prev = torch._dynamo.config.cache_size_limit
            torch._dynamo.config.cache_size_limit = _target_cache_size
            print(
                "NOTICE: Adjusting torch._dynamo.config.cache_size_limit from"
                f" {_prev} to {torch._dynamo.config.cache_size_limit} to "
                f"accommodate prompt size of {max_prompt_length} and "
                f"decode tokens of {max_decode_length}")

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            self.model = torch.compile(
                self.model,
                mode=compile_mode,
                backend=envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND)


class FmsModelWrapper(FmsModelBaseWrapper):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_prompt_length: int,
        max_decode_length: int,
    ) -> None:
        super().__init__(model_config, parallel_config, max_prompt_length,
                         max_decode_length)

        # physical KV cache (fms wrapper/ AIU Spyre)
        # lives in SpyreCausalLM only for convenient model access
        warmup_shapes = current_platform.get_warmup_shapes()
        max_batch = max(shape["batch_size"] for shape in warmup_shapes)
        max_prompt_length = max(shape["prompt_length"]
                                for shape in warmup_shapes)
        max_new_tokens = max(shape["new_tokens"] for shape in warmup_shapes)
        # Eventually max_model_len = self.config.max_position_embeddings,
        # but saving some memory here to only allocate the max in practise
        max_model_len = max_prompt_length + max_new_tokens

        if self.config.model_type == 'llama':
            num_layers = self.config.num_hidden_layers
            num_kv_heads = self.config.num_key_value_heads
            head_dim = self.config.hidden_size // \
                self.config.num_attention_heads
        elif self.config.model_type == 'gpt_bigcode':
            num_layers = self.config.n_layer
            num_kv_heads = 1 if self.config.multi_query else self.config.n_head
            head_dim = self.config.n_embd // self.config.n_head
        else:
            print(f"[SpyreCausalLM] model type {self.config.model_type} "
                  f"not supported in FMS wrapper")

        # (layers)x(k,v)x[max_batch, num_kv_heads, max_model_len, head_dim]
        self.fms_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = [
            (torch.empty((max_batch, num_kv_heads, max_model_len, head_dim)),
             torch.empty((max_batch, num_kv_heads, max_model_len, head_dim)))
            for i in range(num_layers)
        ]

        # variables used for testing insertion of sample input
        if TESTING_CB:
            self.sample_token_id: torch.Tensor = torch.empty((1, 1))
            self.sample_position: torch.Tensor = torch.empty((1, 1))
            self.sample_mask: torch.Tensor = torch.empty((1, 1, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool,
        only_last_token: bool,
        tkv: int,
        active_pages: list[int],
        **extra_kwargs,
    ) -> torch.Tensor:

        if PRINTS_CB:
            print("tkv", tkv)
            print("active_pages", active_pages)

        # read-out (dynamic) kv_cache for decoding steps only,
        # for prefills kv_cache = None
        if tkv == 0:  # prefil
            kv_cache = None
            tkv = input_ids.shape[1]
        else:  # decode
            kv_cache = []
            active_pages_mask = torch.zeros(self.fms_kv_cache[0][0].shape[0],
                                            dtype=torch.bool)
            active_pages_mask[active_pages] = True
            for layer in range(len(self.fms_kv_cache)):
                kv_cache.append(
                    (self.fms_kv_cache[layer][0][active_pages_mask, :, :tkv -
                                                 1, :],
                     self.fms_kv_cache[layer][1][active_pages_mask, :, :tkv -
                                                 1, :]))

        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=kv_cache,
            use_cache=use_cache,
            only_last_token=only_last_token,
            **extra_kwargs,
        )
        logits, key_value_states = output

        # updating (physical) KV cache: self.fms_kv_cache
        for idx, page in enumerate(sorted(active_pages)):
            for layer in range(len(self.fms_kv_cache)):
                # inserting partial KV cache at correct location
                # (page, tkv) in the KV cache of the whole batch
                self.fms_kv_cache[layer][0][
                    page, :, :tkv, :] = key_value_states[layer][0][
                        idx, :, :, :]  # [1, 8, L, 128]
                self.fms_kv_cache[layer][1][
                    page, :, :tkv, :] = key_value_states[layer][1][
                        idx, :, :, :]  # [1, 8, L, 128]

        return logits, tkv + 1

    def update_sample_inputs(
        self,
        logits: torch.Tensor,
    ) -> None:

        self.sample_token_id = torch.argmax(logits).clone().detach().reshape(
            (1, 1))
        self.sample_position = (self.sample_position[0, -1] +
                                1).clone().detach().reshape((1, 1))
        self.sample_mask = torch.nn.functional.pad(
            self.sample_mask[0, -1, :].unsqueeze(0), (0, 1)).unsqueeze(0)


class FmsModelPseudoWrapper(FmsModelBaseWrapper):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_prompt_length: int,
        max_decode_length: int,
    ) -> None:
        super().__init__(model_config, parallel_config, max_prompt_length,
                         max_decode_length)

        # dynamic KV cache
        self.past_key_value_states = None

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        use_cache: bool,
        only_last_token: bool,
        tkv: int,
        active_pages: list[int],
        **extra_kwargs,
    ) -> torch.Tensor:

        if tkv == 0:  # prefil
            tkv = input_ids.shape[1]

        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=self.past_key_value_states,
            use_cache=use_cache,
            only_last_token=only_last_token,
            **extra_kwargs,
        )

        logits, past_key_value_states = output
        self.past_key_value_states = past_key_value_states

        # mark dynamic
        if self.past_key_value_states is not None:
            for layer in self.past_key_value_states:
                for tensor in layer:
                    torch._dynamo.mark_dynamic(tensor, 2)

        return logits, tkv + 1

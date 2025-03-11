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

TESTING_CB = False
PRINTS_CB = False


class SpyreCausalLM(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.logits_processor = LogitsProcessor(config.vocab_size,
                                                logits_as_input=True)
        self.sampler = get_sampler()
        self.past_key_value_states = None
        self.dtype = torch.float16 if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND == \
            'sendnn_decoder' else torch.float32
        # boolean tensor of length batch size with indices:
        # True for unfinished sequences and
        # False for finished or padded sequences
        self.indices = None

        # Lazy initialized
        self.model: nn.Module

        if envs_spyre.VLLM_SPYRE_USE_CB:
            # physical KV cache (fms wrapper/ AIU Spyre)
            # lives in SpyreCausalLM only for convenient model access
            B = 4  # max batch size
            L = 128  # max model length
            # 10 (layers) x 2 (k, v) x [B, 8, L, 128]
            self.fms_kv_cache: list[tuple[torch.Tensor, torch.Tensor]] = [
                (torch.empty((B, 8, L, 128)), torch.empty((B, 8, L, 128)))
                for i in range(10)
            ]

            # horizontal offset in physical KV cache memory block
            self.tkv: int = 0

            # variables used for testing insertion of sample input
            if TESTING_CB:
                self.sample_token_id: torch.Tensor = torch.empty((1, 1))
                self.sample_position: torch.Tensor = torch.empty((1, 1))
                self.sample_mask: torch.Tensor = torch.empty((1, 1, 1))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        masks: torch.Tensor,
        is_prompt: bool,
    ) -> torch.Tensor:

        extra_kwargs = {}
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn_decoder":
            # Bug in 2.3.1 fixed in 2.4.1 for SDPA flash
            # cpu impl when padding too much
            extra_kwargs["attn_algorithm"] = "math"

        if envs_spyre.VLLM_SPYRE_USE_CB:

            if is_prompt and self.tkv == 0:
                self.past_key_value_states = None
                self.tkv = input_ids.shape[1]

            # testing only: prefil after 5 decodes
            if TESTING_CB and self.tkv == (5 + 64):
                # define sample prompt
                input_ids_list = [
                    128000, 39314, 374, 459, 7754, 430, 16964, 264, 3465, 13,
                    9842, 264, 2077, 430, 36001, 45695, 279, 1715, 13, 2893,
                    48887, 304, 701, 2077, 311, 279, 1217, 382, 14711, 30151,
                    512, 61524, 264, 1160, 315, 11470, 369, 20646, 16553,
                    19724, 369, 264, 3070, 315, 3116, 382, 14711, 6075, 25
                ]
                prompt_len = len(input_ids_list)
                padding_len = (self.tkv - 1) - prompt_len

                # construct token and position ids for the sample prompt
                self.sample_token_id = torch.tensor(
                    [0] * padding_len + input_ids_list).unsqueeze(0)
                self.sample_position = torch.tensor(
                    [0] * padding_len +
                    [i for i in range(prompt_len)]).unsqueeze(0)

                # Construct attention mask for the sample prompt
                n = self.tkv - 1
                m = padding_len

                top = torch.cat(
                    (torch.tril(torch.ones(n - m, n - m)), torch.zeros(
                        n - m, m)),
                    dim=1)
                bottom = torch.cat(
                    (torch.zeros(m, n - m), torch.tril(torch.ones(m, m))),
                    dim=1)
                matrix = torch.cat((top, bottom), dim=0)

                matrix = matrix.masked_fill(matrix == 1, 0).masked_fill(
                    matrix == 0, float('-inf'))
                self.sample_mask = matrix.unsqueeze(0)

                # prefil of batch size 1
                output = self.fms_wrapper(
                    self.sample_token_id,
                    position_ids=self.sample_position,
                    mask=self.sample_mask,
                    past_key_value_states=None,
                    use_cache=True,
                    only_last_token=True,
                    tkv=self.tkv - 1,
                    active_pages=[0],
                    **extra_kwargs,
                )
                logits, self.past_key_value_states = output

                if PRINTS_CB:
                    print('inserted sequence token id: ',
                          torch.argmax(logits[0, :]))

                # update sample_token_id, sample_position and sample_mask
                self.update_sample_inputs(logits=logits[0, :])

            if TESTING_CB and self.tkv >= (5 + 64):
                # set input_ids, positions, masks for inserted sequence
                input_ids[0, :] = self.sample_token_id
                positions[0, :] = self.sample_position
                masks[0, :, :] = self.sample_mask

            # normal prefil or decoding step
            output = self.fms_wrapper(
                input_ids,
                position_ids=positions,
                mask=masks,
                past_key_value_states=self.past_key_value_states,
                use_cache=True,
                only_last_token=True,
                tkv=self.tkv,
                active_pages=[i for i in range(input_ids.shape[0])],
                **extra_kwargs,
            )
            if TESTING_CB and self.tkv >= (5 + 64):
                # update sample_token_id, sample_position and sample_mask
                self.update_sample_inputs(logits=output[0][0, :])

            self.tkv += 1
        else:
            # if not envs_spyre.VLLM_SPYRE_USE_CB
            if is_prompt:
                self.past_key_value_states = None

            output = self.model(
                input_ids,
                position_ids=positions,
                mask=masks,
                past_key_value_states=self.past_key_value_states,
                use_cache=True,
                only_last_token=True,
                **extra_kwargs,
            )

        logits, past_key_value_states = output
        self.past_key_value_states = past_key_value_states

        # mark dynamic
        if self.past_key_value_states is not None:
            for layer in self.past_key_value_states:
                for tensor in layer:
                    torch._dynamo.mark_dynamic(tensor, 2)

        if (envs_spyre.VLLM_SPYRE_USE_CB and TESTING_CB and PRINTS_CB
                and self.tkv > (5 + 64)):
            print('inserted sequence token id: ', torch.argmax(logits[0, :]))
            # print(logits[0, 12269], logits[0, 19724])

        # removing finished or padded sequences
        logits = logits[self.indices]

        return logits

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

    def fms_wrapper(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        mask: torch.Tensor,
        past_key_value_states: torch.Tensor,
        use_cache: bool,
        only_last_token: bool,
        tkv: int,
        active_pages: list[int],
        **extra_kwargs,
    ) -> torch.Tensor:

        if PRINTS_CB:
            print("tkv", tkv)
            print("active_pages", active_pages)

        output = self.model(
            input_ids,
            position_ids=position_ids,
            mask=mask,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            only_last_token=only_last_token,
            **extra_kwargs,
        )
        logits, key_value_states = output

        # updating (physical) KV cache: self.fms_kv_cache
        for page in active_pages:
            for layer in range(len(self.fms_kv_cache)):
                # inserting partial KV cache at correct location
                # (page, tkv) in the KV cache of the whole batch
                self.fms_kv_cache[layer][0][
                    page, :, :tkv, :] = key_value_states[layer][0][
                        page, :, :, :]  # [1, 8, L, 128]
                self.fms_kv_cache[layer][1][
                    page, :, :tkv, :] = key_value_states[layer][1][
                        page, :, :, :]  # [1, 8, L, 128]

        # (dynamic) KV cache: self.past_key_value_states
        # for fms CPU eager execution
        updated_kv_cache = []
        for layer in range(len(self.fms_kv_cache)):
            updated_kv_cache.append(
                (self.fms_kv_cache[layer][0][:, :, :tkv, :],
                 self.fms_kv_cache[layer][1][:, :, :tkv, :]))

        return logits, updated_kv_cache

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(None, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, model_config: ModelConfig, max_prompt_length: int,
                     max_decode_length: int,
                     distributed_strategy: Optional[str], **kwargs):

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


def get_spyre_model(model_config: ModelConfig, parallel_config: ParallelConfig,
                    max_prompt_length, max_decode_length) -> nn.Module:

    # Create a model instance.
    model = SpyreCausalLM(model_config.hf_config)

    # Load the weights from the cached or downloaded files.
    model.load_weights(
        model_config,
        max_prompt_length=max_prompt_length,
        max_decode_length=max_decode_length,
        distributed_strategy="tp" if parallel_config.world_size > 1 else None)
    return model

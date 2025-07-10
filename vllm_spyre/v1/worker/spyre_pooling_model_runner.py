import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn
from transformers import AutoModel
from vllm.config import DeviceConfig, VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_spyre.platform import SpyrePlatform
# yapf conflicts with ruff for this block
# yapf: disable
from vllm_spyre.v1.worker.spyre_pooling_input_batch import (
    PoolingInputBatch, PoolingRequestState)

# yapf: enable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                           SchedulerOutput)
else:
    CachedRequestData = None
    SchedulerOutput = None
    NewRequestData = None

from vllm.v1.outputs import ModelRunnerOutput

import vllm_spyre.envs as envs_spyre

logger = init_logger(__name__)

BACKEND_LIST = ['sendnn', 'inductor']


@dataclass(frozen=True)
class ModelForwardInputs:

    input_tokens: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_masks: Optional[torch.Tensor] = None
    is_prompt: Optional[bool] = None


class SpyreModelRunner:

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
    ):
        self.is_driver_worker = is_driver_worker
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.device_config = vllm_config.device_config
        self.speculative_config = vllm_config.speculative_config
        self.prompt_adapter_config = vllm_config.prompt_adapter_config
        self.observability_config = vllm_config.observability_config

        self.pad_token_id = 0

        if self.model_config is not None:
            if self.model_config.hf_config is not None:
                self.pad_token_id = (getattr(self.model_config.hf_config,
                                             "pad_token_id", None) or 0)
            if self.model_config.get_sliding_window():
                logger.warning("Sliding window is not supported on Spyre. "
                               "The model will run without sliding window.")
        if vllm_config.device_config is None:
            self.device_config = DeviceConfig()
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        # Lazy initialization: after load_model.
        self.model: nn.Module

        # Flag to be turned off after warmup is complete
        self.warmup_mode = True

        # Batch state
        self.input_batch = PoolingInputBatch(
            max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
            max_model_len=vllm_config.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=vllm_config.model_config.get_vocab_size(),
        )

        # Requests
        self.requests: dict[str, CachedRequestData] = {}

    def get_model(self) -> nn.Module:
        return self.model

    def load_model(self, prompt_lens: Iterable[int],
                   num_decode_tokens: Iterable[int]) -> None:
        self.model = AutoModel.from_pretrained(self.model_config.model)
        self.model.eval()
        torch.set_grad_enabled(False)
        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND in BACKEND_LIST:
            self.model = torch.compile(
                self.model,
                mode="default",
                dynamic=False,
                backend=envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND)

    @property
    def vocab_size(self) -> int:
        return self.model.config.vocab_size

    def _prepare_pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        token_type_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """left side padding implemented as
        in fms.utils.generation.pad_input_id"""
        max_len = max([min_pad_length] +
                      [seq.size(0) for seq in input_ids_list])
        padded_input_ids_list = []
        padded_token_type_list = []
        mask_list = []
        position_ids_list = []
        for i, input_ids_i in enumerate(input_ids_list):
            seq_len = input_ids_i.size(0)
            if max_len > seq_len:
                logger.info(
                    "Left padding request of length %d tokens to %d tokens.",
                    seq_len, max_len)
            pads = torch.ones(max_len - seq_len,
                              dtype=torch.long,
                              device=input_ids_i.device) * self.pad_token_id
            non_pads = torch.ones(seq_len,
                                  dtype=torch.long,
                                  device=input_ids_i.device)

            pos_ids_pads = pads
            pos_ids_seq = torch.arange(0,
                                       seq_len,
                                       dtype=torch.long,
                                       device=input_ids_i.device)

            # Setting this to 0, however if 0 is the eos, we will end up
            # truncating the output if using truncate_after_eos once this
            # workflow works for nested tensor, this can probably be removed
            if token_type_list:
                padded_token_type_list.append(
                    torch.cat((pads, token_type_list[i])))
            padded_input_ids_list.append(torch.cat((pads, input_ids_i)))
            mask_list.append(torch.cat((torch.zeros_like(pads), non_pads)))
            position_ids_list.append(torch.cat((pos_ids_pads, pos_ids_seq)))

        return padded_input_ids_list, padded_token_type_list,\
             mask_list, position_ids_list

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        token_type_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        padded_input_ids_list, padded_token_type_list, \
            mask_list, position_ids_list = (
            self._prepare_pad_input_ids(input_ids_list, token_type_list,
                                        min_pad_length))

        input_ids = torch.stack(padded_input_ids_list)
        if padded_token_type_list:
            token_type_ids = torch.stack(padded_token_type_list)
        else:
            token_type_ids = None
        mask = torch.stack(mask_list)
        position_ids = torch.stack(position_ids_list)

        return input_ids, token_type_ids, position_ids, mask

    def get_kv_cache_spec(self) -> KVCacheSpec:
        """
        This method should generate the KVCache spec by parsing the kv cache
        format from each Attention module in the static forward context.

        In vLLM, this static forward context is populated by the base Attention
        class in the modeling code. Every attention layer populates an entry
        for itself in vllm_config.compilation_config.static_forward_context,
        which is a dictionary of layer_name -> layer for every attention layer.
        This allows the model runner to correctly create the kv cache spec for
        each layer.

        The spyre modeling code currently comes from `fms`, and does not
        integrate with vLLM's modeling classes, so we don't have access to any
        model-agnostic metadata about the attention layers. This just returns a
        dummy value for now.
        """
        # We do at least use the real size from the cache config.
        block_size = self.vllm_config.cache_config.block_size

        attn_spec = FullAttentionSpec(block_size=block_size,
                                      num_kv_heads=1,
                                      head_size=1,
                                      dtype=torch.float16,
                                      use_mla=False)
        return {"foo": attn_spec}

    def complete_warmup(self):
        """Turn off warmup mode once the warmup is complete"""
        self.warmup_mode = False

    def _update_states(self, scheduler_output: SchedulerOutput):
        # Update the states of the running/resumed requests.
        # Update input_batch's `token_ids_cpu`,
        # `num_tokens`. For continuous batching it cleans
        # finished requests from the batch
        #
        # NOTE: req_state.output_token_ids is being mutated.

        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])

            req_index = self.input_batch.get_req_index(req_id)
            # Add new_token_ids to token_ids_cpu.
            # TODO: Update for spec decoding in the future
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids


class SpyrePoolingModelRunner(SpyreModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
    ):
        super().__init__(vllm_config=vllm_config,
                         is_driver_worker=is_driver_worker)

        # position_ids of all the sequences in current batch
        self._position_ids: torch.Tensor = None
        # attention masks of all the sequences in current batch
        self._mask: torch.Tensor = None

        self.spyre_warmup_shapes = SpyrePlatform.get_warmup_shapes(
            self.scheduler_config)

        pooler_config = vllm_config.model_config.pooler_config
        self.pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.CLS,
            normalize=True,
            softmax=False)

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> ModelForwardInputs:
        assert len(new_requests) > 0
        input_token_list: list[torch.Tensor] = []
        token_type_list: list[torch.Tensor] = []
        padded_batch_size, min_pad_length_batch = self._get_padded_batch_size(
            new_requests)

        # Internal state is reset here.
        # We don't support continuous batching, so we know all previous requests
        # have finished decoding.
        self.input_batch.clear_requests()
        self.requests = {}

        # Build batch and prepare input_token1
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids

            input_token_list.append(
                torch.tensor(prompt_tokens,
                             dtype=torch.long,
                             device=torch.device("cpu")))

            if request_data.token_type_ids is not None:
                token_type_list.append(
                    torch.tensor(request_data.token_type_ids,
                                 dtype=torch.long,
                                 device=torch.device("cpu")))

            # Add new requests to the cached states.
            req_id = request_data.req_id
            pooling_params = request_data.pooling_params
            assert pooling_params is not None

            req_state = PoolingRequestState(
                req_id=req_id,
                prompt_token_ids=request_data.prompt_token_ids,
                token_type_ids=request_data.token_type_ids,
                pooling_params=pooling_params,
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)

        self.input_batch.padded_batch_size = padded_batch_size

        # Refresh sampling metadata after all request are added to the batch
        self.input_batch.refresh_metadata()

        if token_type_list:
            assert len(input_token_list) == len(token_type_list)

        # padding to compiled batch size
        while len(input_token_list) < padded_batch_size:
            input_token_list.append(
                torch.zeros(min_pad_length_batch,
                            dtype=torch.long,
                            device=torch.device("cpu")))
            if token_type_list:
                token_type_list.append(
                    torch.zeros(min_pad_length_batch,
                                dtype=torch.long,
                                device=torch.device("cpu")))

        # get position ids and attention mask
        input_tokens, token_type_ids, self._position_ids, self._mask =\
            self.pad_input_ids(
            input_token_list,
            token_type_list,
            min_pad_length=min_pad_length_batch)

        return ModelForwardInputs(
            input_tokens=input_tokens,
            token_type_ids=token_type_ids,
            input_positions=self._position_ids,
            input_masks=self._mask,
            is_prompt=True,
        )

    def prepare_model_input(
            self, scheduler_output: SchedulerOutput) -> ModelForwardInputs:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare input tensors.
        assert is_prompt
        # Assert no running requests
        assert len(scheduler_output.scheduled_cached_reqs) == 0
        return self._prepare_prompt(scheduler_output.scheduled_new_reqs)

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:

        t0 = time.time()

        # TODO: change to EMPTY_MODEL_RUNNER_OUTPUT, right now this
        # will be a breaking change, or clumsy to make retrocompatible
        # with conditional import
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        self._update_states(scheduler_output)

        model_input = self.prepare_model_input(scheduler_output)
        self._mark_input_tensors(model_input)

        model_kwargs: dict[str, Any] = {}
        if model_input.token_type_ids:
            model_kwargs["token_type_ids"] =\
                  model_input.token_type_ids

        outputs = self.model(
            input_ids=model_input.input_tokens,
            # TODO: verify this position id thing
            #position_ids=model_input.input_positions,
            attention_mask=model_input.input_masks)
        hidden_states = outputs["last_hidden_state"]

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        t1 = time.time() - t0
        logger.debug("t_batch: %.2fms", (t1 * 1000))

        pooling_metadata = self.input_batch.make_pooling_metadata()

        # prepare unpadded output for the pooler
        hidden_state_list: list[torch.Tensor] = []
        for hidden_state, prompt_len in zip(hidden_states,
                                            pooling_metadata.prompt_lens):
            # we're left padding
            hidden_state_list.append(hidden_state[-prompt_len:])

        raw_pooler_output = self.pooler(hidden_states=hidden_state_list,
                                        pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []

        for raw_output in raw_pooler_output:
            pooler_output.append(raw_output.data.to("cpu"))

        model_output = ModelRunnerOutput(
            req_ids=self.input_batch.requests_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )
        return model_output

    def _get_padded_batch_size(self, new_requests: list[NewRequestData]):
        # find warmup shape to be used for padding and batching
        applicable_spyre_warmup_shapes = [
            shape for shape in self.spyre_warmup_shapes
            if len(new_requests) <= shape["batch_size"]
        ]
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids
            new_tokens = (request_data.sampling_params.max_tokens
                          if request_data.sampling_params is not None else 0)

            updated_spyre_warmup_shapes = [
                shape for shape in applicable_spyre_warmup_shapes
                if len(prompt_tokens) <= shape["prompt_length"]
                and new_tokens <= shape["new_tokens"]
            ]
            applicable_spyre_warmup_shapes = updated_spyre_warmup_shapes

        assert (
            applicable_spyre_warmup_shapes
        ), "No shapes available to run prefill batch. (This should not happen)"

        # If multiple warmup shapes apply, the first one is selected.
        # For improving performance, the warmup shapes in scheduler_config
        # are ordered by "processing speed".
        min_pad_length_batch = applicable_spyre_warmup_shapes[0][
            "prompt_length"]
        padded_batch_size = applicable_spyre_warmup_shapes[0]["batch_size"]
        return padded_batch_size, min_pad_length_batch

    def _mark_input_tensors(self, model_input: ModelForwardInputs) -> None:
        """Yoinked from
        https://github.com/foundation-model-stack/aiu-fms-testing-utils/pull/13
        """
        if not self.warmup_mode:
            # Only mark tensors when we're warming up and compiling the graphs
            return

        # To produce like graphs during pre-fill, we mark the prefill
        # batch x seq as static, but relax this for decode for the seq
        if model_input.is_prompt:
            # we always want prefill to be static to produce same-like graph
            torch._dynamo.mark_static(model_input.input_tokens, 0)
            torch._dynamo.mark_static(model_input.input_tokens, 1)
            if model_input.token_type_ids is not None:
                torch._dynamo.mark_static(model_input.token_type_ids, 0)
                torch._dynamo.mark_static(model_input.token_type_ids, 1)
            torch._dynamo.mark_static(model_input.input_masks, 0)
            torch._dynamo.mark_static(model_input.input_masks, 1)
            torch._dynamo.mark_static(model_input.input_masks, 2)
            torch._dynamo.mark_static(model_input.input_positions, 0)
            torch._dynamo.mark_static(model_input.input_positions, 1)
        else:
            # we always want the decode to be dynamic on sequence
            torch._dynamo.mark_dynamic(model_input.input_masks, 2)

            # here self.model.model is a StaticBatchingFmsModel
            for layer in self.model.model.past_key_value_states:
                for tensor in layer:
                    torch._dynamo.mark_static(tensor, 0)
                    # This used to be baked into the model's forward pass
                    torch._dynamo.mark_dynamic(tensor, 2)

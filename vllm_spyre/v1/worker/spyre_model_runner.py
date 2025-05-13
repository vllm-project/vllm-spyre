import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
from torch import nn
from vllm.config import DeviceConfig, VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingType
from vllm.utils import is_pin_memory_available
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec
from vllm.v1.outputs import SamplerOutput

from vllm_spyre.model_executor.model_loader.spyre import SpyreCausalLM
from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_input_batch import (CachedRequestState,
                                                    InputBatch)

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


@dataclass(frozen=True)
class ModelForwardInputs:
    """
    Used by the SpyreModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_masks: Optional[torch.Tensor] = None
    current_tkv_mask: Optional[torch.Tensor] = None
    left_padded_prompt_mask: Optional[torch.Tensor] = None
    block_table: Optional[torch.Tensor] = None
    slot_mapping: Optional[torch.Tensor] = None
    is_prompt: Optional[bool] = None


@dataclass
class CBSpyreModelRunnerOutput(ModelRunnerOutput):
    # Add the current tkv to the output
    tkv: int = 0


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
        self.input_batch = InputBatch(
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
        max_pad_length = max(prompt_lens)
        max_decode_length = max(num_decode_tokens)
        self.model = SpyreCausalLM(
            self.model_config,
            parallel_config=self.parallel_config,
            scheduler_config=self.scheduler_config,
            max_prompt_length=max_pad_length,
            max_decode_length=max_decode_length,
        )

    @property
    def vocab_size(self) -> int:
        return self.model.model.model.config.src_vocab_size

    def _prepare_pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """left side padding implemented as
        in fms.utils.generation.pad_input_id"""
        max_len = max([min_pad_length] +
                      [seq.size(0) for seq in input_ids_list])
        padded_input_ids_list = []
        mask_list = []
        position_ids_list = []
        for input_ids_i in input_ids_list:
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
            padded_input_ids_list.append(torch.cat((pads, input_ids_i)))
            mask_list.append(torch.cat((torch.zeros_like(pads), non_pads)))
            position_ids_list.append(torch.cat((pos_ids_pads, pos_ids_seq)))

        return padded_input_ids_list, mask_list, position_ids_list

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        padded_input_ids_list, mask_list, position_ids_list = (
            self._prepare_pad_input_ids(input_ids_list, min_pad_length))

        input_ids = torch.stack(padded_input_ids_list)
        mask = torch.stack(mask_list).bool()
        # this is a causal mask for generation
        mask = (mask.unsqueeze(-1) == mask.unsqueeze(-2)).tril()
        mask = torch.where(mask.logical_not(), -torch.inf, 0.0)
        mask = mask.to(self.model.model.dtype)
        position_ids = torch.stack(position_ids_list)

        return input_ids, position_ids, mask

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

        attn_spec = FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=1,
            head_size=1,
            dtype=torch.float16,
            use_mla=False,
        )
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


class StaticBatchingSpyreModelRunner(SpyreModelRunner):

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

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> ModelForwardInputs:
        assert len(new_requests) > 0
        input_token_list: list[torch.Tensor] = []
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

            # Add new requests to the cached states.
            req_id = request_data.req_id
            sampling_params = request_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=request_data.prompt_token_ids,
                sampling_params=sampling_params,
                generator=generator,
                output_token_ids=[],
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)

        self.input_batch.padded_batch_size = padded_batch_size

        # Refresh sampling metadata after all request are added to the batch
        self.input_batch.refresh_sampling_metadata()

        # padding to compiled batch size
        while len(input_token_list) < padded_batch_size:
            input_token_list.append(
                torch.zeros(min_pad_length_batch,
                            dtype=torch.long,
                            device=torch.device("cpu")))

        # get position ids and attention mask
        input_tokens, self._position_ids, self._mask = self.pad_input_ids(
            input_token_list, min_pad_length=min_pad_length_batch)

        return ModelForwardInputs(
            input_tokens=input_tokens,
            input_positions=self._position_ids,
            input_masks=self._mask,
            is_prompt=True,
        )

    def _prepare_decode(
        self,
        cached_requests: list[CachedRequestData],
    ) -> ModelForwardInputs:
        assert len(cached_requests) > 0
        input_tokens: list[list[int]] = [
            [0] for _ in range(self._position_ids.shape[0])
        ]

        for cached_request in cached_requests:
            # TODO: Will this always just be one token ID if there's no spec
            # or jump decoding?
            generation_token = cached_request.new_token_ids[-1]
            input_tokens[self.input_batch.req_id_to_index[
                cached_request.req_id]] = [generation_token]

        # update position ids and attention mask
        self._update_position_ids()
        self._update_mask()

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        return ModelForwardInputs(
            input_tokens=input_tokens,
            input_positions=self._position_ids,
            input_masks=self._mask,
            is_prompt=False,
        )

    def _update_position_ids(self) -> None:
        """Updating the position ids of all sequences
        in a batch. Will be called in decoding phase"""

        self._position_ids = self._position_ids[:, -1] + 1
        self._position_ids = self._position_ids.unsqueeze(-1)

    def _update_mask(self) -> None:
        """Updating/extending the attention masks of all
        sequences in a batch. Will be called in decoding phase"""

        assert self._mask is not None
        masks = self._mask

        masks_new = []
        for mask in masks:
            # get the last row of the 3d mask
            mask_new = mask[-1:, :]

            # extend the mask one slot
            mask_new = torch.cat(
                (
                    mask_new,
                    torch.zeros(
                        1, 1, dtype=mask_new.dtype, device=mask_new.device),
                ),
                dim=1,
            )
            masks_new.append(mask_new)

        self._mask = torch.stack(masks_new, dim=0)

    def prepare_model_input(
            self, scheduler_output: SchedulerOutput) -> ModelForwardInputs:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare input tensors.
        if is_prompt:
            # Assert no running requests
            assert len(scheduler_output.scheduled_cached_reqs) == 0

            return self._prepare_prompt(scheduler_output.scheduled_new_reqs)
        else:
            if scheduler_output.finished_req_ids:
                for req_id in scheduler_output.finished_req_ids:
                    self.input_batch.remove_request(req_id)
                self.input_batch.refresh_sampling_metadata()

            return self._prepare_decode(scheduler_output.scheduled_cached_reqs)

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
            )

        self._update_states(scheduler_output)

        model_input = self.prepare_model_input(scheduler_output)
        self._mark_input_tensors(model_input)

        # TODO(Wallas): I think it would be better move the indices as argument
        # of the forward rather than set as an attribute of the model. I'm not
        # sure how easy is that right now.

        # Always get the indices from the input_batch
        self.model.indices = self.input_batch.get_model_indices()

        # Execute the model
        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            masks=model_input.input_masks,
            is_prompt=model_input.is_prompt,
        )

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=self.input_batch.sampling_metadata,
        )
        t1 = time.time() - t0
        logger.debug("t_token: %.2fms", (t1 * 1000))

        model_output = ModelRunnerOutput(
            req_ids=self.input_batch.requests_ids,
            req_id_to_index=self.input_batch.get_unpadded_output_indices(),
            sampled_token_ids=output.sampled_token_ids.tolist(),
            spec_token_ids=None,
            logprobs=(output.logprobs_tensors.tolists()
                      if output.logprobs_tensors else None),
            prompt_logprobs_dict={
                req_id: None
                for req_id in self.input_batch.req_id_to_index
            },  # TODO(wallas?): prompt logprobs too
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


class ContinuousBatchingSpyreModelRunner(SpyreModelRunner):

    def __init__(
        self,
        vllm_config: VllmConfig,
        is_driver_worker: bool,
    ):
        super().__init__(vllm_config=vllm_config,
                         is_driver_worker=is_driver_worker)

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        max_model_len = vllm_config.scheduler_config.max_model_len

        # TODO: remove this limitation once we update the warm-up logic to
        # support batch_size=1
        assert max_batch_size >= 2, "Currently, continuous batching needs " \
            "config to set batch_size >= 2"

        self.BLOCK_SIZE = 64
        NUM_BLOCKS = max_batch_size * max_model_len // self.BLOCK_SIZE  # 64

        # TO DO: move to InputBatch
        self.req_ids2blocks: dict[str, deque[int]] = {}
        self.req_ids2left_pads: dict[str, int] = {}
        self.tkv = 0
        self.free_blocks = deque([i for i in range(NUM_BLOCKS)])

        # TODO: Remove this once we can prefill and decode
        # in the same step
        self.prefill_batch = InputBatch(
            # TODO: review this, currently we only support prefill for
            # `batch_size=1`
            max_num_reqs=1,
            max_model_len=vllm_config.model_config.max_model_len,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=vllm_config.model_config.get_vocab_size(),
        )

    def _update_states(self, scheduler_output):

        super()._update_states(scheduler_output)

        # Continuous batching stuff
        for req_id in scheduler_output.finished_req_ids:
            if req_id in self.req_ids2blocks:
                logger.debug("Freeing request id: %s", req_id)
                for freed_block in self.req_ids2blocks[req_id]:
                    self.free_blocks.append(freed_block)
                del self.req_ids2blocks[req_id]
                del self.req_ids2left_pads[req_id]

            del self.requests[req_id]
            self.input_batch.remove_request(req_id)

    def _prepare_prompt(
        self,
        new_requests: list[NewRequestData],
    ) -> ModelForwardInputs:
        assert len(new_requests) > 0
        input_token_list: list[torch.Tensor] = []

        # ceil division to pad to next block boundary
        new_batch = len(self.req_ids2blocks) == 0
        max_prompt_len = max([len(r.prompt_token_ids) for r in new_requests])
        if not new_batch:
            assert max_prompt_len <= self.tkv
        d = self.BLOCK_SIZE
        n = max_prompt_len if new_batch else self.tkv
        block_padding = ((n + d - 1) // d) * d
        if new_batch:
            self.tkv = block_padding

        # Internal state is managed here.
        slot_mapping = []

        self.prefill_batch.clear_requests()

        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids
            self.req_ids2left_pads[
                request_data.req_id] = self.tkv - len(prompt_tokens)
            input_token_list.append(
                torch.tensor(prompt_tokens,
                             dtype=torch.long,
                             device=torch.device("cpu")))

            # filling block table and slot mapping
            block_table_i = []
            slot_mapping_i = []
            for pos_i in range(block_padding):
                if pos_i % self.BLOCK_SIZE == 0:
                    block_number = self.free_blocks.popleft()
                    block_table_i.append(block_number)
                block_offset = pos_i % self.BLOCK_SIZE
                slot = block_number * self.BLOCK_SIZE + block_offset
                slot_mapping_i.append(slot)
            self.req_ids2blocks[request_data.req_id] = deque(block_table_i)
            slot_mapping.append(slot_mapping_i)

            # Add new requests to the cached states.
            req_id = request_data.req_id
            sampling_params = request_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=request_data.prompt_token_ids,
                sampling_params=sampling_params,
                generator=generator,
                output_token_ids=[],
            )
            self.requests[req_id] = req_state
            self.input_batch.add_request(req_state)
            self.prefill_batch.add_request(req_state)

        # Refresh sampling metadata after all request are added to the batch
        self.input_batch.refresh_sampling_metadata()
        self.prefill_batch.refresh_sampling_metadata()

        # TODO: Review this in the future
        # prefills are always of batch size 1 for this milestone
        # Also, we added an input batch just for that.
        actual_batch_size = len(input_token_list)
        assert actual_batch_size == 1
        self.model.indices = torch.ones(actual_batch_size,
                                        dtype=torch.bool,
                                        device='cpu')
        # construct tensor from list
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
        block_table = None

        # get position ids and attention mask
        # applies left padding to align with tkv of current decode batch
        # and right padding to align with the next block boundary
        input_tokens, position_ids, mask =\
            self.pad_input_ids(input_token_list, min_pad_length=block_padding)
        mask = mask.unsqueeze(1)

        # not needed for prefill
        current_tkv_mask = None
        left_padded_prompt_mask = None

        return ModelForwardInputs(
            input_tokens=input_tokens,
            input_positions=position_ids,
            input_masks=mask,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=True,
        )

    def _prepare_decode(
        self,
        cached_requests: list[CachedRequestData],
    ) -> ModelForwardInputs:
        assert len(cached_requests) > 0
        input_tokens = []
        input_positions = []
        block_table = []
        slot_mapping = []
        left_padded_prompt_mask = []
        self.model.indices = torch.ones(len(cached_requests),
                                        dtype=torch.bool,
                                        device="cpu")

        if envs_spyre.VLLM_SPYRE_RM_PADDED_BLOCKS:
            self.reduce_left_padding(cached_requests)

        for cached_request in cached_requests:
            # TODO: Will this always just be one token ID if there's no spec
            # or jump decoding?

            # adding new blocks if needed
            if self.tkv // self.BLOCK_SIZE + 1 > len(
                    self.req_ids2blocks[cached_request.req_id]):
                self.req_ids2blocks[cached_request.req_id].append(
                    self.free_blocks.popleft())
            block_table.append(self.req_ids2blocks[cached_request.req_id])
            # slot_mapping for all blocks of sequence
            start_slot = block_table[-1][-1] * self.BLOCK_SIZE
            offset = self.tkv % self.BLOCK_SIZE
            slot = [start_slot + offset]
            slot_mapping.append(slot)
            generation_token = cached_request.new_token_ids[-1]
            input_tokens.append([generation_token])
            seq_len = cached_request.num_computed_tokens
            input_positions.append([seq_len])
            left_padded_prompt_mask.append(
                self.req_ids2left_pads[cached_request.req_id])

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)
        position_ids = torch.tensor(input_positions,
                                    dtype=torch.long,
                                    device=self.device)
        left_padded_prompt_mask = torch.tensor(left_padded_prompt_mask,
                                               dtype=torch.long,
                                               device=self.device)
        # construct tensors from lists
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int64)
        block_table = torch.tensor(block_table, dtype=torch.int64)

        # not needed for decode
        mask = None

        # update tkv
        self.tkv = self.tkv + 1

        current_tkv_mask = torch.tensor([self.tkv] * len(cached_requests),
                                        dtype=torch.int64)

        return ModelForwardInputs(
            input_tokens=input_tokens,
            input_positions=position_ids,
            input_masks=mask,
            current_tkv_mask=current_tkv_mask,
            left_padded_prompt_mask=left_padded_prompt_mask,
            block_table=block_table,
            slot_mapping=slot_mapping,
            is_prompt=False,
        )

    def reduce_left_padding(self, requests: list[CachedRequestData]) -> None:

        min_left_pad = min(
            [self.req_ids2left_pads[r.req_id] for r in requests])
        n_padded_blocks = min_left_pad // self.BLOCK_SIZE

        if n_padded_blocks > 0:
            logger.debug("Number of removed blocks due to left padding: %d",
                         n_padded_blocks)

            for req in requests:
                self.req_ids2left_pads[
                    req.req_id] -= n_padded_blocks * self.BLOCK_SIZE

                # free blocks
                for _ in range(n_padded_blocks):
                    freed_block_id = self.req_ids2blocks[req.req_id].popleft()
                    self.free_blocks.append(freed_block_id)

        # update tkv
        self.tkv -= n_padded_blocks * self.BLOCK_SIZE

        return

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # left padding to align with tkv of current decode batch
        input_tokens_left, position_ids_left, mask_left =\
            super().pad_input_ids(input_ids_list, min_pad_length=self.tkv)

        # right padding to align with the next block boundary
        left_pad_len = input_tokens_left.shape[1]
        n_pads_right = min_pad_length - left_pad_len

        # set number of right pads for the next model forward pass:
        # need to be excluded before sampling tokens
        self.model.n_pads_right = n_pads_right

        if n_pads_right > 0:
            # apply right padding to input_tokens, position_ids and mask
            logger.info(
                "Right padding request of length %d tokens to %d tokens.",
                left_pad_len, min_pad_length)

            input_tokens_right = torch.tensor(
                [[self.pad_token_id for i in range(n_pads_right)]],
                device=input_tokens_left.device,
                dtype=input_tokens_left.dtype)
            input_tokens = torch.concat(
                (input_tokens_left, input_tokens_right), dim=1)

            # Note: same output with i as padding for position ids
            pos_start = position_ids_left[0][-1] + 1
            position_ids_right = torch.tensor(
                [[0 for i in range(pos_start, pos_start + n_pads_right)]],
                device=position_ids_left.device,
                dtype=position_ids_left.dtype)
            position_ids = torch.concat(
                (position_ids_left, position_ids_right), dim=1)

            # pad left padded mask with -inf to the next block boundary
            mask = torch.nn.functional.pad(mask_left,
                                           (0, n_pads_right, 0, n_pads_right),
                                           value=-torch.inf)

            # lower triangle: 0.0, upper triangle -inf
            mask_pads = torch.zeros(n_pads_right, n_pads_right)
            mask_pads[~torch.tril(torch.ones(n_pads_right, n_pads_right)).bool(
            )] = float('-inf')

            # insert triangular matrix for right pads
            mask[:, -n_pads_right:, -n_pads_right:] = mask_pads.unsqueeze(0)
        else:
            # no right padding needed
            input_tokens = input_tokens_left
            position_ids = position_ids_left
            mask = mask_left

        return input_tokens, position_ids, mask

    def prepare_model_input(
            self, scheduler_output: SchedulerOutput) -> ModelForwardInputs:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare and return input tensors.
        if is_prompt:
            model_inputs = \
                self._prepare_prompt(scheduler_output.scheduled_new_reqs)
        else:
            model_inputs = \
                self._prepare_decode(scheduler_output.scheduled_cached_reqs)

        return model_inputs

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: SchedulerOutput,
        **kwargs,
    ) -> ModelRunnerOutput:

        t0 = time.time()

        self._update_states(scheduler_output)
        # TODO: change to EMPTY_MODEL_RUNNER_OUTPUT, right now this
        # will be a breaking change, or clumsy to make retrocompatible
        # with conditional import
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return CBSpyreModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                spec_token_ids=None,
                logprobs=None,
                prompt_logprobs_dict={},
                tkv=0,
            )

        model_input = self.prepare_model_input(scheduler_output)

        # Marking dimensions static/dynamic
        if model_input.is_prompt:

            # batch static (batch size 1)
            torch._dynamo.mark_static(model_input.input_tokens, 0)
            torch._dynamo.mark_static(model_input.slot_mapping, 0)
            torch._dynamo.mark_static(model_input.input_positions, 0)
            torch._dynamo.mark_static(model_input.input_masks, 0)

            # sequence dynamic
            torch._dynamo.mark_dynamic(model_input.input_tokens, 1)
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 1)
            torch._dynamo.mark_dynamic(model_input.input_positions, 1)
            torch._dynamo.mark_dynamic(model_input.input_masks, 2)
            torch._dynamo.mark_dynamic(model_input.input_masks, 3)

        # decode
        else:
            # mask is no longer used here

            # batch dynamic
            torch._dynamo.mark_dynamic(model_input.input_tokens, 0)
            torch._dynamo.mark_dynamic(model_input.block_table, 0)
            torch._dynamo.mark_dynamic(model_input.slot_mapping, 0)
            torch._dynamo.mark_dynamic(model_input.input_positions, 0)
            torch._dynamo.mark_dynamic(model_input.current_tkv_mask, 0)
            torch._dynamo.mark_dynamic(model_input.left_padded_prompt_mask, 0)

            # sequence
            torch._dynamo.mark_static(model_input.input_tokens, 1)  # always 1
            torch._dynamo.mark_dynamic(model_input.block_table, 1)
            torch._dynamo.mark_static(model_input.slot_mapping, 1)  # always 1
            torch._dynamo.mark_static(model_input.input_positions,
                                      1)  # always 1

        # Execute the model
        hidden_states = self.model(
            input_ids=model_input.input_tokens,
            positions=model_input.input_positions,
            masks=model_input.input_masks,
            is_prompt=model_input.is_prompt,
            current_tkv_mask=model_input.current_tkv_mask,
            left_padded_prompt_mask=model_input.left_padded_prompt_mask,
            block_table=model_input.block_table,
            slot_mapping=model_input.slot_mapping)

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token.
        # TODO: review this, once we can prefill and decode at the same step
        sampling_metadata = self.prefill_batch.sampling_metadata \
            if model_input.is_prompt else self.input_batch.sampling_metadata
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )
        t1 = time.time() - t0
        logger.debug("t_token: %.2fms", (t1 * 1000))

        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0
        scheduled_req = (scheduler_output.scheduled_new_reqs if is_prompt else
                         scheduler_output.scheduled_cached_reqs)
        # since same order as in _prepare_prompt/decode req_ids2idx not needed
        req_ids = [req.req_id for req in scheduled_req]
        req_id_to_index = {req_id: i for i, req_id in enumerate(req_ids)}

        model_output = CBSpyreModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=output.sampled_token_ids.tolist(),
            spec_token_ids=None,
            logprobs=(output.logprobs_tensors.tolists()
                      if output.logprobs_tensors else None),
            prompt_logprobs_dict={req_id: None
                                  for req_id in req_ids
                                  },  # TODO(wallas?): prompt logprobs too
            tkv=self.tkv,
        )
        return model_output

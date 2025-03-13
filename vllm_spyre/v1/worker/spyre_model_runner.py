import time
from dataclasses import dataclass
from typing import (TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple,
                    Type, TypeVar)

import torch
from torch import nn
from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.utils import is_pin_memory_available
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_sampling_metadata_broadcastable_dict,
    _init_sampling_metadata_from_tensor_dict)

from vllm_spyre.model_executor.model_loader.spyre import get_spyre_model
from vllm_spyre.platform import SpyrePlatform

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.model_executor.pooling_metadata import PoolingMetadata

from vllm.v1.core.scheduler import (CachedRequestData, NewRequestData,
                                    SchedulerOutput)
from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

TModelInputForSpyre = TypeVar('TModelInputForSpyre',
                              bound="ModelInputForSpyre")


@dataclass(frozen=True)
class ModelInputForSpyre(ModelRunnerInputBase):
    """
    Used by the SpyreModelRunner.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    input_masks: Optional[torch.Tensor] = None
    sampling_metadata: Optional[SamplingMetadata] = None
    pooling_metadata: Optional["PoolingMetadata"] = None
    is_prompt: Optional[bool] = None
    # unused
    virtual_engine: Optional[int] = None

    def as_broadcastable_tensor_dict(self) -> Dict[str, Any]:
        tensor_dict = {
            "input_tokens": self.input_tokens,
            "input_positions": self.input_positions,
            "input_masks": self.input_masks,
            "is_prompt": self.is_prompt,
        }
        _add_sampling_metadata_broadcastable_dict(tensor_dict,
                                                  self.sampling_metadata)
        return tensor_dict

    @classmethod
    def from_broadcasted_tensor_dict(
        cls: Type[TModelInputForSpyre],
        tensor_dict: Dict[str, Any],
        attn_backend: Optional["AttentionBackend"] = None,
    ) -> TModelInputForSpyre:
        tensor_dict = _init_sampling_metadata_from_tensor_dict(tensor_dict)
        return cls(**tensor_dict)


class SpyreModelRunner(ModelRunnerBase[ModelInputForSpyre]):

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool,
    ):
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.is_driver_worker = is_driver_worker

        self.pad_token_id = 0
        if model_config is not None:
            if model_config.hf_config is not None:
                self.pad_token_id = getattr(model_config.hf_config,
                                            "pad_token_id", None) or 0
            if model_config.get_sliding_window():
                logger.warning("Sliding window is not supported on Spyre. "
                               "The model will run without sliding window.")
        self.device_config = (device_config
                              if device_config is not None else DeviceConfig())
        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()
        # position_ids of all the sequences in current batch
        self._position_ids: torch.Tensor = None
        # attention masks of all the sequences in current batch
        self._mask: torch.Tensor = None
        # mapping: request id to index in batch
        self._req_ids2idx: dict = {}
        # Lazy initialization: after load_model.
        self.model: nn.Module
        # mapping of request ID to sampling params
        self._sampling_params_by_request: dict[str, SamplingParams] = {}
        self._max_logprobs: Optional[int] = None

    def get_model(self) -> nn.Module:
        return self.model

    def load_model(self, prompt_lens: Iterable[int],
                   num_decode_tokens: Iterable[int]) -> None:
        max_pad_length = max(prompt_lens)
        max_decode_length = max(num_decode_tokens)
        self.model = get_spyre_model(self.model_config,
                                     parallel_config=self.parallel_config,
                                     max_prompt_length=max_pad_length,
                                     max_decode_length=max_decode_length)

    @property
    def vocab_size(self) -> int:
        return self.model.model.config.src_vocab_size

    def _prepare_prompt(
        self,
        new_requests: List[NewRequestData],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
        assert len(new_requests) > 0
        input_token_list: List[torch.Tensor] = []

        # find warmup shape to be used for padding and batching
        spyre_warmup_shapes = current_platform.get_warmup_shapes()
        applicable_spyre_warmup_shapes = [
            shape for shape in spyre_warmup_shapes
            if len(new_requests) <= shape['batch_size']
        ]
        for request_data in new_requests:
            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids
            new_tokens = request_data.sampling_params.max_tokens\
                  if request_data.sampling_params is not None else 0

            updated_spyre_warmup_shapes = [
                shape for shape in applicable_spyre_warmup_shapes
                if len(prompt_tokens) <= shape['prompt_length']
                and new_tokens <= shape['new_tokens']
            ]
            applicable_spyre_warmup_shapes = updated_spyre_warmup_shapes

        assert applicable_spyre_warmup_shapes, \
            "No shapes available to run prefill batch. (This should not happen)"

        # If multiple warmup shapes apply, the first one is selected.
        # For improving performance, the warmup shapes in scheduler_config
        # are ordered by "processing speed".
        min_pad_length_batch = applicable_spyre_warmup_shapes[0][
            'prompt_length']
        padded_batch_size = applicable_spyre_warmup_shapes[0]['batch_size']

        # Internal state is reset here.
        # We don't support continuous batching, so we know all previous requests
        # have finished decoding.
        self._req_ids2idx = {}
        self._sampling_params_by_request = {}
        self._max_logprobs = None
        for idx, request_data in enumerate(new_requests):
            self._req_ids2idx[request_data.req_id] = idx
            self._sampling_params_by_request[
                request_data.req_id] = request_data.sampling_params

            # retrieve initial (unpadded) tokens
            prompt_tokens = request_data.prompt_token_ids

            input_token_list.append(
                torch.tensor(prompt_tokens,
                             dtype=torch.long,
                             device=torch.device("cpu")))
        # Cache the max requested logprobs for this batch
        logprobs: list[int] = [
            sampling_params.logprobs
            for sampling_params in self._sampling_params_by_request.values()
            if sampling_params is not None
            and sampling_params.logprobs is not None
        ]
        if logprobs:
            self._max_logprobs = max(logprobs)

        actual_batch_size = len(input_token_list)
        self.model.indices = torch.cat([
            torch.ones(actual_batch_size, dtype=torch.bool, device='cpu'),
            torch.zeros(padded_batch_size - actual_batch_size,
                        dtype=torch.bool,
                        device='cpu')
        ])

        # padding to compiled batch size
        while len(input_token_list) < padded_batch_size:
            input_token_list.append(
                torch.zeros(min_pad_length_batch,
                            dtype=torch.long,
                            device=torch.device("cpu")))

        # get position ids and attention mask
        input_tokens, self._position_ids, self._mask = self.pad_input_ids(
            input_token_list, min_pad_length=min_pad_length_batch)

        seq_lens = [t.shape[0] for t in input_token_list]

        return input_tokens, self._position_ids, self._mask, seq_lens

    def _prepare_decode(
        self,
        cached_requests: List[CachedRequestData],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(cached_requests) > 0
        input_tokens: List[List[int]] = [
            [0] for _ in range(self._position_ids.shape[0])
        ]

        for cached_request in cached_requests:
            # TODO: Will this always just be one token ID if there's no spec
            # or jump decoding?
            generation_token = cached_request.new_token_ids[-1]
            input_tokens[self._req_ids2idx[cached_request.req_id]] = [
                generation_token
            ]

        # update position ids and attention mask
        self._update_position_ids()
        self._update_mask()

        input_tokens = torch.tensor(input_tokens,
                                    dtype=torch.long,
                                    device=self.device)

        return input_tokens, self._position_ids, self._mask

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

    def make_model_input_from_broadcasted_tensor_dict(
            self, tensor_dict: Dict[str, Any]) -> ModelInputForSpyre:
        return ModelInputForSpyre.from_broadcasted_tensor_dict(tensor_dict)

    def prepare_model_input(
            self, scheduler_output: SchedulerOutput) -> ModelInputForSpyre:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        # Also assuming that new sequences are prefills
        is_prompt = len(scheduler_output.scheduled_new_reqs) > 0

        # Prepare input tensors.
        if is_prompt:

            # Assert no running requests
            assert len(scheduler_output.scheduled_cached_reqs) == 0

            (input_tokens, input_positions, input_masks,
             _) = self._prepare_prompt(scheduler_output.scheduled_new_reqs)
            # seq_lens = [
            #     input_tokens.shape[1] for i in range(input_tokens.shape[0])
            # ]
            num_reqs = len(scheduler_output.scheduled_new_reqs)
        else:
            # updating indices: set indices of newly finished sequences False
            if scheduler_output.finished_req_ids:
                for seq_id in scheduler_output.finished_req_ids:
                    if seq_id in self._req_ids2idx:
                        self.model.indices[self._req_ids2idx[seq_id]] = False
            (input_tokens, input_positions,
             input_masks) = self._prepare_decode(
                 scheduler_output.scheduled_cached_reqs)
            # seq_lens = []
            num_reqs = len(scheduler_output.scheduled_cached_reqs)

        # TODO: Build the rest of the SamplingMetadata correctly
        dummy_tensors = lambda v: torch.full(
            (num_reqs, ), v, device=self.device)
        dummy_metadata = SamplingMetadata(
            temperature=dummy_tensors(0.0),
            all_greedy=False,
            all_random=False,
            top_p=None,
            top_k=None,
            min_p=None,
            generators={},
            max_num_logprobs=self._max_logprobs,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[[] for _ in range(num_reqs)],
            min_tokens={},
            logit_bias=[None for _ in range(num_reqs)],
            allowed_token_ids_mask=None,
        )

        return ModelInputForSpyre(input_tokens=input_tokens,
                                  input_positions=input_positions,
                                  input_masks=input_masks,
                                  sampling_metadata=dummy_metadata,
                                  is_prompt=is_prompt)

    @SpyrePlatform.inference_mode()
    def execute_model(
        self,
        scheduler_output: Optional[SchedulerOutput] = None,
        **kwargs
    ) -> Optional[ModelRunnerOutput]:
        """
        Runs model execution for either warm up or real inference.
        """

        t0 = time.time()
        warmup_mode = kwargs.get("warmup_mode", False)

        if warmup_mode:
            prompt_len = kwargs["prompt_len"]
            num_decode_tokens = kwargs["num_decode_tokens"]
            batch_size = kwargs["batch_size"]
            num_scheduled_tokens = {}
            total_num_scheduled_tokens = 0
            dummy_requests = []
            num_scheduled_tokens = {}
            total_num_scheduled_tokens = 0
            for i in range(batch_size):
                dummy_requests.append(
                    NewRequestData(
                        req_id=f"warmup-{i}",
                        prompt_token_ids=[1] * prompt_len,
                        prompt="test",
                        mm_inputs=[],
                        mm_hashes=[],
                        mm_positions=[],
                        sampling_params=SamplingParams(max_tokens=num_decode_tokens),
                        block_ids=[0],
                        num_computed_tokens=0,
                        lora_request=None,
                    ))
                num_scheduled_tokens[i] = prompt_len
                total_num_scheduled_tokens += num_scheduled_tokens[i]

            scheduler_output = SchedulerOutput(
                scheduled_new_reqs=dummy_requests,
                scheduled_cached_reqs=[],
                num_scheduled_tokens=num_scheduled_tokens,
                total_num_scheduled_tokens=total_num_scheduled_tokens,
                scheduled_spec_decode_tokens={},
                scheduled_encoder_inputs={},
                num_common_prefix_blocks=0,
                finished_req_ids=set(),
                free_encoder_input_ids=[],
            )

        model_input = self.prepare_model_input(scheduler_output)
        input_tokens = model_input.input_tokens
        input_positions = model_input.input_positions
        input_masks = model_input.input_masks
        is_prompt = model_input.is_prompt

        hidden_states = self.model(
            input_ids=input_tokens,
            positions=input_positions,
            masks=input_masks,
            is_prompt=is_prompt,
        )

        # Only perform sampling in the driver worker.
        if not self.is_driver_worker:
            return []

        # Compute the logits.
        logits = self.model.compute_logits(hidden_states, None)

        # Sample the next token.
        output: SamplerOutput = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        t1 = time.time() - t0
        print("[spyre_model_runner:execute_model] t_token: %.2fms" %
              (t1 * 1000))

        model_output = ModelRunnerOutput(
            req_ids=list(self._req_ids2idx.keys()),
            req_id_to_index=self._req_ids2idx,
            sampled_token_ids=output.sampled_token_ids.tolist(),
            spec_token_ids=None,
            logprobs=output.logprobs_tensors.tolists()
            if output.logprobs_tensors else None,
            prompt_logprobs_dict={
                req_id: None
                for req_id in self._req_ids2idx
            }  # TODO: prompt logprobs too
        )
        return model_output

    def _prepare_pad_input_ids(
        self,
        input_ids_list: List[torch.Tensor],
        min_pad_length: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                print(f"[SpyreModelRunner] INFO: Padding request of length "
                      f"{seq_len} tokens to {max_len} tokens.")
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
        input_ids_list: List[torch.Tensor],
        min_pad_length: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        padded_input_ids_list, mask_list, position_ids_list = self.\
            _prepare_pad_input_ids(input_ids_list, min_pad_length)

        input_ids = torch.stack(padded_input_ids_list)
        mask = torch.stack(mask_list).bool()
        # this is a causal mask for generation
        mask = (mask.unsqueeze(-1) == mask.unsqueeze(-2)).tril()
        mask = torch.where(mask.logical_not(), -torch.inf, 0.0)
        mask = mask.to(self.model.dtype)
        position_ids = torch.stack(position_ids_list)

        return input_ids, position_ids, mask

    def _raw_model_forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[List[Tuple[torch.Tensor,
                                                   torch.Tensor]]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        attn_algorithm: Optional[str] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor,
                                                 torch.Tensor]]]]:

        return self.model.model(input_ids,
                                mask=mask,
                                position_ids=position_ids,
                                past_key_value_states=past_key_value_states,
                                use_cache=use_cache,
                                only_last_token=only_last_token,
                                attn_algorithm=attn_algorithm)

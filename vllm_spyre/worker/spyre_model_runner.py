import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, TypeVar

import torch
from torch import nn
from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor import SamplingMetadata
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors, SequenceGroupMetadata
from vllm.utils import is_pin_memory_available
from vllm.worker.model_runner_base import (
    ModelRunnerBase, ModelRunnerInputBase,
    _add_sampling_metadata_broadcastable_dict,
    _init_sampling_metadata_from_tensor_dict)

from vllm_spyre.model_executor.model_loader.spyre import SpyreCausalLM

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionBackend
    from vllm.model_executor.pooling_metadata import PoolingMetadata

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

    def as_broadcastable_tensor_dict(self) -> dict[str, Any]:
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
        cls: type[TModelInputForSpyre],
        tensor_dict: dict[str, Any],
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
        self.spyre_warmup_shapes = current_platform.get_warmup_shapes(
            self.scheduler_config)

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

    def _prepare_prompt(
        self,
        seq_group_metadata_list: list[SequenceGroupMetadata],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        assert len(seq_group_metadata_list) > 0
        input_token_list: list[torch.Tensor] = []

        # find warmup shape to be used for padding and batching
        applicable_spyre_warmup_shapes = [
            shape for shape in self.spyre_warmup_shapes
            if len(seq_group_metadata_list) <= shape['batch_size']
        ]
        for seq_group_metadata in seq_group_metadata_list:
            seq_data = seq_group_metadata.seq_data[list(
                seq_group_metadata.seq_data.keys())[0]]
            # retrieve initial (unpadded) tokens
            prompt_tokens = seq_data.get_token_ids()
            new_tokens = seq_group_metadata.sampling_params.max_tokens\
                  if seq_group_metadata.sampling_params is not None else 0

            updated_spyre_warmup_shapes = [
                shape for shape in applicable_spyre_warmup_shapes
                if len(prompt_tokens) <= shape['prompt_length']
                and new_tokens <= shape['new_tokens']
            ]
            applicable_spyre_warmup_shapes = updated_spyre_warmup_shapes

        assert applicable_spyre_warmup_shapes

        # If multiple warmup shapes apply, the first one is selected.
        # For improving performance, the warmup shapes in scheduler_config
        # are ordered by "processing speed".
        min_pad_length_batch = applicable_spyre_warmup_shapes[0][
            'prompt_length']
        padded_batch_size = applicable_spyre_warmup_shapes[0]['batch_size']

        self._req_ids2idx = {}
        for idx, seq_group_metadata in enumerate(seq_group_metadata_list):
            assert seq_group_metadata.is_prompt
            self._req_ids2idx[seq_group_metadata.request_id] = idx
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            # retrieve initial (unpadded) tokens
            prompt_tokens = seq_data.get_token_ids()

            input_token_list.append(
                torch.tensor(prompt_tokens,
                             dtype=torch.long,
                             device=torch.device("cpu")))

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
        seq_group_metadata_list: list[SequenceGroupMetadata],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert len(seq_group_metadata_list) > 0
        input_tokens: list[list[int]] = [
            [0] for _ in range(self._position_ids.shape[0])
        ]

        for seq_group_metadata in seq_group_metadata_list:
            assert not seq_group_metadata.is_prompt
            seq_ids = list(seq_group_metadata.seq_data.keys())
            assert len(seq_ids) == 1
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            generation_token = seq_data.get_last_token_id()
            input_tokens[self._req_ids2idx[seq_group_metadata.request_id]] = [
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
            self, tensor_dict: dict[str, Any]) -> ModelInputForSpyre:
        return ModelInputForSpyre.from_broadcasted_tensor_dict(tensor_dict)

    def prepare_model_input(
        self,
        seq_group_metadata_list: list[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[list[str]] = None
    ) -> ModelInputForSpyre:

        # NOTE: We assume that all sequences in the group are all prompts or
        # all decodes.
        is_prompt = seq_group_metadata_list[0].is_prompt
        # Prepare input tensors.
        if is_prompt:
            (input_tokens, input_positions, input_masks,
             _) = self._prepare_prompt(seq_group_metadata_list)
            seq_lens = [
                input_tokens.shape[1] for i in range(input_tokens.shape[0])
            ]
        else:
            # updating indices: set indices of newly finished sequences False
            if finished_requests_ids:
                for seq_id in finished_requests_ids:
                    # ignore requests that are not in the batch, eg. requests
                    # cancelled while waiting
                    if (idx := self._req_ids2idx.get(seq_id)) is not None:
                        self.model.indices[idx] = False
            (input_tokens, input_positions,
             input_masks) = self._prepare_decode(seq_group_metadata_list)
            seq_lens = []

        sampling_metadata = SamplingMetadata.prepare(
            seq_group_metadata_list,
            seq_lens,
            # query_lens is not needed if chunked prefill is not
            # supported. Since Spyre worker doesn't support chunked prefill
            # just use seq_lens instead.
            seq_lens,
            self.device,
            self.pin_memory,
            self.get_generators(finished_requests_ids))

        return ModelInputForSpyre(input_tokens=input_tokens,
                                  input_positions=input_positions,
                                  input_masks=input_masks,
                                  sampling_metadata=sampling_metadata,
                                  is_prompt=is_prompt)

    def execute_model(
        self,
        model_input: ModelInputForSpyre,
        kv_caches: Optional[list[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[list[SamplerOutput]]:

        t0 = time.time()

        if num_steps > 1:
            raise ValueError(
                "SpyreModelRunner does not support multi-step execution.")

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
        logits = self.model.compute_logits(hidden_states,
                                           model_input.sampling_metadata)

        # Sample the next token.
        output = self.model.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )
        t1 = time.time() - t0
        print("[spyre_model_runner:execute_model] t_token: %.2fms" %
              (t1 * 1000))

        return [output]

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
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        padded_input_ids_list, mask_list, position_ids_list = self.\
            _prepare_pad_input_ids(input_ids_list, min_pad_length)

        input_ids = torch.stack(padded_input_ids_list)
        mask = torch.stack(mask_list).bool()
        # this is a causal mask for generation
        mask = (mask.unsqueeze(-1) == mask.unsqueeze(-2)).tril()
        mask = torch.where(mask.logical_not(), -torch.inf, 0.0)
        mask = mask.to(self.model.model.dtype)
        position_ids = torch.stack(position_ids_list)

        return input_ids, position_ids, mask

    def _raw_model_forward(
        self,
        input_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value_states: Optional[list[tuple[torch.Tensor,
                                                   torch.Tensor]]] = None,
        use_cache: bool = False,
        only_last_token: bool = False,
        attn_algorithm: Optional[str] = None
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor,
                                                 torch.Tensor]]]]:

        # For AIU, we should not assign attn_algorithm
        extra_kargs = {}
        if attn_algorithm is not None:
            extra_kargs['attn_algorithm'] = attn_algorithm

        return self.model.model.model(
            input_ids,
            mask=mask,
            position_ids=position_ids,
            past_key_value_states=past_key_value_states,
            use_cache=use_cache,
            only_last_token=only_last_token,
            **extra_kargs)

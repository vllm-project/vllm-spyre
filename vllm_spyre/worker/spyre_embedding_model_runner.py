import time
from collections.abc import Iterable
from typing import Optional

import torch
from transformers import AutoModel
from vllm.config import (DeviceConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import Pooler, PoolingType
from vllm.model_executor.pooling_metadata import PoolingMetadata
from vllm.pooling_params import PoolingParams
from vllm.sequence import (IntermediateTensors, PoolerOutput, SequenceData,
                           SequenceGroupMetadata)

import vllm_spyre.envs as envs_spyre

from .spyre_model_runner import ModelInputForSpyre, SpyreModelRunner

logger = init_logger(__name__)

BACKEND_LIST = ['sendnn', 'inductor']


class SpyreEmbeddingModelRunner(SpyreModelRunner):

    # Map of request_id -> generator used for seeded random sampling
    generators: dict[str, torch.Generator] = {}

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        device_config: DeviceConfig,
        is_driver_worker: bool,
    ):
        super().__init__(model_config=model_config,
                         parallel_config=parallel_config,
                         scheduler_config=scheduler_config,
                         device_config=device_config,
                         is_driver_worker=is_driver_worker)

        pooler_config = model_config.pooler_config
        self.pooler = Pooler.from_config_with_defaults(
            pooler_config,
            pooling_type=PoolingType.CLS,
            normalize=True,
            softmax=False)

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

    def prepare_input_tensors(
        self,
        seq_group_metadata_list: list[SequenceGroupMetadata],
        finished_requests_ids: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, PoolingMetadata]:
        # NOTE: We assume that all sequences in the group are all prompts
        (input_tokens, input_positions, input_masks,
         seq_lens) = self._prepare_prompt(seq_group_metadata_list)

        pooling_metadata = self._prepare_pooling(
            seq_group_metadata_list=seq_group_metadata_list,
            prompt_lens=seq_lens)
        return (input_tokens, input_positions, input_masks, pooling_metadata)

    def _prepare_pooling(
        self,
        seq_group_metadata_list: list[SequenceGroupMetadata],
        prompt_lens: list[int],
    ) -> PoolingMetadata:
        """Prepare PoolingMetadata for the sequence group metadata list."""
        seq_groups: list[tuple[list[int], PoolingParams]] = []
        for i, seq_group_metadata in enumerate(seq_group_metadata_list):
            seq_ids = list(seq_group_metadata.seq_data.keys())
            pooling_params = seq_group_metadata.pooling_params
            seq_groups.append((seq_ids, pooling_params))

        seq_data: dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        pooling_metadata = PoolingMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
        )

        return pooling_metadata

    def pad_input_ids(
        self,
        input_ids_list: list[torch.Tensor],
        min_pad_length: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        padded_input_ids_list, mask_list, position_ids_list = self.\
            _prepare_pad_input_ids(input_ids_list, min_pad_length)

        input_ids = torch.stack(padded_input_ids_list)
        mask = torch.stack(mask_list)
        position_ids = torch.stack(position_ids_list)

        return input_ids, position_ids, mask

    def prepare_model_input(
        self,
        seq_group_metadata_list: list[SequenceGroupMetadata],
        virtual_engine: int = 0,
        finished_requests_ids: Optional[list[str]] = None
    ) -> ModelInputForSpyre:

        (input_tokens, input_positions, input_masks,
         pooling_metadata) = self.prepare_input_tensors(
             seq_group_metadata_list, finished_requests_ids)

        return ModelInputForSpyre(input_tokens=input_tokens,
                                  input_positions=input_positions,
                                  input_masks=input_masks,
                                  pooling_metadata=pooling_metadata)

    def execute_model(
        self,
        model_input: ModelInputForSpyre,
        kv_caches: Optional[list[torch.Tensor]] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_steps: int = 1,
        **kwargs,
    ) -> Optional[list[PoolerOutput]]:

        t0 = time.time()

        outputs = self.model(
            input_ids=model_input.input_tokens,
            # Let the Embedding layer use it's default
            # because the rules can be a bit different
            # e.g. For Roberta models the inputs start
            # at padding_inx +1
            #position_ids=input_positions,
            attention_mask=model_input.input_masks)
        hidden_states = outputs["last_hidden_state"]

        unpadded = []
        max_len = hidden_states.shape[1]

        if model_input.pooling_metadata is not None:
            for i, seq_len in enumerate(
                    model_input.pooling_metadata.prompt_lens):
                unpadded.append(hidden_states[i, max_len - seq_len:, :])

        hidden_states = torch.concat(unpadded)

        pooler_output = self.pooler(
            hidden_states=hidden_states,
            pooling_metadata=model_input.pooling_metadata)

        t1 = time.time() - t0
        print("[spyre_model_runner:execute_model] t_token: %.2fms" %
              (t1 * 1000))

        return [pooler_output]

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

        hidden_states, _ = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            #position_ids=position_ids
        )
        return hidden_states, None

# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

# Based on vllm/vllm/v1/worker/gpu_input_batch.py

from dataclasses import dataclass
from typing import Optional

import torch
from vllm.pooling_params import PoolingParams
from vllm.v1.pool.metadata import PoolingMetadata

from vllm_spyre.v1.worker.spyre_base_input_batch import (BaseInputBatch,
                                                         BaseRequestState)


@dataclass
class PoolingRequestState(BaseRequestState):

    pooling_params: PoolingParams = PoolingParams()

    def __post_init__(self):
        self.num_prompt_tokens = len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        return self.num_prompt_tokens


class PoolingInputBatch(BaseInputBatch[PoolingRequestState]):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
    ):
        super().__init__(
            max_num_reqs,
            max_model_len,
            device,
            pin_memory,
            vocab_size,
        )
        self.pooling_params: dict[str, PoolingParams] = {}

    def get_available_index(self) -> Optional[int]:
        return self._num_requests

    def add_request(
        self,
        request: "PoolingRequestState",
        req_index: Optional[int] = None,
    ) -> int:

        req_index = super().add_request(request, req_index)

        assert request.pooling_params is not None
        self.pooling_params[request.req_id] = request.pooling_params
        return req_index

    def clear_requests(self):
        '''
        Clear the batch, mostly used by static batching
        '''
        super().clear_requests()
        self.pooling_params = {}

    def remove_request(self, req_id: str):

        req_index = super().remove_request(req_id)
        if req_index is None:
            return

        self.pooling_params.pop(req_id, None)

    def make_pooling_metadata(self) -> PoolingMetadata:
        prompt_token_ids = self._make_prompt_token_ids_tensor()

        # Note, for now this assumes that all request in the batch
        # are either sampling or pooling requests
        assert len(self.requests_ids) == len(self.pooling_params)
        pooling_params = [
            self.pooling_params[req_id] for req_id in self.requests_ids
        ]

        return PoolingMetadata(
            prompt_lens=torch.from_numpy(self._get_num_prompt_tokens()).to(
                self.device),
            prompt_token_ids=prompt_token_ids,
            pooling_params=pooling_params,
        )

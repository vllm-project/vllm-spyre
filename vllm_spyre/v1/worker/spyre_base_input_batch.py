# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

# Based on vllm/vllm/v1/worker/gpu_input_batch.py

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar, cast

import numpy as np
import torch

_SAMPLING_EPS = 1e-5


@dataclass
class BaseRequestState:

    req_id: str
    prompt_token_ids: list[int]
    left_padding: int = 0  # Defaults to 0, i. e. not padding

    @property
    @abstractmethod
    def num_tokens(self) -> int:
        raise NotImplementedError


RequestState = TypeVar("RequestState", bound=BaseRequestState)


class BaseInputBatch(Generic[RequestState]):

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
    ):
        assert device.type == 'cpu'
        # NOTE: max_num_reqs should be consistent with the warmup shapes
        self.max_num_reqs = max_num_reqs
        self.max_model_len = max_model_len
        self.device = device
        self.pin_memory = pin_memory
        self.vocab_size = vocab_size

        self._req_ids: list[Optional[str]] = [None] * max_num_reqs
        self.req_id_to_index: dict[str, int] = {}

        # TODO(woosuk): This buffer could be too large if max_model_len is big.
        # Find a way to reduce the CPU memory usage.
        # This buffer is not directly transferred to the GPU, so it does not
        # need to be pinned.
        self.token_ids_cpu_tensor = torch.zeros(
            (max_num_reqs, max_model_len),
            device="cpu",
            dtype=torch.int32,
            pin_memory=False,
        )
        self.token_ids_cpu = self.token_ids_cpu_tensor.numpy()
        self.num_prompt_tokens = np.zeros(max_num_reqs, dtype=np.int32)

        # Initialize with max number of requests
        self.padded_batch_size = self.max_num_reqs

        # Keep tracking of number of requests
        self._num_requests = 0

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def get_available_index(self) -> int:
        raise NotImplementedError

    def add_request(
        self,
        request: "RequestState",
        req_index: Optional[int] = None,
    ) -> None:
        raise NotImplementedError

    def _add_request(
        self,
        request: "RequestState",
        req_index: Optional[int] = None,
    ) -> int:
        if req_index is None:
            req_index = self.get_available_index()
        assert req_index is not None
        assert req_index < self.max_num_reqs

        req_id = request.req_id
        self._req_ids[req_index] = req_id

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens

        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids

        self._num_requests += 1
        assert self._num_requests <= self.max_num_reqs
        return req_index

    def clear_requests(self):
        '''
        Clear the batch, mostly used by static batching
        '''
        self.req_id_to_index = {}

        self._req_ids = [None] * self.max_num_reqs

        self._num_requests = 0

    def remove_request(self, req_id: str):
        raise NotImplementedError

    def _remove_request(self, req_id: str) -> Optional[int]:
        '''
        Free a slot of a request from the batch
        
        It does the following:
        - mask out the removed request.
        - Remove reference from the sets that track the type of parameter 
          e.g. greeedy_reqs 
        - Update some containers by reference to update the sampling parameters
          e.g. req_output_token_ids
        
        For the continuous batching, the removed request indices can be 
        overwritten by new requests
        '''
        req_index = self.req_id_to_index.pop(req_id, None)

        if req_index is None:
            return None

        # Remove the references

        self._req_ids[req_index] = None
        self._num_requests -= 1
        return req_index

    def refresh_metadata(self):
        pass

    def _get_num_prompt_tokens(self) -> np.ndarray:
        return self.num_prompt_tokens

    def _get_token_ids(self) -> np.ndarray:
        return self.token_ids_cpu[:self._num_requests]

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:

        num_prompt_tokens = self._get_num_prompt_tokens()
        max_prompt_len = num_prompt_tokens.max()
        prompt_token_ids_tensor = torch.empty(
            (self._num_requests, max_prompt_len),
            device=self.device,
            dtype=torch.int64,
        )
        prompt_token_ids = prompt_token_ids_tensor.numpy()
        prompt_token_ids[:] = self._get_token_ids()[:, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.

        for i in range(self._num_requests):
            prompt_token_ids[i, num_prompt_tokens[i]:] = self.vocab_size
        return prompt_token_ids_tensor

    def get_req_index(self, req_id):
        return self.req_id_to_index.get(req_id)

    @property
    def num_reqs(self) -> int:
        return self._num_requests

    @property
    def requests_ids(self) -> list[str]:
        return list(self.req_id_to_index.keys())

    @property
    def sorted_requests_ids(self) -> list[str]:
        return sorted(self.req_id_to_index,
                      key=self.req_id_to_index.get)  # type: ignore

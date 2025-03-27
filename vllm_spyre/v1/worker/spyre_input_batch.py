# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

# Based on vllm/vllm/v1/worker/gpu_input_batch.py

from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.sample.metadata import SamplingMetadata

_SAMPLING_EPS = 1e-5


@dataclass
class CachedRequestState:

    req_id: str
    prompt_token_ids: list[int]
    prompt: Optional[str]
    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    output_token_ids: list[int]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class InputBatch:
    '''
    This class was based on the InputBatch for GPU of vLLM V1.
    
    The implementation of vLLM was designed to track the request parameters
    and does some optimizations to keep the data organized tight. It also
    build the sampling parameters and do lazy allocations when possible.
    
    For the Spyre, we do something similar, however we do not worry (for now)
    the transfer data from CPU -> GPU as vLLM does. One key difference between 
    those implementations is that we have a mask for active request based on 
    the indices stored in `model_indices_mask`. Sometimes we need to check it
    to get the correct index of a request see `get_unpadded_output_indices`. 
    
    
    Since we do not yet support continuous batch, the correct usage of this
    class consists in add requests and clear the whole batch before process
    more requests. 
    '''

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

        self._req_ids: list[Optional[str]] = []
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

        # Sampling-related.
        self.temperature = torch.empty((max_num_reqs, ),
                                       dtype=torch.float32,
                                       device=device)
        self.temperature_cpu = self.temperature.numpy()
        self.greedy_reqs: set[str] = set()
        self.random_reqs: set[str] = set()

        self.top_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)

        self.top_p_cpu = self.top_p.numpy()
        self.top_p_reqs: set[str] = set()

        self.top_k = torch.empty((max_num_reqs, ),
                                 dtype=torch.int32,
                                 device=device)
        self.top_k_cpu = self.top_k.numpy()
        self.top_k_reqs: set[str] = set()

        self.min_p = torch.empty((max_num_reqs, ),
                                 dtype=torch.float32,
                                 device=device)
        self.min_p_cpu = self.min_p.numpy()
        self.min_p_reqs: set[str] = set()

        # Frequency penalty related data structures
        self.frequency_penalties = torch.empty((max_num_reqs, ),
                                               dtype=torch.float,
                                               device=device)
        self.frequency_penalties_cpu = \
            self.frequency_penalties.numpy()
        self.frequency_penalties_reqs: set[str] = set()

        # Presence penalty related data structures
        self.presence_penalties = torch.empty((max_num_reqs, ),
                                              dtype=torch.float,
                                              device=device)
        self.presence_penalties_cpu = self.presence_penalties.numpy()
        self.presence_penalties_reqs: set[str] = set()

        # Repetition penalty related data structures
        self.repetition_penalties = torch.empty((max_num_reqs, ),
                                                dtype=torch.float,
                                                device=device)
        self.repetition_penalties_cpu = \
            self.repetition_penalties.numpy()
        self.repetition_penalties_reqs: set[str] = set()

        # req_index -> (min_tokens, stop_token_ids)
        self.min_tokens: dict[int, tuple[int, set[int]]] = {}

        # req_index -> generator
        # NOTE(woosuk): The indices of the requests that do not have their own
        # generator should not be included in the dictionary.
        self.generators: dict[int, torch.Generator] = {}

        self.num_logprobs: dict[str, int] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        self.logit_bias: list[Optional[dict[int,
                                            float]]] = [None] * max_num_reqs
        self.has_allowed_token_ids: set[str] = set()
        self.allowed_token_ids_mask: Optional[torch.Tensor] = None

        # req_index -> bad_words_token_ids
        self.bad_words_token_ids: dict[int, list[list[int]]] = {}

        self.req_output_token_ids: list[Optional[list[int]]] = []

        # Model indices to mask padded request
        self.model_indices_mask = torch.zeros(self.max_num_reqs,
                                              dtype=torch.bool,
                                              device=device)

        # Initialize with max number of requests
        self.padded_batch_size = self.max_num_reqs

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

        # Keep tracking of number of requests
        self._num_requests = 0

    @property
    def req_ids(self) -> list[str]:
        # None elements should only be present transiently
        # while performing state updates to the batch.
        return cast(list[str], self._req_ids)

    def add_request(
        self,
        request: "CachedRequestState",
        req_index: Optional[int] = None,
    ) -> None:
        if req_index is None:
            req_index = self.num_reqs
        assert req_index < self.max_num_reqs

        self.model_indices_mask[req_index] = True
        req_id = request.req_id
        if req_index == len(self._req_ids):
            self._req_ids.append(req_id)
            self.req_output_token_ids.append(request.output_token_ids)
        else:
            self._req_ids[req_index] = req_id
            self.req_output_token_ids[req_index] = request.output_token_ids

        self.req_id_to_index[req_id] = req_index

        # Copy the prompt token ids and output token ids.
        num_prompt_tokens = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = num_prompt_tokens

        self.token_ids_cpu[
            req_index, :num_prompt_tokens] = request.prompt_token_ids
        start_idx = num_prompt_tokens
        end_idx = start_idx + len(request.output_token_ids)
        self.token_ids_cpu[req_index,
                           start_idx:end_idx] = request.output_token_ids

        sampling_params = request.sampling_params
        if sampling_params.sampling_type == SamplingType.GREEDY:
            # Avoid later division by zero.
            self.temperature_cpu[req_index] = -1.0
            self.greedy_reqs.add(req_id)
        else:
            self.temperature_cpu[req_index] = sampling_params.temperature
            self.random_reqs.add(req_id)

        self.top_p_cpu[req_index] = sampling_params.top_p
        if sampling_params.top_p < 1:
            self.top_p_reqs.add(req_id)
        self.top_k_cpu[req_index] = sampling_params.top_k
        if sampling_params.top_k > 0:
            self.top_k_reqs.add(req_id)
        self.min_p_cpu[req_index] = sampling_params.min_p
        self.frequency_penalties_cpu[
            req_index] = sampling_params.frequency_penalty
        if sampling_params.min_p > _SAMPLING_EPS:
            self.min_p_reqs.add(req_id)
        if sampling_params.frequency_penalty != 0.0:
            self.frequency_penalties_reqs.add(req_id)
        self.presence_penalties_cpu[
            req_index] = sampling_params.presence_penalty
        if sampling_params.presence_penalty != 0.0:
            self.presence_penalties_reqs.add(req_id)
        self.repetition_penalties_cpu[
            req_index] = sampling_params.repetition_penalty
        if sampling_params.repetition_penalty != 1.0:
            self.repetition_penalties_reqs.add(req_id)
        if sampling_params.min_tokens:
            self.min_tokens[req_index] = (sampling_params.min_tokens,
                                          sampling_params.all_stop_token_ids)

        # NOTE(woosuk): self.generators should not include the requests that
        # do not have their own generator.
        if request.generator is not None:
            self.generators[req_index] = request.generator

        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = sampling_params.logprobs
        if sampling_params.prompt_logprobs is not None:
            self.num_prompt_logprobs[req_id] = sampling_params.prompt_logprobs
        if sampling_params.logit_bias is not None:
            self.logit_bias[req_index] = sampling_params.logit_bias

        if sampling_params.allowed_token_ids:
            self.has_allowed_token_ids.add(req_id)
            if self.allowed_token_ids_mask is None:
                # Lazy allocation for this tensor, which can be large.
                self.allowed_token_ids_mask = torch.zeros(self.max_num_reqs,
                                                          self.vocab_size,
                                                          dtype=torch.bool,
                                                          device=self.device)
            self.allowed_token_ids_mask[req_index][
                sampling_params.allowed_token_ids] = True

        if sampling_params.bad_words_token_ids:
            self.bad_words_token_ids[
                req_index] = sampling_params.bad_words_token_ids
        self._num_requests += 1
        assert self._num_requests <= self.max_num_reqs

    def clear_requests(self):
        '''
        TODO: While we do not support continuous batch, the simplest solution
        is to clear the whole batch
        '''
        self.req_id_to_index = {}
        self.model_indices_mask.fill_(False)

        self._req_ids = []
        self.req_output_token_ids = []

        self.greedy_reqs = set()
        self.random_reqs = set()
        self.top_p_reqs = set()
        self.top_k_reqs = set()
        self.min_p_reqs = set()
        self.min_tokens = {}
        self.frequency_penalties_reqs = set()
        self.presence_penalties_reqs = set()
        self.repetition_penalties_reqs = set()
        self.generators = {}
        self.num_logprobs = {}
        self.num_prompt_logprobs = {}

        self.logit_bias = [None] * self.max_num_reqs
        self.has_allowed_token_ids = set()
        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask.fill_(False)

        self._num_requests = 0

    def soft_remove_request(self, req_id: str):
        '''
        Disable requests from the batch
        
        It does the following
        - mask out the padded requests.
        - Remove reference from the sets that track the type of param 
          e.g. greeedy_reqs 
        - Update some containers by reference to update the sampling parameters
        
        This method however, does not change the self.req_id_to_index which is
        needed to keep the map of finished padded requests.
        
        TODO: in the future we shall remove/change this method to actual clean 
        up all the data 
        '''
        if req_id not in self.req_id_to_index:
            return
        req_index = self.req_id_to_index[req_id]

        # Mask out the request
        self.model_indices_mask[req_index] = False

        # Remove the references

        # Index corrected based on the padded/deactivated requests
        shifted_index = self.model_indices_mask[:req_index].sum().item()
        self.req_output_token_ids.pop(shifted_index)

        self.greedy_reqs.discard(req_id)
        self.random_reqs.discard(req_id)
        self.top_p_reqs.discard(req_id)
        self.top_k_reqs.discard(req_id)
        self.min_p_reqs.discard(req_id)

        self.frequency_penalties_reqs.discard(req_id)
        self.presence_penalties_reqs.discard(req_id)
        self.repetition_penalties_reqs.discard(req_id)
        self.generators.pop(req_index, None)
        self.num_logprobs.pop(req_id, None)
        self.num_prompt_logprobs.pop(req_id, None)

        self.logit_bias[req_index] = None
        self.has_allowed_token_ids.discard(req_id)

        if self.allowed_token_ids_mask is not None:
            self.allowed_token_ids_mask[req_index].fill_(False)

        self.min_tokens.pop(req_index, None)

        self._num_requests -= 1

        self.bad_words_token_ids.pop(req_index, None)

    def refresh_sampling_metadata(self):
        self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:

        # Mask truncated by the num of requests
        indices_mask = self.model_indices_mask

        if not self.all_greedy:
            temperature = self.temperature[indices_mask]
        else:
            temperature = None

        if not self.no_penalties:

            # The prompt tokens are used only for applying penalties during
            # the sampling process. Hence copy these tensors only when
            # there are requests which need penalties to be applied.
            prompt_token_ids = self._make_prompt_token_ids_tensor()
        else:
            prompt_token_ids = None

        allowed_token_ids_mask: Optional[torch.Tensor] = None
        if not self.no_allowed_token_ids:
            assert self.allowed_token_ids_mask is not None
            allowed_token_ids_mask = self.allowed_token_ids_mask[indices_mask]

        indices = indices_mask.nonzero().squeeze(dim=-1).tolist()
        logit_bias = [self.logit_bias[i] for i in indices]

        generators = { i: self.generators[idx] \
            for i, idx in enumerate(indices) \
                if self.generators.get(idx) is not None}

        min_tokens = { i: self.min_tokens[idx] \
            for i, idx in enumerate(indices) \
                if self.min_tokens.get(idx) is not None}

        return SamplingMetadata(
            temperature=temperature,
            all_greedy=self.all_greedy,
            all_random=self.all_random,
            top_p=None if self.no_top_p else self.top_p[indices_mask],
            top_k=None if self.no_top_k else self.top_k[indices_mask],
            min_p=None if self.no_min_p else self.min_p[indices_mask],
            generators=generators,
            max_num_logprobs=self.max_num_logprobs,
            prompt_token_ids=prompt_token_ids,
            frequency_penalties=self.frequency_penalties[indices_mask],
            presence_penalties=self.presence_penalties[indices_mask],
            repetition_penalties=self.repetition_penalties[indices_mask],
            # WARN: dangerous side-effect. Here output_token_ids is a reference
            # and may be updated from other contexts. For instance,
            # spyre_model_runner updates this data at _update_states.
            output_token_ids=cast(list[list[int]], self.req_output_token_ids),
            min_tokens=min_tokens,
            no_penalties=self.no_penalties,
            logit_bias=logit_bias,
            allowed_token_ids_mask=allowed_token_ids_mask,
            bad_words_token_ids=self.bad_words_token_ids,
        )

    def _make_prompt_token_ids_tensor(self) -> torch.Tensor:

        model_indices_mask_cpu = self.model_indices_mask.numpy()
        num_prompt_tokens = self.num_prompt_tokens[model_indices_mask_cpu]
        max_prompt_len = num_prompt_tokens.max()
        prompt_token_ids_tensor = torch.empty(
            (self._num_requests, max_prompt_len),
            device=self.device,
            dtype=torch.int64,
        )
        prompt_token_ids = prompt_token_ids_tensor.numpy()
        prompt_token_ids[:] = self.token_ids_cpu[
            model_indices_mask_cpu, :max_prompt_len]
        # Use the value of vocab_size as a pad since we don't have a
        # token_id of this value.

        for i in range(self._num_requests):
            prompt_token_ids[i, num_prompt_tokens[i]:] = self.vocab_size
        return prompt_token_ids_tensor

    def get_unpadded_output_indices(self) -> dict[str, int]:
        """The inputs to the model are all padded to a constant batch size, and
        self.req_id_to_index is the map of request id -> padded index.
        However, finished requests and padded requests are stripped from the
        output, so the mapping of request id -> unpadded output index needs to
        be created to be returned in `ModelRunnerOutput`.

        For example if:
        - self.model_indices_mask = [F, T, T, F]
        - self.req_id_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
        This will output: {"B": 0, "C": 1}
        """

        indices = self.model_indices_mask.nonzero().squeeze(dim=-1).tolist()
        return {self._req_ids[idx]: i for i, idx in enumerate(indices)}

    def get_model_indices(self):
        return self.model_indices_mask[:self.padded_batch_size]

    def get_req_index(self, req_id):
        return self.req_id_to_index.get(req_id)

    @property
    def num_reqs(self) -> int:
        return self._num_requests

    @property
    def all_greedy(self) -> bool:
        return len(self.random_reqs) == 0

    @property
    def all_random(self) -> bool:
        return len(self.greedy_reqs) == 0

    @property
    def no_top_p(self) -> bool:
        return len(self.top_p_reqs) == 0

    @property
    def no_top_k(self) -> bool:
        return len(self.top_k_reqs) == 0

    @property
    def no_min_p(self) -> bool:
        return len(self.min_p_reqs) == 0

    @property
    def no_penalties(self) -> bool:
        return (len(self.presence_penalties_reqs) == 0
                and len(self.frequency_penalties_reqs) == 0
                and len(self.repetition_penalties_reqs) == 0)

    @property
    def max_num_logprobs(self) -> Optional[int]:
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    @property
    def no_prompt_logprob(self) -> bool:
        return not self.num_prompt_logprobs

    @property
    def no_allowed_token_ids(self) -> bool:
        return len(self.has_allowed_token_ids) == 0

    @property
    def requests_ids(self) -> list[str]:
        return list(self.req_id_to_index.keys())

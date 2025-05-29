# SPDX-License-Identifier: Apache-2.0
# Datastructures defining an input batch

# Based on vllm/vllm/v1/worker/gpu_input_batch.py

from dataclasses import dataclass
from typing import Optional, cast

import torch
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_spyre.v1.worker.spyre_base_input_batch import (BaseInputBatch,
                                                         BaseRequestState)

_SAMPLING_EPS = 1e-5


@dataclass
class SamplingRequestState(BaseRequestState):

    sampling_params: SamplingParams
    generator: Optional[torch.Generator]

    output_token_ids: list[int]

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)


class SamplingInputBatch(BaseInputBatch[SamplingRequestState]):
    '''
    This class was based on the InputBatch for GPU of vLLM V1.
    
    The implementation of vLLM was designed to track the request parameters
    and does some optimizations to keep the data organized tight. It also
    build the sampling parameters and do lazy allocations when possible.
    
    For the Spyre, we do something similar, however we do not worry (for now)
    the transfer data from CPU -> GPU as vLLM does. One key difference between 
    those implementations is that we have a mask for active request based on 
    the indices stored in `req_indices_mask`. Sometimes we need to check it
    to get the correct index of a request see `get_unpadded_output_indices`. 
    
    For static batching, the correct usage of this class consists in add 
    requests and clear the whole batch before process more requests. 
    
    For continuous batching, when a request is removed, it frees a slot where 
    a new request can be inserted. Then, the request index mask is used to 
    condense the sampling parameters.
    '''

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

        # This is updated each time the batch constituents change.
        self.sampling_metadata = self._make_sampling_metadata()

    def add_request(
        self,
        request: "SamplingRequestState",
        req_index: Optional[int] = None,
    ) -> None:

        req_index = super()._add_request(request, req_index)
        req_id = request.req_id

        # NOTE: differently from gpu input batch, self.req_output_token_ids
        # is not synced with self._req_ids, it should use
        # self.req_indices_mask to resolve its index considering masked
        # out requests.
        dense_index = self.req_idx_to_dense_index(req_index)
        self.req_output_token_ids.insert(dense_index, request.output_token_ids)

        # Copy the output token ids.
        start_idx = len(request.prompt_token_ids)
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

    def clear_requests(self):
        '''
        Clear the batch, mostly used by static batching
        '''
        super().clear_requests()
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

    def remove_request(self, req_id: str):
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

        req_index = super()._remove_request(req_id)
        if req_index is None:
            return

        # Index corrected based on the padded/deactivated requests
        dense_index = self.req_idx_to_dense_index(req_index)
        self.req_output_token_ids.pop(dense_index)

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

        self.bad_words_token_ids.pop(req_index, None)

    def refresh(self):
        self.sampling_metadata = self._make_sampling_metadata()

    def _make_sampling_metadata(self) -> SamplingMetadata:

        # Mask truncated by the num of requests
        indices_mask = self.req_indices_mask

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

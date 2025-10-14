import itertools
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import (BUILTIN_LOGITS_PROCESSORS,
                                             STR_POOLING_REJECTS_LOGITSPROCS,
                                             BatchUpdate,
                                             LogitBiasLogitsProcessor,
                                             LogitsProcessor,
                                             MinPLogitsProcessor,
                                             MinTokensLogitsProcessor,
                                             _load_custom_logitsprocs,
                                             process_dict_updates)
from vllm.v1.sample.logits_processor.state import LogitsProcessors

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config import VllmConfig
else:
    SamplingParams = None
    VllmConfig = None

assert len(BUILTIN_LOGITS_PROCESSORS) == 3


def build_logitsprocs_for_cb(
    vllm_config: VllmConfig,
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    batch_size: int,
    custom_logitsprocs: Sequence[Union[str, type[LogitsProcessor]]] = (),
) -> LogitsProcessors:
    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug("Skipping logits processor loading because pooling models"
                     " do not support logits processors.")
        return LogitsProcessors()
    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)

    return LogitsProcessors(
        LogitsProcessorWrapper(logit_processor,
                              vllm_config,
                              device,
                              is_pin_memory,
                              batch_size) \
            for logit_processor in itertools.chain(
                BUILTIN_LOGITS_PROCESSORS,
                custom_logitsprocs_classes
            )
        )


class SpyreLogitsProcessor:

    def set_prefill_index(self, idx: int) -> None:
        raise NotImplementedError


class LogitsProcessorWrapper(LogitsProcessor, SpyreLogitsProcessor):
    """Logit processor to inject expected token during generation for tests"""

    def __init__(self, logit_processor: LogitsProcessor,
                 vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool, batch_size: int):
        self.logitprocs: list[LogitsProcessor] = [
            logit_processor(vllm_config, device, is_pin_memory) \
                for _ in range(batch_size)
        ]

        self._is_argmax_invariant : bool = \
            self.logitprocs[0].is_argmax_invariant()

        self._prefill_index: Optional[int] = None

    def is_argmax_invariant(self) -> bool:
        """Never impacts greedy sampling"""
        return self._is_argmax_invariant

    def update_state(self, batch_update: Optional[BatchUpdate]):
        # This method keeps the indices consistent of request while the
        # persistent batch is changing.
        if not batch_update:
            return

        # Process added requests.
        for index, params, prompt_tok_ids, out_tok_ids in batch_update.added:
            self.logitprocs[index].update_state(
                BatchUpdate(
                    batch_size=1,
                    removed=[],
                    moved=[],
                    added=[(0, params, prompt_tok_ids, out_tok_ids)],
                ))

        for index in batch_update.removed:
            self.logitprocs[index].update_state(
                BatchUpdate(batch_size=1, removed=[0], moved=[], added=[]))

        for adx, bdx, _ in batch_update.moved:
            self.logitprocs[adx], self.logitprocs[bdx] = \
                self.logitprocs[bdx], self.logitprocs[adx]

    def apply(self, logits: torch.Tensor) -> torch.Tensor:

        if self._prefill_index is not None:
            logits = self.logitprocs[self._prefill_index].apply(logits)
            self._prefill_index = None
            return logits

        batch_size = logits.shape[0]
        for i in range(batch_size):
            logits[i] = self.logitprocs[i].apply(logits[i].unsqueeze(0))

        return logits

    def set_prefill_index(self, idx: int) -> None:
        self._prefill_index = idx


class SpyreMinPLogitsProcessor(MinPLogitsProcessor, SpyreLogitsProcessor):

    def __init__(self, vllm_config: "VllmConfig", device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        self.prefill_index: Optional[int] = None

    def apply(self, logits: torch.Tensor) -> torch.Tensor:

        if not self.min_p_count:
            return logits

        # if self.prefill_index is not None:
        #     pass

        # # Convert logits to probability distribution
        # probability_values = torch.nn.functional.softmax(logits, dim=-1)
        # # Calculate maximum probabilities per sequence
        # max_probabilities = torch.amax(probability_values,
        #                                dim=-1,
        #                                keepdim=True)
        # # Adjust min_p
        # adjusted_min_p = max_probabilities.mul_(self.min_p)
        # # Identify valid tokens using threshold comparison
        # invalid_token_mask = probability_values < adjusted_min_p
        # # Apply mask using boolean indexing
        # logits[invalid_token_mask] = -float('inf')
        # return logits


class SpyreLogitBiasLogitsProcessor(LogitBiasLogitsProcessor,
                                    SpyreLogitsProcessor):

    def __init__(self, _, device: torch.device, is_pin_memory: bool):
        self.device = device
        self.pin_memory = is_pin_memory
        self.biases: dict[int, dict[int, float]] = {}

        self.bias_tensor: torch.Tensor = torch.tensor(())
        self.logits_slice = (self._device_tensor([], torch.int32),
                             self._device_tensor([], torch.int32))

    def is_argmax_invariant(self) -> bool:
        """Logit bias can rebalance token probabilities and change the
        outcome of argmax in greedy sampling."""
        return False

    def update_state(self, batch_update: Optional[BatchUpdate]):
        needs_update = process_dict_updates(
            self.biases, batch_update,
            lambda params, _, __: params.logit_bias or None)

        # Update tensors if needed.
        if needs_update:
            reqs: list[int] = []
            tok_ids: list[int] = []
            biases: list[float] = []
            for req, lb in self.biases.items():
                reqs.extend([req] * len(lb))
                tok_ids.extend(lb.keys())
                biases.extend(lb.values())

            self.bias_tensor = self._device_tensor(biases, torch.float32)
            self.logits_slice = (self._device_tensor(reqs, torch.int32),
                                 self._device_tensor(tok_ids, torch.int32))

    def _device_tensor(self, data: list, dtype: torch.dtype) -> torch.Tensor:
        return (torch.tensor(data,
                             device="cpu",
                             dtype=dtype,
                             pin_memory=self.pin_memory).to(device=self.device,
                                                            non_blocking=True))

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self.biases:
            logits[self.logits_slice] += self.bias_tensor
        return logits


class SpyreMinTokensLogitsProcessor(MinTokensLogitsProcessor,
                                    SpyreLogitsProcessor):

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory)
        self._prefill_slice : Optional[tuple[torch.Tensor, torch.Tensor]] \
            = None
        self._is_prefill: bool = False

    def set_prefill_index(self, idx: int) -> None:

        reqs: list[int] = []
        tok_ids: list[int] = []
        for req, (_, _, stop_tok_ids) in self.min_toks.items():
            if req == idx:
                # NOTE: always request 0 for prefill
                # logits will only have logits for a single request
                reqs.extend([0] * len(stop_tok_ids))
                tok_ids.extend(stop_tok_ids)

        if reqs and tok_ids:
            self._prefill_slice = (self._device_tensor(reqs, torch.int32),
                                   self._device_tensor(tok_ids, torch.int32))
        self._is_prefill = True

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._prefill_slice is not None:
            logits[self._prefill_slice] = -float("inf")
            self._prefill_slice = None
            self._is_prefill = False
            return logits
        elif self._is_prefill:
            # It is prefill but we do not need to do anything
            # for the prefill request, just return logits to
            # avoid slice the logits with batch_size = 1
            self._is_prefill = False
            return logits

        return super().apply(logits)

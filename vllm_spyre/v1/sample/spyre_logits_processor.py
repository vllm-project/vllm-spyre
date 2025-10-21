import itertools
from typing import Optional, Sequence, Union

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.sample.logits_processor import (BUILTIN_LOGITS_PROCESSORS,
                                             STR_POOLING_REJECTS_LOGITSPROCS,
                                             BatchUpdate, LogitsProcessor,
                                             _load_custom_logitsprocs)
from vllm.v1.sample.logits_processor.state import LogitsProcessors

logger = init_logger(__name__)


def build_logitsprocs_for_cb(
    vllm_config: "VllmConfig",
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
        LogitProcessorWrapper(logit_processor,
                              vllm_config,
                              device,
                              is_pin_memory,
                              batch_size) \
            for logit_processor in itertools.chain(
                BUILTIN_LOGITS_PROCESSORS,
                custom_logitsprocs_classes
            )
        )


class LogitProcessorWrapper(LogitsProcessor):
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

import itertools
from typing import TYPE_CHECKING, Optional, Sequence, Union

import torch
from vllm.logger import init_logger
# yapf: disable
from vllm.v1.sample.logits_processor import (BUILTIN_LOGITS_PROCESSORS,
                                             STR_POOLING_REJECTS_LOGITSPROCS,
                                             BatchUpdate,
                                             LogitBiasLogitsProcessor,
                                             LogitsProcessor,
                                             MinPLogitsProcessor,
                                             MinTokensLogitsProcessor,
                                             _load_custom_logitsprocs,
                                             process_dict_updates)
# yapf: enable
from vllm.v1.sample.logits_processor.state import LogitsProcessors

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm import SamplingParams
    from vllm.config import VllmConfig
else:
    SamplingParams = None
    VllmConfig = None

SPYRE_BUILTIN_LOGITS_PROCESSORS = [
    MinPLogitsProcessor, MinTokensLogitsProcessor, LogitBiasLogitsProcessor
]


def build_logitsprocs_for_cb(
    vllm_config: VllmConfig,
    device: torch.device,
    is_pin_memory: bool,
    is_pooling_model: bool,
    batch_size: int,
    custom_logitsprocs: Sequence[Union[str, type[LogitsProcessor]]] = (),
) -> LogitsProcessors:

    if len(BUILTIN_LOGITS_PROCESSORS) > 3:
        logger.warning(
            "There are %d logits processors, which is unexpected "
            "for this vllm-spyre version. Consider upgrade "
            "vllm-spyre or open an issue to investigate this",
            len(BUILTIN_LOGITS_PROCESSORS))

    if is_pooling_model:
        if custom_logitsprocs:
            raise ValueError(STR_POOLING_REJECTS_LOGITSPROCS)
        logger.debug("Skipping logits processor loading because pooling models"
                     " do not support logits processors.")
        return LogitsProcessors()
    custom_logitsprocs_classes = _load_custom_logitsprocs(custom_logitsprocs)

    # Collect builtin LPs to fallback to the wrapper
    builtin_logitsprocs = [lp for lp in BUILTIN_LOGITS_PROCESSORS \
                           if lp not in SPYRE_BUILTIN_LOGITS_PROCESSORS]

    logitprocs_classes = custom_logitsprocs_classes + builtin_logitsprocs

    # To avoid circular import
    from vllm_spyre.v1.sample.golden_token_injector import GoldenTokenInjector

    return LogitsProcessors( itertools.chain(
        [SpyreLogitBiasLogitsProcessor(vllm_config,
                              device,
                              is_pin_memory),
         SpyreMinPLogitsProcessor(vllm_config,
                              device,
                              is_pin_memory),
        SpyreMinTokensLogitsProcessor(vllm_config,
                              device,
                              is_pin_memory),
        GoldenTokenInjector(vllm_config, device, is_pin_memory)

        ],
        [LogitsProcessorWrapper(logit_processor,
                              vllm_config,
                              device,
                              is_pin_memory,
                              batch_size) \
        for logit_processor in logitprocs_classes]
    ))


class SpyreLogitsProcessor:

    def set_prefill(self, idx: int) -> None:
        raise NotImplementedError


class PrefillHelperLogitsProcessor(LogitsProcessor, SpyreLogitsProcessor):
    """ 
    Logits processor (LP) that separates two instances of a concrete LPS:
    one for the prefill, and other for the batch. This class only works if 
    the state of the LP is independent between prefill and decoding. for 
    example this class is not suitable for the golden token injector LP.
    """

    def __init__(self, config: VllmConfig, device: torch.device,
                 is_pin_memory: bool, logit_processor: LogitsProcessor):
        self._prefill_lp : LogitsProcessor = \
            logit_processor(config, device, is_pin_memory)
        self._batch_lp : LogitsProcessor = \
            logit_processor(config, device, is_pin_memory)

        self._is_prefill: bool = False

        # This dictionary stores the sampling parameters of `update_state` so
        # we can get when we call `set_prefill` to proper setup the prefill_lp.
        self._params: dict[int, tuple[SamplingParams, list[int],
                                      list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        """Never impacts greedy sampling"""
        return self._batch_lp.is_argmax_invariant()

    @staticmethod
    def update_batch_params(
        params: SamplingParams, prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int]
    ) -> tuple[SamplingParams, Sequence[int] | None, Sequence[int]] | None:
        return params, prompt_tok_ids, output_tok_ids

    def update_state(self, batch_update: BatchUpdate | None):

        process_dict_updates(self._params, batch_update,
                             self.update_batch_params)

        # Always pass to the batch LP
        self._batch_lp.update_state(batch_update)

    def set_prefill(self, idx: int) -> None:

        params, prompt_tok_ids, out_tok_ids = self._params[idx]
        self._prefill_lp.update_state(
            BatchUpdate(
                batch_size=1,
                removed=[],
                moved=[],
                added=[(0, params, prompt_tok_ids, out_tok_ids)],
            ))
        # self._params.pop(idx)
        self._is_prefill = True

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._is_prefill:
            logits = self._prefill_lp.apply(logits)

            # Clean the prefill LP
            self._is_prefill = False
            self._prefill_lp.update_state(
                BatchUpdate(batch_size=1, removed=[0], moved=[], added=[]))
            return logits

        return self._batch_lp.apply(logits)


class LogitsProcessorWrapper(LogitsProcessor, SpyreLogitsProcessor):
    """Logit processor to isolate logits processors to run individually"""

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

    def set_prefill(self, idx: int) -> None:
        self._prefill_index = idx


class SpyreMinPLogitsProcessor(PrefillHelperLogitsProcessor):

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory,
                         MinPLogitsProcessor)


class SpyreLogitBiasLogitsProcessor(PrefillHelperLogitsProcessor):

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory,
                         LogitBiasLogitsProcessor)


class SpyreMinTokensLogitsProcessor(PrefillHelperLogitsProcessor):

    def __init__(self, vllm_config: VllmConfig, device: torch.device,
                 is_pin_memory: bool):
        super().__init__(vllm_config, device, is_pin_memory,
                         MinTokensLogitsProcessor)

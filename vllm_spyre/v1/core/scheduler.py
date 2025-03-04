# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Deque, Optional

from vllm.config import (CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig,
                         SpeculativeConfig)
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.core.scheduler_output import SchedulerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class SpyreScheduler(Scheduler):
    """Small extension of the V1 scheduler that adds constraints for Sypre:
    - No continuous batching
    - Only schedules batches of requests that fit a common warmup shape
    """

    def __init__(
        self,
        scheduler_config: SchedulerConfig,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig],
        speculative_config: Optional[SpeculativeConfig],
        log_stats: bool,
    ) -> None:
        # Initialize vLLM scheduler
        super().__init__(scheduler_config=scheduler_config,
                         model_config=model_config,
                         cache_config=cache_config,
                         lora_config=lora_config,
                         speculative_config=speculative_config,
                         log_stats=log_stats)

        # Add our own state for handling Spyre constraints

        # All warmup shapes that we can support
        self.spyre_warmup_shapes: tuple[dict[str, int], ...] = \
            current_platform.get_warmup_shapes()

        # We'll put all new requests into this queue so that the base scheduler
        # does not attempt to schedule them until we release them into the
        # waiting queue. This lets us ensure that the set of requests the base
        # scheduler sees have at least one common warmup shape.
        self.holdback_queue: Deque[Request] = deque()

    def schedule(self) -> "SchedulerOutput":
        """This override adds constraints and then delegates most of the work
        to the base scheduler"""

        # First purge the full waiting queue into our holdback queue, preserving
        # priority
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # If no requests are currently running, we can now release requests back
        # into the waiting queue in priority order for the scheduler to prefill.
        # These must share a common warmup shape
        if len(self.running) == 0:

            # Make a copy of the warmup shapes
            available_warmup_shapes = list(self.spyre_warmup_shapes)

            while self.holdback_queue:
                request = self.holdback_queue[0]

                # prune the possible shapes to only those that fit this request
                # and the growing batch size
                max_tokens = 0
                if request.sampling_params is not None and\
                        request.sampling_params.max_tokens is not None:
                    max_tokens = request.sampling_params.max_tokens

                available_warmup_shapes = [
                    shape for shape in available_warmup_shapes
                    if request.num_prompt_tokens <= shape['prompt_length']
                    and max_tokens <= shape['new_tokens']
                    and len(self.running) < shape['batch_size']
                ]

                if len(available_warmup_shapes) > 0:
                    # There is still at least one valid shape, so add to the
                    # waiting queue
                    self.waiting.append(self.holdback_queue.popleft())
                else:
                    # We can't schedule this one.
                    # If it's the first request, then it fits _no_ shapes at all
                    # So we reject it entirely
                    if len(self.running) == 0:
                        logger.warning(
                            "No applicable warmup shape exists for "
                            "combination of prompt length (%d tokens) "
                            "and maximum number of output tokens to be "
                            "generated (%d tokens)", request.num_prompt_tokens,
                            request.sampling_params.max_tokens)

                        request.status = RequestStatus.FINISHED_IGNORED
                        self._free_request(self.holdback_queue.popleft())
                    else:
                        # Otherwise, we simply stop here so that the scheduler
                        # can work with the batch we have
                        break

        outputs = super().schedule()
        return outputs

    def get_num_unfinished_requests(self) -> int:
        # Override this to include our extra queue
        return len(self.waiting) + len(self.running) + len(self.holdback_queue)

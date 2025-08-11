# SPDX-License-Identifier: Apache-2.0

import math
import os
from collections import deque
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_model_runner import CBSpyreModelRunnerOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    SchedulerOutput = None

logger = init_logger(__name__)

NO_WARMUP_FIT_STOP_REASON = "Request did not fit any warmup shape"


class SpyreScheduler(Scheduler):
    """Base class inheriting from the V1 scheduler to support static 
    and continuous batching respecting AIU Spyre constraints."""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)

        # Requests are temporarily moved to this queue so that the base
        # scheduler does not see them. This lets us ensure that the set of
        # requests scheduled have at least one common warmup shape.
        self.holdback_queue: deque[Request] = deque()


class StaticBatchingSpyreScheduler(SpyreScheduler):
    """ Support of static batching """

    def __init__(self, *args, **kwargs) -> None:
        # Initialize SpyreScheduler
        super().__init__(*args, **kwargs)

        # Add our own state for handling Spyre constraints:
        # all warmup shapes that we can support
        self.spyre_warmup_shapes: tuple[dict[str, int], ...] = \
            SpyrePlatform.get_warmup_shapes(self.scheduler_config)

    def schedule(self) -> SchedulerOutput:
        """This override adds constraints and then delegates most of the work
        to the base scheduler"""
        # First purge the full waiting queue into our holdback queue, preserving
        # priority
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # store requests which don't fit the warmup shapes of the current batch
        skip_queue: deque[Request] = deque()

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
                available_warmup_shapes = self._get_matching_warmup_shapes(
                    request=request,
                    warmup_shapes=available_warmup_shapes,
                    current_batch_size=len(self.waiting))

                if len(available_warmup_shapes) > 0:
                    # There is still at least one valid shape, so add to the
                    # waiting queue
                    self.waiting.append(self.holdback_queue.popleft())
                    # remember the available warmup shapes of the current batch
                    last_available_warmup_shapes = available_warmup_shapes
                else:
                    # calculating the max possible batch size among the
                    # available warmup shapes of the scheduled requests
                    max_batch = max([
                        d['batch_size'] for d in last_available_warmup_shapes
                    ])

                    # if there is potential space in the batch but the current
                    # request does not fit, skip it and try with the next
                    if len(self.waiting) < max_batch:
                        available_warmup_shapes = last_available_warmup_shapes
                        skip_queue.append(self.holdback_queue.popleft())
                    else:
                        # If the batch is full, we exit the loop here
                        break

            logger.debug(
                "Scheduling a new batch of %d requests, holding back %d "
                "requests", len(self.waiting), len(self.holdback_queue))
        else:
            logger.debug("Scheduling a running batch of %d requests",
                         len(self.running))

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule()

        # first move skipped and then unscheduled requests back
        # to the waiting queue, preserving priority
        while skip_queue:
            self.waiting.append(skip_queue.popleft())

        while self.holdback_queue:
            self.waiting.append(self.holdback_queue.popleft())

        return outputs

    def _get_matching_warmup_shapes(
            self, request: Request, warmup_shapes: list[dict[str, int]],
            current_batch_size: int) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request"""
        max_tokens = 0
        if request.sampling_params is not None and\
                request.sampling_params.max_tokens is not None:
            max_tokens = request.sampling_params.max_tokens

        return [
            shape for shape in warmup_shapes
            if request.num_prompt_tokens <= shape['prompt_length']
            and max_tokens <= shape['new_tokens']
            and current_batch_size < shape['batch_size']
        ]


class ContinuousBatchingSpyreScheduler(SpyreScheduler):
    """ Support of continuous batching """

    # inherited from V1 base scheduler but mypy needs to know the type
    running: list[Request]

    def __init__(self, *args, **kwargs) -> None:
        # Initialize SpyreScheduler
        super().__init__(*args, **kwargs)
        self.tkv = 0
        self.n_free_blocks = 0
        self.block_size = SpyrePlatform.get_block_size()
        self.max_batch_tkv_limit = os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT",
                                             default='-1')
        assert self.max_batch_tkv_limit != '-1', "Expecting the env var"
        "VLLM_DT_MAX_BATCH_TKV_LIMIT to be set in platform.py"

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # Need an instance of CBSpyreModelRunnerOutput which holds the tkv value
        assert isinstance(
            model_runner_output, CBSpyreModelRunnerOutput
        ), "Expecting an instance of CBSpyreModelRunnerOutput"
        "when doing continuous batching."
        self.tkv = model_runner_output.tkv
        self.n_free_blocks = model_runner_output.n_free_blocks
        return super(SpyreScheduler,
                     self).update_from_output(scheduler_output,
                                              model_runner_output)

    def schedule(self) -> "SchedulerOutput":
        """This override adds constraints and then delegates most of the work
        to the base scheduler

        To avoid additional specialization, some requests are held back from the
        base scheduler but are restored after.
        """
        # First purge the full waiting queue into our holdback queue, preserving
        # priority
        while self.waiting:
            self.holdback_queue.append(self.waiting.popleft())

        # Check if new requests can be scheduled.
        while self.holdback_queue:
            if self.can_schedule(self.holdback_queue[0]):
                # Add request to the waiting queue
                self.waiting.append(self.holdback_queue.popleft())
            else:
                # Otherwise, we simply stop here so that the scheduler
                # can work with the batch we have
                break

        # Schedule Prefill and Decode separately
        if len(self.waiting) > 0:
            # For prefill, hide current decodes from the scheduler
            running_holdback = self.running
            self.running = []
            logger.debug(
                "Scheduling a prefill step of %d requests, holding back %d "
                "requests", len(self.waiting), len(self.holdback_queue))
        else:
            running_holdback = []
            logger.debug("Scheduling a decode step of %d requests",
                         len(self.running))

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule()

        # restore holdbacks after running the base scheduler
        self.running = self.running + running_holdback
        while self.holdback_queue:
            self.waiting.append(self.holdback_queue.popleft())

        return outputs

    def can_schedule(self, request) -> bool:
        max_prompt_batch_size = 1
        max_context_len = self.scheduler_config.max_model_len

        # running and waiting queues are both empty -> start new batch
        start_new_batch = len(self.running) + len(self.waiting) == 0
        # check that there is space in the current decode batch
        cond1 = len(self.running) + len(
            self.waiting) < self.max_num_running_reqs
        # check that there is space in the prefill batch
        cond2 = len(self.waiting) < max_prompt_batch_size
        # check that the prompt length does not exceed the current tkv
        cond3 = request.num_prompt_tokens <= self.tkv
        # check that the number of requested tokens can be served
        cond4 = request.max_tokens <= (max_context_len - self.tkv)
        # check that there are enough free blocks/pages remaining
        # Note: we only have to do check in case of a running batches
        # (not start_new_batch), because the minimal number of blocks covers
        # the context length for a single sequence, so tkv < block size is ok
        num_blocks_required = math.ceil(
            (self.tkv + request.max_tokens - 1) / self.block_size)
        # optimization: subtract the padding blocks from the reserved blocks
        num_fully_padded_blocks = math.floor(
            (self.tkv - request.num_prompt_tokens) / self.block_size)
        num_blocks_required -= num_fully_padded_blocks
        cond5 = num_blocks_required <= self.n_free_blocks
        # check that batch size x tkv is smaller than the max supported number
        cond6 = self.check_batch_tkv_limit(request)

        return start_new_batch or (cond1 and cond2 and cond3 and cond4
                                   and cond5 and cond6)

    def check_batch_tkv_limit(self, request) -> bool:
        """
        Check whether adding a new sequence to the decode batch would violate
        Spyre's maximum batch volume constraint.

        In Spyre, the product of `batch_size` and the current `tkv` 
        (tokens-per-sequence) must not exceed the limit defined by 
        `VLLM_DT_MAX_BATCH_TKV_LIMIT`. Before scheduling a new sequence, 
        we must ensure that this constraint will hold for all decoding 
        steps that result from combining the new sequence with the currently 
        running decode batch.

        This implementation:
        1. Computes the maximum possible `tkv` for each sequence in the 
        decode batch.
        2. Sorts these values in ascending order.
        3. Iterates through them, stopping once the `tkv` of the new sequence.
        is reached. Remaining sequences do not need to be checked explicitly, 
        since they were validated when they were added (by inductive reasoning).

        Note: drawing explaining the algorithm in more detail uploaded here: 
        https://github.com/vllm-project/vllm-spyre/pull/363#issuecomment-3173605517
        
        WIP: The result of this check could be cached and reused if both the 
        decode batch and the new input request are unchanged between calls.
        """

        # Compute the effective token length of the new request
        new_req_tkv = self.tkv + request.max_tokens - 1

        # Compute token lengths for all running requests (decode batch)
        decode_req_tkvs = [
            self.tkv + req.max_tokens - 1 - req.num_computed_tokens
            for req in self.running
        ]
        # Sort decode requests token lengths in ascending order
        decode_req_tkvs.sort()

        # Initialize values
        batch_size = len(self.running) + 1
        max_batch_tkv = 0

        # Try adding the new request to the batch and check the max volume
        for decode_req_tkv in decode_req_tkvs:
            if new_req_tkv <= decode_req_tkv:
                # If the new request is shorter, it limits the batch volume
                max_batch_tkv = max(max_batch_tkv, batch_size * new_req_tkv)
                break
            else:
                # Otherwise, use the current (longer) request's volume
                max_batch_tkv = max(max_batch_tkv, batch_size * decode_req_tkv)
                # decrease batch_size by 1 as the current request finished
                batch_size -= 1

        return max_batch_tkv <= int(self.max_batch_tkv_limit)

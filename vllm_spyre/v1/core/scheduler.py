# SPDX-License-Identifier: Apache-2.0

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
        return start_new_batch or (cond1 and cond2 and cond3 and cond4)

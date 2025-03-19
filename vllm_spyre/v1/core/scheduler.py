# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Deque

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.sampling_params import SamplingParams
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.core.scheduler_output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class SpyreScheduler(Scheduler):
    """Small extension of the V1 scheduler that adds constraints for Sypre:
    - No continuous batching
    - Only schedules batches of requests that fit a common warmup shape
    """

    def __init__(self, *args, **kwargs) -> None:
        # Initialize vLLM scheduler
        super().__init__(*args, **kwargs)

        # Add our own state for handling Spyre constraints

        # All warmup shapes that we can support
        self.spyre_warmup_shapes: tuple[dict[str, int], ...] = \
            current_platform.get_warmup_shapes()

        # We'll put all new requests into this queue so that the base scheduler
        # does not attempt to schedule them until we release them into the
        # waiting queue. This lets us ensure that the set of requests the base
        # scheduler sees have at least one common warmup shape.
        self.holdback_queue: Deque[Request] = deque()

        self.rejected_requests_this_iteration: set[str] = set()

    def add_request(self, request: Request) -> None:
        """This override rejects requests that fit no warmup shape"""
        if len(
                self._get_matching_warmup_shapes(
                    request, list(self.spyre_warmup_shapes))) == 0:
            logger.warning(
                "No applicable warmup shape exists for "
                "combination of prompt length (%d tokens) "
                "and maximum number of output tokens to be "
                "generated (%d tokens)", request.num_prompt_tokens,
                request.sampling_params.max_tokens)
            # TODO: There are open PRs that should enable raising an error for
            # a single request like this, which will gracefully return an error
            # for the request, instead of shutting down the engine.
            # See https://github.com/vllm-project/vllm/pull/11737
            # raise ValueError("Request does not fit any spyre warmup shape")

            # For now, we'll insert a dummy request and manually reject it when
            # we construct the outputs later
            self.rejected_requests_this_iteration.add(request.request_id)
            request.prompt_token_ids = [0]
            request.num_prompt_tokens = 1
            request.sampling_params = SamplingParams(max_tokens=1)

        # delegate to super
        super().add_request(request=request)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> EngineCoreOutputs:
        """Temporary override to handle rejected requests that were too large
        to schedule."""
        reject_outputs = self._handle_rejects()
        outputs = super().update_from_output(scheduler_output,
                                             model_runner_output)
        outputs.outputs.extend(reject_outputs)
        return outputs

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
                available_warmup_shapes = self._get_matching_warmup_shapes(
                    request=request, warmup_shapes=available_warmup_shapes)

                if len(available_warmup_shapes) > 0:
                    # There is still at least one valid shape, so add to the
                    # waiting queue
                    self.waiting.append(self.holdback_queue.popleft())
                else:
                    # Otherwise, we simply stop here so that the scheduler
                    # can work with the batch we have
                    break

            logger.debug(
                "Scheduling a new batch of %d requests, holding back %d "
                "requests", len(self.waiting), len(self.holdback_queue))
        else:
            logger.debug("Scheduling a running batch of %d requests",
                         len(self.running))

        outputs = super().schedule()
        return outputs

    def get_num_unfinished_requests(self) -> int:
        # Override this to include our extra queue
        return len(self.waiting) + len(self.running) + len(self.holdback_queue)

    def _get_matching_warmup_shapes(
            self, request: Request,
            warmup_shapes: list[dict[str, int]]) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request"""
        max_tokens = 0
        if request.sampling_params is not None and\
                request.sampling_params.max_tokens is not None:
            max_tokens = request.sampling_params.max_tokens

        return [
            shape for shape in warmup_shapes
            if request.num_prompt_tokens <= shape['prompt_length']
            and max_tokens <= shape['new_tokens']
            and len(self.waiting) < shape['batch_size']
        ]

    def _handle_rejects(self) -> list[EngineCoreOutput]:
        """Temporary solution to reject requests that were too large to
        schedule. This removes the rejected requests from the scheduler, and 
        returns empty outputs for them with finish reason `abort`.
        """
        if len(self.rejected_requests_this_iteration) == 0:
            return []

        reject_outputs: list[EngineCoreOutput] = []
        rejected_requests: list[Request] = [
            request for request in self.running
            if request.request_id in self.rejected_requests_this_iteration
        ]

        for request in rejected_requests:
            self.running.remove(request)
            reject_outputs.append(
                EngineCoreOutput(request.request_id,
                                 new_token_ids=[],
                                 finish_reason=FinishReason.ABORT,
                                 stop_reason="Request did not fit any warmup "
                                 "shape"))
            request.status = RequestStatus.FINISHED_ABORTED
            self._free_request(request)
        self.rejected_requests_this_iteration.clear()
        return reject_outputs

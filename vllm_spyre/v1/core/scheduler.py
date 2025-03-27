# SPDX-License-Identifier: Apache-2.0

from collections import deque
from typing import Deque

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.core.scheduler import Scheduler
from vllm.v1.core.scheduler_output import SchedulerOutput
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from vllm_spyre.platform import SpyrePlatform

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
            SpyrePlatform.get_warmup_shapes(self.scheduler_config)

        # We'll put all new requests into this queue so that the base scheduler
        # does not attempt to schedule them until we release them into the
        # waiting queue. This lets us ensure that the set of requests the base
        # scheduler sees have at least one common warmup shape.
        self.holdback_queue: Deque[Request] = deque()

        self.rejected_requests: set[str] = set()

    def add_request(self, request: Request) -> None:
        """This override rejects requests that fit no warmup shape"""
        if len(
                self._get_matching_warmup_shapes(request=request,
                                                 warmup_shapes=list(
                                                     self.spyre_warmup_shapes),
                                                 current_batch_size=0)) == 0:
            logger.warning(
                "No applicable warmup shape exists for "
                "combination of prompt length (%d tokens) "
                "and maximum number of output tokens to be "
                "generated (%d tokens) from request id %s",
                request.num_prompt_tokens, request.sampling_params.max_tokens,
                request.request_id)
            # TODO: There are open PRs that should enable raising an error for
            # a single request like this, which will gracefully return an error
            # for the request, instead of shutting down the engine.
            # See https://github.com/vllm-project/vllm/pull/11737
            # raise ValueError("Request does not fit any spyre warmup shape")

            # For now, we'll insert a dummy request and manually reject it when
            # we construct the outputs later
            self.rejected_requests.add(request.request_id)
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
                    request=request,
                    warmup_shapes=available_warmup_shapes,
                    current_batch_size=len(self.waiting))

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

    def _handle_rejects(self) -> list[EngineCoreOutput]:
        """Temporary solution to reject requests that were too large to
        schedule. This removes the rejected requests from the scheduler, and 
        returns empty outputs for them with finish reason `abort`.
        """
        if len(self.rejected_requests) == 0:
            return []

        # Remove rejected requests from all queues
        reject_outputs = self._reject_from_queue(self.running)
        reject_outputs.extend(self._reject_from_queue(self.waiting))
        reject_outputs.extend(self._reject_from_queue(self.holdback_queue))
        self.rejected_requests.clear()

        return reject_outputs

    def _reject_from_queue(self,
                           queue: Deque[Request]) -> list[EngineCoreOutput]:
        """Remove rejected requests from a given queue and return a list of 
        engine core outputs to return for them"""
        reject_outputs: list[EngineCoreOutput] = []
        rejected_requests: list[Request] = [
            request for request in queue
            if request.request_id in self.rejected_requests
        ]

        for request in rejected_requests:
            queue.remove(request)
            reject_outputs.append(
                EngineCoreOutput(request.request_id,
                                 new_token_ids=[],
                                 finish_reason=FinishReason.ABORT,
                                 stop_reason="Request did not fit any warmup "
                                 "shape"))
            request.status = RequestStatus.FINISHED_ABORTED
            self._free_request(request)
            self.rejected_requests.remove(request.request_id)

        return reject_outputs

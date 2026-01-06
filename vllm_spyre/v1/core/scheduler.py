# SPDX-License-Identifier: Apache-2.0

import math
import os
from collections import deque
from typing import TYPE_CHECKING, Iterable, Union

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutputs
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

import vllm_spyre.envs as envs_spyre
from vllm_spyre.platform import SpyrePlatform
from vllm_spyre.v1.worker.spyre_model_runner import (
    CBSpyreModelRunnerOutput,
    CPSpyreModelRunnerOutput,
)

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
        self.model_config = self.vllm_config.model_config


class StaticBatchingSpyreScheduler(SpyreScheduler):
    """Support of static batching"""

    def __init__(self, *args, **kwargs) -> None:
        # Initialize SpyreScheduler
        super().__init__(*args, **kwargs)

        # Add our own state for handling Spyre constraints:
        # all warmup shapes that we can support
        self.spyre_warmup_shapes: tuple[dict[str, int], ...] = SpyrePlatform.get_warmup_shapes(
            self.scheduler_config
        )

    def schedule(self) -> SchedulerOutput:
        """This override adds constraints and then delegates most of the work
        to the base scheduler"""
        # First purge the full waiting queue into our holdback queue, preserving
        # priority, so that the base scheduler does not see them.
        # This lets us ensure that the set of requests scheduled have at least
        # one common warmup shape.
        holdback_queue: deque[Request] = deque()
        while self.waiting:
            holdback_queue.append(self.waiting.popleft())

        # store requests which don't fit the warmup shapes of the current batch
        skip_queue: deque[Request] = deque()

        # If no requests are currently running, we can now release requests back
        # into the waiting queue in priority order for the scheduler to prefill.
        # These must share a common warmup shape
        if len(self.running) == 0:
            # Make a copy of the warmup shapes
            available_warmup_shapes = list(self.spyre_warmup_shapes)

            while holdback_queue:
                request = holdback_queue[0]

                # prune the possible shapes to only those that fit this request
                # and the growing batch size
                available_warmup_shapes = self._get_matching_warmup_shapes(
                    request=request,
                    warmup_shapes=available_warmup_shapes,
                    current_batch_size=len(self.waiting),
                )

                if len(available_warmup_shapes) > 0:
                    # There is still at least one valid shape, so add to the
                    # waiting queue
                    self.waiting.append(holdback_queue.popleft())
                    # remember the available warmup shapes of the current batch
                    last_available_warmup_shapes = available_warmup_shapes
                else:
                    # calculating the max possible batch size among the
                    # available warmup shapes of the scheduled requests
                    max_batch = max([d["batch_size"] for d in last_available_warmup_shapes])

                    # if there is potential space in the batch but the current
                    # request does not fit, skip it and try with the next
                    if len(self.waiting) < max_batch:
                        available_warmup_shapes = last_available_warmup_shapes
                        skip_queue.append(holdback_queue.popleft())
                    else:
                        # If the batch is full, we exit the loop here
                        break

            logger.debug(
                "Scheduling a new batch of %d requests, holding back %d requests",
                len(self.waiting),
                len(holdback_queue),
            )
        else:
            logger.debug("Scheduling a running batch of %d requests", len(self.running))

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule()

        # first move skipped and then unscheduled requests back
        # to the waiting queue, preserving priority
        while skip_queue:
            self.waiting.append(skip_queue.popleft())

        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

        return outputs

    def _get_matching_warmup_shapes(
        self, request: Request, warmup_shapes: list[dict[str, int]], current_batch_size: int
    ) -> list[dict[str, int]]:
        """Return the subset of shapes that match this request"""
        max_tokens = 0
        if request.sampling_params is not None and request.sampling_params.max_tokens is not None:
            max_tokens = request.sampling_params.max_tokens

        return [
            shape
            for shape in warmup_shapes
            if request.num_prompt_tokens <= shape["prompt_length"]
            and max_tokens <= shape["new_tokens"]
            and current_batch_size < shape["batch_size"]
        ]


class ContinuousBatchingSpyreScheduler(SpyreScheduler):
    """Support of continuous batching"""

    # inherited from V1 base scheduler but mypy needs to know the type
    running: list[Request]

    def __init__(self, *args, **kwargs) -> None:
        # Initialize SpyreScheduler
        super().__init__(*args, **kwargs)
        self.tkv = 0
        self.n_free_blocks = 0
        self.block_size = SpyrePlatform.get_block_size()
        self.max_batch_tkv_limit = os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT", default="-1")
        assert self.max_batch_tkv_limit != "-1", (
            "Expecting the env var VLLM_DT_MAX_BATCH_TKV_LIMIT to be set in platform.py"
        )
        # cache for self.check_batch_tkv_limit() outer key: tuple(request_ids),
        # inner key: (request_id, max_batch_tkv_limit), value: (lower, upper)
        self._cache_check_batch_tkv_limit: dict[tuple, dict[tuple, tuple]] = {}

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        # Need an instance of CBSpyreModelRunnerOutput which holds the tkv value
        assert isinstance(model_runner_output, CBSpyreModelRunnerOutput), (
            "Expecting an instance of CBSpyreModelRunnerOutput when doing continuous batching."
        )
        self.tkv = model_runner_output.tkv
        self.n_free_blocks = model_runner_output.n_free_blocks
        return super(SpyreScheduler, self).update_from_output(scheduler_output, model_runner_output)

    def schedule(self) -> "SchedulerOutput":
        """This override adds constraints and then delegates most of the work
        to the base scheduler

        To avoid additional specialization, some requests are held back from the
        base scheduler but are restored after.
        """
        # First purge the full waiting queue into our holdback queue, preserving
        # priority, so that the base scheduler does not see them.
        holdback_queue: deque[Request] = deque()
        while self.waiting:
            holdback_queue.append(self.waiting.popleft())

        # Check if new requests can be scheduled.
        while holdback_queue:
            if self.can_schedule(holdback_queue[0]):
                # Add request to the waiting queue
                self.waiting.append(holdback_queue.popleft())
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
                "Scheduling a prefill step (%d prompt tokens), holding back %d requests",
                self.waiting[-1].num_prompt_tokens,
                len(holdback_queue),
            )
        else:
            running_holdback = []
            logger.debug("Scheduling a decode step of %d requests", len(self.running))

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule()

        # restore holdbacks after running the base scheduler
        self.running = self.running + running_holdback
        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

        return outputs

    def can_schedule(self, request) -> bool:
        max_prompt_batch_size = 1
        max_context_len = self.model_config.max_model_len

        # running and waiting queues are both empty -> start a new batch
        # which can always be scheduled
        if len(self.running) + len(self.waiting) == 0:
            return True

        # check that there is space in the current decode batch
        cond1 = len(self.running) + len(self.waiting) < self.max_num_running_reqs

        # check that there is space in the prefill batch
        cond2 = len(self.waiting) < max_prompt_batch_size

        tkv = self.tkv

        # shift tkv if the prompt length exceeds the current tkv
        if request.num_prompt_tokens > self.tkv:
            tkv += (
                math.ceil((request.num_prompt_tokens - self.tkv) / self.block_size)
                * self.block_size
            )

        # check that the number of requested tokens can be served
        cond3 = request.max_tokens <= (max_context_len - tkv)

        # check condition for all other sequences in the current decode batch
        # if the tkv has been shifted
        if tkv > self.tkv:
            for req in self.running:
                n_generated_output_tokens = req.num_computed_tokens - req.num_prompt_tokens
                max_tokens_remaining = req.max_tokens - n_generated_output_tokens
                cond3_current = max_tokens_remaining <= (max_context_len - tkv)
                cond3 = cond3 and cond3_current
                # early exiting loop if condition is violated
                if not cond3:
                    return False

        # check that there are enough free blocks/pages remaining
        # Note: we only have to do check in case of a running batches
        # (not start_new_batch), because the minimal number of blocks covers
        # the context length for a single sequence, so tkv < block size is ok
        num_blocks_required = math.ceil((tkv + request.max_tokens - 1) / self.block_size)
        # optimization: subtract the padding blocks from the reserved blocks
        num_fully_padded_blocks = math.floor((tkv - request.num_prompt_tokens) / self.block_size)
        num_blocks_required -= num_fully_padded_blocks
        cond4 = num_blocks_required <= self.n_free_blocks

        # check that batch size x tkv is smaller than the max supported number
        cond5 = lambda: self.check_batch_tkv_limit(
            request=request,
            tkv=tkv,
            running=self.running,
            max_batch_tkv_limit=self.max_batch_tkv_limit,
        )

        return cond1 and cond2 and cond3 and cond4 and cond5()

    def check_batch_tkv_limit(self, request, tkv, running, max_batch_tkv_limit) -> bool:
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
        """
        # checking if cached result can be used
        outer_key = tuple(r.request_id for r in running)  # decode batch changes
        inner_key = (request.request_id, max_batch_tkv_limit)  # new request changes
        cache = self._cache_check_batch_tkv_limit
        if (outer_key in cache) and (inner_key in cache[outer_key]):
            (lower, upper) = cache[outer_key][inner_key]
            if tkv <= lower or tkv >= upper:
                logger.debug(
                    "Cache hit function check_batch_tkv_limit: returning %s", str(tkv <= lower)
                )
                return tkv <= lower

        # Compute the effective token length of the new request
        new_req_max_tkv = tkv + request.max_tokens - 1

        # Compute token lengths for all running requests (decode batch)
        decode_req_max_tkvs = [
            tkv + req.max_tokens - 1 - (req.num_computed_tokens - req.num_prompt_tokens)
            for req in running
        ]
        # Sort decode requests token lengths in ascending order
        decode_req_max_tkvs.sort()

        # Initialize values
        batch_size = len(running) + 1
        max_batch_tkv = 0

        # Try adding the new request to the batch and check the max volume
        for decode_req_max_tkv in decode_req_max_tkvs:
            if new_req_max_tkv <= decode_req_max_tkv:
                # If the new request is shorter, it limits the batch volume
                max_batch_tkv = max(max_batch_tkv, batch_size * new_req_max_tkv)
                break
            else:
                # Otherwise, use the current (longer) request's volume
                max_batch_tkv = max(max_batch_tkv, batch_size * decode_req_max_tkv)
                # decrease batch_size by 1 as the current request finished
                batch_size -= 1

        return_value = max_batch_tkv <= int(max_batch_tkv_limit)

        if outer_key in cache:
            # decode batch has not changed
            if inner_key not in cache[outer_key]:
                # adding new request to present decode batch
                cache[outer_key][inner_key] = (-math.inf, math.inf)
        else:
            # decode batch has changed, empty the cache in place
            cache.clear()
            cache[outer_key] = {inner_key: (-math.inf, math.inf)}
            logger.debug(
                "Cleared cache of function check_batch_tkv_limit as the decode batch has changed."
            )

        # update lower bound (of acceptance) and upper bound (of rejection)
        (lower, upper) = cache[outer_key][inner_key]
        if return_value:
            lower = max(lower, tkv)
        else:
            upper = min(upper, tkv)
        assert lower < upper
        cache[outer_key][inner_key] = (lower, upper)
        logger.debug(
            "Saved cache of function check_batch_tkv_limit: %s",
            self._cache_check_batch_tkv_limit[outer_key][inner_key],
        )

        return return_value


class ChunkedPrefillSpyreScheduler(ContinuousBatchingSpyreScheduler):
    """
    Chunked-Prefill Scheduling policy

    The prefill vs. decode priority policy is the following:
        - Current prefill request priority: A new request cannot start prefill
           while another request's prefill is on-going

        - Prefill step interleaving: The prefill steps are interleaved with
            one decode step: as long as there are decoding requests, two
            prefill steps cannot be consecutive

        - General prefill priority: conditioned on interleaving constraint,
            prefill has priority over decode

        - No empty step: if a prefill step is prevented because it doesn't
            satisfy Spyre's specific constraints, a decode step is scheduled

    Spyre scheduling constraints:
        - Blocks constraint: all the blocks necessary to serve a request are
            allocated at the time of scheduling the first chunk of a chunked
            prefill. There must be enough blocks to accommodate the prompt and
            the maximum number of output tokens

        Note: all the remaining constraints need to be satisfied at the time
        of scheduling the last chunk of a chunked prefill

        - Prefill batch size: prefill batch is of size 1, only one request's
            chunked prefill can be scheduled at a time

        - Decode batch size: cannot have more than max_num_seqs running
            requests, including prefill and decode

        - Max model length constraint: the number of requested tokens must fit
            between the maximum TKV of all the running requests and the end of
            the model's context

        - Volumetric constraint: the total "surface" defined by the running
            requests should never exceed `VLLM_DT_MAX_BATCH_TKV_LIMIT`. See
            `check_batch_tkv_limit()` method for details.

        the surface defined by the maximum TKV of
            all the running requests and the number of running requests must
            not exceed the limit defined by `VLLM_DT_MAX_BATCH_TKV_LIMIT`
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.chunk_size = self.scheduler_config.max_num_batched_tokens

        # We want to keep track of requests for which the prefill is ongoing.
        # Theoretically, only one request can be prefilled at a time, but we
        # keep a list to be able to batch prefills in the future.
        self.ongoing_prefills: list[Request] = []

        # Prefills interleaving: if the feature flag is set, prefill operations
        # are interleaved with a decode step. This allows to minimize currently
        # decoding requests
        self.do_interleaving: bool = envs_spyre.VLLM_SPYRE_CP_INTERLEAVE_STEPS
        self.previous_step_was_prefill: bool = False

        # KV cache metrics
        # Prefix cache stats aggregated between each `make_stats` call
        self.prefix_cache_stats = PrefixCacheStats()
        # Gauge for kv cache usage, updated from every model runner output
        self.kv_cache_usage_percent = 0.0

    def update_from_output(self, scheduler_output, model_runner_output):
        assert isinstance(model_runner_output, CPSpyreModelRunnerOutput), (
            "Expecting an instance of CPSpyreModelRunnerOutput when doing chunked prefill."
        )

        # Update the correct num_computed_tokens value given left-padding and
        # prefix cache hit info
        for req in self.ongoing_prefills:
            # The number of computed tokens only need to be adapted when it is
            # the first chunk of a multi-chunk prefill
            is_first_chunk = req.num_computed_tokens <= self.chunk_size
            is_last_chunk = req.num_computed_tokens == req.num_prompt_tokens
            if is_first_chunk and not is_last_chunk:
                left_padding = model_runner_output.left_padding.get(req.request_id, 0)
                prefix_cache_len = model_runner_output.prefix_cache_hit_len.get(req.request_id, 0)

                req.num_computed_tokens = self.adjust_computed_tokens(
                    computed_tokens=req.num_computed_tokens,
                    left_padding=left_padding,
                    prefix_cache_len=prefix_cache_len,
                )

        # Remove completed prefills
        self.ongoing_prefills = [
            req for req in self.ongoing_prefills if req.num_computed_tokens < req.num_prompt_tokens
        ]

        # Update KV Cache info
        self.kv_cache_usage_percent = model_runner_output.kv_cache_usage
        prefix_cache_stats = model_runner_output.prefix_cache_stats
        if prefix_cache_stats is not None:
            # We don't use PrefixCacheStats.record because:
            # 1. It's not supported in vllm v0.11.0
            # 2. It could eventually be the case that requests > 1
            self.prefix_cache_stats.requests += prefix_cache_stats.requests
            self.prefix_cache_stats.queries += prefix_cache_stats.queries
            self.prefix_cache_stats.hits += prefix_cache_stats.hits

        return super().update_from_output(scheduler_output, model_runner_output)

    def adjust_computed_tokens(
        self, computed_tokens: int, left_padding: int, prefix_cache_len: int
    ) -> int:
        """
        Returns an adjusted `num_computed_tokens` given left padding and prefix
        cache hit info.
        """
        # The prefix cache length is already adjusted for left padding.
        # If it's bigger than the number of computed tokens, then we hit more
        # prefix cache than we scheduled.
        if prefix_cache_len > computed_tokens:
            assert (prefix_cache_len + left_padding) % self.chunk_size == 0
            return prefix_cache_len
        # Otherwise just account for the left padding
        return computed_tokens - left_padding

    def schedule(self) -> "SchedulerOutput":
        """
        The chunked prefill scheduling policy is enforced in this method, then
        delegates the final scheduling decision to the base scheduler

        To avoid additional specialization, some requests are held back from the
        base scheduler but are restored after
        """
        # First purge the full waiting queue into our holdback queue, preserving
        # priority, so that the base scheduler does not see them.
        holdback_queue: deque[Request] = deque()
        while self.waiting:
            holdback_queue.append(self.waiting.popleft())

        # Check if new requests can be scheduled for prefill
        while holdback_queue:
            if self.can_schedule_prefill(holdback_queue[0]):
                new_request = holdback_queue.popleft()
                logger.debug(
                    "Scheduling a new request (%d prompt tokens), holding back %d requests",
                    new_request.num_prompt_tokens,
                    len(holdback_queue),
                )

                # Add request to the waiting queue
                self.waiting.append(new_request)
            else:
                # Otherwise, we simply stop here so that the scheduler
                # can work with the batch we have
                break

        assert len(self.ongoing_prefills) <= 1, (
            "Only one request can be prefilled at a time, but got %d" % len(self.ongoing_prefills)
        )
        assert len(self.waiting) == 0 or len(self.ongoing_prefills) == 0, (
            "Cannot schedule new requests while another request prefill is ongoing."
        )
        assert all(r in self.running for r in self.ongoing_prefills), (
            "Ongoing prefill requests must be in the running queue."
        )

        # Check ongoing prefills
        if self.ongoing_prefills:
            # Some running requests are currently being prefilled. We need to
            # separate them from currently decoding requests, and schedule
            # them separately. Either we schedule a chunked prefill step, or a
            # decoding step

            assert len(self.ongoing_prefills) == 1

            schedule_prefill = self.can_schedule_prefill(self.ongoing_prefills[0])

            if schedule_prefill:
                running_holdback = [r for r in self.running if r not in self.ongoing_prefills]
                self.running = self.ongoing_prefills
                self.previous_step_was_prefill = True
            else:
                self.running = [r for r in self.running if r not in self.ongoing_prefills]
                running_holdback = self.ongoing_prefills
                self.previous_step_was_prefill = False

        # Check new requests to prefill
        elif len(self.waiting) > 0:
            self.ongoing_prefills.extend(self.waiting)
            # Hide current decodes from the scheduler
            running_holdback = self.running
            self.running = []
            self.previous_step_was_prefill = True
        else:
            self.previous_step_was_prefill = False
            running_holdback = []

        # delegate to super of SpyreScheduler: base V1 Scheduler
        outputs = super(SpyreScheduler, self).schedule()

        # restore holdbacks after running the base scheduler
        self.running = self.running + running_holdback
        while holdback_queue:
            self.waiting.append(holdback_queue.popleft())

        # Log the scheduled tokens not at every step, but when doing chunked
        # prefill. These include decode steps during interleaving
        if self.ongoing_prefills or any(
            r.num_computed_tokens <= r.num_prompt_tokens + 1 for r in self.running
        ):
            logger.debug("Scheduled tokens in this step: %s", outputs.num_scheduled_tokens)
        return outputs

    def can_schedule_prefill(self, request: Request) -> bool:
        # running and waiting queues are both empty, we can start a new batch
        # which can always be scheduled
        if len(self.running) + len(self.waiting) == 0:
            return True

        if not self._has_scheduling_priority(request):
            return False

        return self._satisfies_constraints(request)

    def _satisfies_constraints(self, request: Request) -> bool:
        is_first_chunk = request.num_computed_tokens == 0
        is_last_chunk = (request.num_prompt_tokens - request.num_computed_tokens) <= self.chunk_size

        if not self.do_interleaving:
            # All the prefills are consecutive, so the first chunk has to
            # satisfy all the constraints, and we don't need to check them again
            # for subsequent chunks.
            if not is_first_chunk:
                return True

            return self._satisfies_first_chunk_constraints(
                request
            ) and self._satisfies_last_chunk_constraints(request)

        can_schedule = True
        if is_first_chunk:
            can_schedule = self._satisfies_first_chunk_constraints(request)

        if is_last_chunk:
            can_schedule = can_schedule and self._satisfies_last_chunk_constraints(request)

        return can_schedule

    def _satisfies_first_chunk_constraints(self, request: Request) -> bool:
        """First chunked prefill can be scheduled only if there is space in the
        prefill batch and if there is enough blocks to accommodate entire
        request"""

        # TODO theoretically we could already do a chunked prefill even
        # if the decode batch is full, but the current implementation of input
        # batch doesn't allow to do so.
        num_running = len(self.running)
        cond1 = num_running + len(self.waiting) < self.max_num_running_reqs

        # check that there is space in the prefill batch
        max_prefill_batch_size = 1
        cond2 = len(self.waiting) < max_prefill_batch_size

        # all the blocks for the request are allocated at the time of the first
        # chunked prefill. We need to check here that there are enough free
        # blocks/pages remaining
        # Note: we only have to do check in case of a running batches
        # (not start_new_batch), because the minimal number of blocks covers
        # the context length for a single sequence, so tkv < block size is ok
        prompt_len = request.num_prompt_tokens
        total_tokens = prompt_len + request.max_tokens - 1
        num_blocks_required = math.ceil(total_tokens / self.block_size)
        cond3 = num_blocks_required <= self.n_free_blocks
        return cond1 and cond2 and cond3

    def _satisfies_last_chunk_constraints(self, request: Request) -> bool:
        """Last chunked prefill can be scheduled only if there is enough space
        in the decode batch, and if all the other spyre-related conditions
        are satisfied."""
        decoding_requests = [r for r in self.running if r not in self.ongoing_prefills]
        max_context_len = self.model_config.max_model_len

        # check that there is space in the current decode batch
        num_running = len(decoding_requests)
        cond1 = num_running + len(self.waiting) < self.max_num_running_reqs

        # calculate new max tkv of the batch given the new sequence joins
        # considers all possible cases:
        # - prompt_len > self.tkv and fall into different blocks
        # - prompt_len and self.tkv fall within the same block
        # - prompt_len < self.tkv and fall into different blocks
        prompt_len = request.num_prompt_tokens
        n_blocks = math.floor(max(self.tkv, prompt_len) / self.block_size)
        new_req_tkv = n_blocks * self.block_size + prompt_len % self.block_size

        # check that the number of requested tokens can be served for the
        # new sequence (optimal condition)
        # note that the -1 comes from the token we generate during prefill
        cond2 = request.max_tokens - 1 <= (max_context_len - new_req_tkv)
        # check cond2 for all other sequences in the current decode batch
        for req in decoding_requests:
            # current tkv of the (left aligned) decode sequence
            dec_req_tkv = n_blocks * self.block_size + req.num_computed_tokens % self.block_size
            n_generated_output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            max_tokens_remaining = req.max_tokens - n_generated_output_tokens
            # note that the -1 comes from the token we generate during prefill
            cond2_current = max_tokens_remaining - 1 <= (max_context_len - dec_req_tkv)
            cond2 = cond2 and cond2_current
            # early exiting loop if violated 2nd condition
            if not cond2:
                return False

        # check that batch size x tkv is smaller than the max supported number
        # Note: using max_tkv is a conservative upper bound here. For the
        # optimal check we need model runner to return per sequence tkvs
        cond3 = lambda: self.check_batch_tkv_limit_cp(
            request=request,
            new_req_tkv=new_req_tkv,
            n_blocks=n_blocks,
            running=decoding_requests,
            max_batch_tkv_limit=self.max_batch_tkv_limit,
        )

        return cond1 and cond2 and cond3()

    def _has_scheduling_priority(self, request):
        decoding_requests = [r for r in self.running if r not in self.ongoing_prefills]

        # If we do interleaving, then two consecutive prefill steps are
        # forbidden when there are decoding requests
        if self.do_interleaving and self.previous_step_was_prefill and len(decoding_requests) > 0:
            return False

        # Requests that are already prefilling are prioritized over new requests
        if request in self.ongoing_prefills:
            return True

        # We can start prefilling a new requests if we satisfy the maximum
        # number of concurrent prefills
        max_concurrent_prefills = 1
        num_prefills = len(self.waiting) + len(self.ongoing_prefills)
        return num_prefills < max_concurrent_prefills

    def check_batch_tkv_limit_cp(
        self, request, new_req_tkv, n_blocks, running, max_batch_tkv_limit
    ) -> bool:
        """
        Check whether adding a new sequence to the decode batch would violate
        Spyre's maximum batch volume constraint for chunked prefill.

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
        """

        # Compute the effective token length of the new request
        new_req_max_tkv = new_req_tkv + request.max_tokens - 1

        # Compute token lengths for all running requests (decode batch)
        decode_req_max_tkvs = []
        for req in running:
            # current tkv of the (left aligned) decode sequence
            dec_req_tkv = n_blocks * self.block_size + req.num_computed_tokens % self.block_size
            n_generated_output_tokens = req.num_computed_tokens - req.num_prompt_tokens
            dec_req_max_tkv = dec_req_tkv + (req.max_tokens - n_generated_output_tokens) - 1
            decode_req_max_tkvs.append(dec_req_max_tkv)

        # Sort decode requests token lengths in ascending order
        decode_req_max_tkvs.sort()

        # Initialize values
        # The request is already in the running queue if it has done a first
        # chunked prefill
        batch_size = len(running)
        if request not in running:
            batch_size += 1
        max_batch_tkv = 0

        # Try adding the new request to the batch and check the max volume
        for decode_req_max_tkv in decode_req_max_tkvs:
            if new_req_max_tkv <= decode_req_max_tkv:
                # If the new request is shorter, it limits the batch volume
                max_batch_tkv = max(max_batch_tkv, batch_size * new_req_max_tkv)
                break
            else:
                # Otherwise, use the current (longer) request's volume
                max_batch_tkv = max(max_batch_tkv, batch_size * decode_req_max_tkv)
                # decrease batch_size by 1 as the current request finished
                batch_size -= 1

        return max_batch_tkv <= int(max_batch_tkv_limit)

    def finish_requests(
        self,
        request_ids: Union[str, Iterable[str]],
        finished_status,
    ) -> None:
        """Handles removing finished requests from ongoing_prefills"""
        if isinstance(request_ids, str):
            request_ids = (request_ids,)

        # first defer to vLLM scheduler where validation is handled
        super(SpyreScheduler, self).finish_requests(
            request_ids=request_ids, finished_status=finished_status
        )

        self.ongoing_prefills = [
            r for r in self.ongoing_prefills if r.request_id not in request_ids
        ]

    def make_stats(self, *args, **kwargs) -> SchedulerStats | None:
        """Update the scheduler stats from the base scheduler.
        These are used in the vllm StatLoggers, which are responsible for
        reporting stats in logs and metrics.
        """
        base_stats = super().make_stats(*args, **kwargs)

        if base_stats is not None:
            base_stats.kv_cache_usage = self.kv_cache_usage_percent
            base_stats.prefix_cache_stats = self.prefix_cache_stats
            # Every time `make_stats` is called we reset the prefix cache stats.
            # This mimics how the base scheduler handles the kv cache stats
            self.prefix_cache_stats = PrefixCacheStats()

        return base_stats

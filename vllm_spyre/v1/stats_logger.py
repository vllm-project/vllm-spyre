import dataclasses
import json
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Optional

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.engine import async_llm, llm_engine
from vllm.v1.metrics.loggers import StatLoggerBase, StatLoggerManager
from vllm.v1.metrics.stats import (FinishedRequestStats, IterationStats,
                                   SchedulerStats)

from vllm_spyre import envs as envs_spyre

logger = init_logger(__name__)


@dataclasses.dataclass
class PerfRecord:
    """One single record for that goes into the .jsonl file.
    Contains info about a single request"""
    # ISO timestamp w/ milliseconds
    timestamp: str
    # timing info
    engine_stats: FinishedRequestStats
    # estimated time pre-empted for other prefills
    estimated_prefill_interrupt: float
    # estimated ITL without prefill interrupts
    estimated_decode_only_itl: float


class FileStatLogger(StatLoggerBase):

    def __init__(self, vllm_config: VllmConfig, engine_index=0):
        super().__init__(vllm_config, engine_index)

        self.enabled = (envs_spyre.VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED
                        and envs_spyre.VLLM_SPYRE_USE_CB)

        perf_dir = Path(envs_spyre.VLLM_SPYRE_PERF_METRIC_LOGGING_DIR)
        if not perf_dir.exists():
            perf_dir.mkdir(parents=True)

        self.perf_file = Path(envs_spyre.VLLM_SPYRE_PERF_METRIC_LOGGING_DIR
                              ) / "request_metrics.jsonl"

        if self.enabled and engine_index == 0:
            logger.info(
                "Initializing vllm-spyre perf debug logger. Writing perf info "
                "to: %s", str(self.perf_file))

        # Clear any old metrics out first
        if self.perf_file.exists():
            self.perf_file.unlink()

        self.perf_file.touch()

        self.iso_format = "%Y-%m-%dT%H:%M:%S.%f"

        self._prefill_tuples = []
        self._max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self._last_ts = 0

    def record(self,
               scheduler_stats: Optional[SchedulerStats],
               iteration_stats: Optional[IterationStats],
               engine_idx: int = 0):
        if not self.enabled or engine_idx != 0:
            # Only log from rank 0
            return

        if iteration_stats is None:
            return

        if iteration_stats.num_prompt_tokens > 0:
            self._save_prefill_time(iteration_stats)
        self._last_ts = iteration_stats.iteration_timestamp

        if not iteration_stats.finished_requests:
            # Only log finished requests
            return

        ### Convert float timestamp to human readable string
        text_timestamp = datetime.fromtimestamp(
            iteration_stats.iteration_timestamp).strftime(self.iso_format)[:-3]

        records_to_write: list[str] = []
        for r in iteration_stats.finished_requests:
            # Calculate some estimates to add to the engine stats
            estimated_prefill_interrupt = \
                self.estimate_prefill_interrupt_lower_bound(r)

            estimated_decode_itl = (r.decode_time -
                                    estimated_prefill_interrupt) / max(
                                        r.num_generation_tokens - 1, 1)

            record = PerfRecord(
                timestamp=text_timestamp,
                engine_stats=r,
                estimated_decode_only_itl=estimated_decode_itl,
                estimated_prefill_interrupt=estimated_prefill_interrupt)
            records_to_write.append(json.dumps(dataclasses.asdict(record)))

        with open(self.perf_file, "a") as f:
            f.write("\n".join(records_to_write) + "\n")

    def log_engine_initialized(self):
        return super().log_engine_initialized()

    def _save_prefill_time(self, iteration_stats: IterationStats):
        """If this iteration was a prefill, then save the a tuple of the current
        time and prefill time. This will be used later to estimate a lower bound
        of the amount of time that other sequences were
        interrupted for this prefill to happen.
        
        This is only relevant because the batching implementation has to pause
        the running batch of decoding sequences to prefill a single sequence.
        """
        maybe_prefill_time = iteration_stats.iteration_timestamp - self._last_ts
        # TTFT here includes queueing and we don't have access to the iteration
        # duration itself so we have to try to calculate our own prefill time.
        # If we calculate an interval that was less than the reported TTFT, then
        # use it as the prefill time
        maybe_prefill_time = min(maybe_prefill_time,
                                 iteration_stats.time_to_first_tokens_iter[0])

        # Tuple is (timestamp, prefill_time)
        self._prefill_tuples.append(
            (iteration_stats.iteration_timestamp, maybe_prefill_time))
        if len(self._prefill_tuples) > 2 * self._max_batch_size:
            # Delete older prefills, we can't hold everything in memory
            # Not guaranteed to be lossless
            self._prefill_tuples.pop(0)

    def estimate_prefill_interrupt_lower_bound(
            self, finished_request: FinishedRequestStats) -> float:
        """Returns a lower bound estimate on the time (in ms) that this request
        was interrupted for other requests to prefill to join the batch"""
        estimated_prefill_interrupt = 0

        # NB: use current time instead of iteration timestamp to ensure that we
        # exclude current request's prefill
        slop = 0.001
        decode_start_time = time.time() - finished_request.decode_time + slop
        for i in range(len(self._prefill_tuples)):
            if self._prefill_tuples[i][0] > decode_start_time:
                # Sum up all prefills past decode start time
                estimated_prefill_interrupt = sum(
                    r[1] for r in self._prefill_tuples[i:])
                break
        return estimated_prefill_interrupt


def file_stat_logger_factory(config: VllmConfig | None,
                             engine_index=0) -> FileStatLogger:
    """Factory method accepted by vllm engine initializers"""
    return FileStatLogger(config, engine_index)


def patch_async_llm_stat_loggers():
    """
    🌶️🌶️🌶️
    Platforms cannot alter the initialization of a vllm engine, and the 
    `stat_loggers` parameter is not user-settable via `EngineArgs`.

    So we resort to patching the initialization of the StatsLoggerManager to 
    inject our own stats logger. This _should_ also be compatible with versions
    of vllm prior to the addition of `stats_loggers` engine parameter.
    🌶️🌶️🌶️
    """
    logger.debug("Setting up perf logger injection")
    original_init = StatLoggerManager.__init__

    @wraps(original_init)
    def new_init(self, *args, **kwargs):
        logger.debug("Injecting vllm-spyre perf logger factory")
        if "custom_stat_loggers" not in kwargs or kwargs[
                "custom_stat_loggers"] is None:
            kwargs["custom_stat_loggers"] = []

        kwargs["custom_stat_loggers"].append(file_stat_logger_factory)

        original_init(self, *args, **kwargs)

    async_llm.StatLoggerManager.__init__ = new_init
    llm_engine.StatLoggerManager.__init__ = new_init

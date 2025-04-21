""" Spyre performance metric logging """
import os
import time

import vllm_spyre.envs as envs


def create_perf_metric_logger(rank: int):
    """ Create a performance metric logging object. """
    if envs.VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED == 1:
        return SpyrePerfMetricFileLogger(rank)
    return SpyrePerfMetricLoggerBase(rank)


class SpyrePerfMetricLoggerBase:
    """ A no-op base class for use when logging is disabled """

    def __init__(self, rank: int):
        self.rank = rank

    def __del__(self):
        pass

    def log(self, description: str, value, **kwargs):
        """ Log value with description. kwargs is used as a dictionary of
            additional labels to further describe the logged value. """
        pass


class SpyrePerfMetricFileLogger(SpyrePerfMetricLoggerBase):
    """ A per-rank file logging object """

    def __init__(self, rank: int):
        super().__init__(rank)
        self.time_fmt = "%m-%d %H:%M:%S"
        self.log_path = os.path.join(envs.VLLM_SPYRE_PERF_METRIC_LOGGING_DIR,
                                     f"perf_log_rank_{str(rank)}.txt")
        # Cleanup previous metrics files
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
        # Output configuration variables to ease understanding of logs
        self.log("VLLM_SPYRE_USE_CB", envs.VLLM_SPYRE_USE_CB)
        self.log("VLLM_SPYRE_WARMUP_BATCH_SIZES",
                 envs.VLLM_SPYRE_WARMUP_BATCH_SIZES)
        self.log("VLLM_SPYRE_WARMUP_PROMPT_LENS",
                 envs.VLLM_SPYRE_WARMUP_PROMPT_LENS)
        self.log("VLLM_SPYRE_WARMUP_NEW_TOKENS",
                 envs.VLLM_SPYRE_WARMUP_NEW_TOKENS)
        self.log("AIU_WORLD_SIZE", os.getenv("AIU_WORLD_SIZE", 0))
        self.log("DT_OPT", os.getenv("DT_OPT", ""))

    def log(self, description: str, value, **kwargs):
        text = f"{time.strftime(self.time_fmt)}, {description}, {value}"
        for kw in kwargs:
            text += f", {kw}, {kwargs[kw]}"
        text += "\n"
        with open(self.log_path, "a") as f:
            f.write(text)

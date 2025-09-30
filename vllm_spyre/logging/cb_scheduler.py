"""Spyre continuous batching scheduler logging. Collects state at each step"""
import json
import os
import time
from datetime import datetime
from typing import Any

import vllm_spyre.envs as envs


def create_cb_scheduler_logger(max_model_len: int, max_num_seqs: int):
    if envs.VLLM_SPYRE_CB_SCHEDULER_LOGGING_ENABLED == 1:
        return CBSchedulerLogger(max_model_len, max_num_seqs)
    return CBSchedulerLoggerBase()


class CBSchedulerLoggerBase:
    """ A no-op base class for use when logging is disabled """

    def __init__(self):
        pass

    def __del__(self):
        pass

    def log(self, waiting: list[Any], running: list[Any], tkv: int,
            n_free_blocks: int):
        pass


class CBSchedulerLogger(CBSchedulerLoggerBase):
    """ A continuous batching logging object"""

    def __init__(self, max_model_len: int, max_num_seqs: int):
        super().__init__()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(
            envs.VLLM_SPYRE_CB_SCHEDULER_LOGGING_DIR,
            f"cb_scheduler_logging_{timestamp}.jsonl")

        first_line = {
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs
        }
        json_data_line = json.dumps(first_line)
        with open(self.log_path, "a") as f:
            f.write(json_data_line + "\n")

    def log(self, waiting: list[Any], running: list[Any], tkv: int,
            n_free_blocks: int):
        data = {}
        # metadata
        data["logging_time"] = time.time()
        data["tkv"] = tkv
        data["n_free_blocks"] = n_free_blocks

        # waiting list
        data["waiting"] = {}
        data["waiting"]["ids"] = [w.request_id for w in waiting]
        data["waiting"]["arrival_times"] = [w.arrival_time for w in waiting]

        # running list
        data["running"] = {}
        data["running"]["ids"] = [r.request_id for r in running]
        data["running"]["arrival_time"] = [r.arrival_time for r in running]
        data["running"]["prompt_len"] = [
            len(r.prompt_token_ids or []) for r in running
        ]
        data["running"]["max_tokens"] = [r.max_tokens for r in running]

        json_data_line = json.dumps(data)
        with open(self.log_path, "a") as f:
            f.write(json_data_line + "\n")

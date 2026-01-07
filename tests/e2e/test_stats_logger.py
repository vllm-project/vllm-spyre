import json
from pathlib import Path

import pytest
from spyre_util import ModelInfo, get_chicken_soup_prompts
from vllm import LLM

from vllm_spyre import envs as envs_spyre


@pytest.mark.cpu
@pytest.mark.cb
def test_file_stats_logger(model: ModelInfo, max_model_len, max_num_seqs, tmp_path):
    prompts = get_chicken_soup_prompts(4)

    envs_spyre.override("VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED", "1")
    envs_spyre.override("VLLM_SPYRE_PERF_METRIC_LOGGING_DIR", str(tmp_path))
    envs_spyre.override("VLLM_SPYRE_USE_CB", "1")
    envs_spyre.override("VLLM_SPYRE_DYNAMO_BACKEND", "eager")

    model = LLM(
        model=model.name,
        revision=model.revision,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        disable_log_stats=False,
    )
    model.generate(prompts=prompts)

    assert Path(tmp_path / "request_metrics.jsonl").exists()

    with Path(tmp_path / "request_metrics.jsonl").open() as f:
        for line in f.readlines():
            data = json.loads(line)
            assert "prefill_interrupt_seconds" in data
            assert "e2e_latency_seconds" in data
            assert "timestamp" in data

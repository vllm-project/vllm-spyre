import pytest
from spyre_util import write_sample_model_config
from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)

from vllm_spyre.platform import SpyrePlatform


@pytest.mark.parametrize("batch_type", ["sb", "cb"])
def test_handle_disable_compilation(monkeypatch, tmp_path, batch_type):
    """
    Test _handle_disable_compilation for static and continuous batching.
    """

    # Patch version to avoid test failures around version mismatch
    monkeypatch.setattr("vllm_spyre._version.version", "0.8.0")

    if batch_type == "sb":
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "0")
        sample_model_config = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 4,
                "VLLM_SPYRE_WARMUP_PROMPT_LENS": "128",
                "VLLM_SPYRE_WARMUP_NEW_TOKENS": "128",
                "VLLM_SPYRE_WARMUP_BATCH_SIZES": "1",
            },
        }
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS", "128")
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS", "128")
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_BATCH_SIZES", "1")

    else:
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
        sample_model_config = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 4,
                "VLLM_DT_MAX_CONTEXT_LEN": 256,
                "VLLM_DT_MAX_BATCH_SIZE": 2
            },
        }

    write_sample_model_config(tmp_path, sample_model_config)

    monkeypatch.setenv("DISABLE_COMPILATION", "true")
    monkeypatch.setenv("TORCH_SENDNN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "1")

    dummy_vllm_config = VllmConfig(
        model_config=ModelConfig(
            model="ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8",
            max_model_len=256),
        parallel_config=ParallelConfig(tensor_parallel_size=4),
        scheduler_config=SchedulerConfig(max_num_seqs=2))

    SpyrePlatform._handle_disable_compilation(dummy_vllm_config,
                                              is_decoder=True)

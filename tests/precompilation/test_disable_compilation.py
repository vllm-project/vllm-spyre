import logging
import os

import pytest
from spyre_util import (DecodeWarmupShapes, patch_warmup_shapes,
                        write_sample_model_config)
from vllm.config import (ModelConfig, ParallelConfig, SchedulerConfig,
                         VllmConfig)

from vllm_spyre.compilation_utils import PRE_COMPILE_MODEL_CATALOG_FILENAME


@pytest.mark.precompilation
@pytest.mark.parametrize("batch_type", ["sb", "cb"])
def test_handle_disable_compilation(caplog_vllm_spyre, monkeypatch, tmp_path,
                                    batch_type):
    """
    Test handle_disable_compilation for static and continuous batching.
    Note: since the validation here is only giving warning in case of mismatch,
    these tests will only fail if there is a bug in the logic
    """

    # Patch version to avoid test failures around version mismatch
    monkeypatch.setattr("vllm_spyre._version.version", "0.8.0")

    if batch_type == "sb":
        patch_warmup_shapes(DecodeWarmupShapes([(128, 128, 1)]), monkeypatch)
        sample_model_config = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_SPYRE_WARMUP_PROMPT_LENS": "128",
                "VLLM_SPYRE_WARMUP_NEW_TOKENS": "128",
                "VLLM_SPYRE_WARMUP_BATCH_SIZES": "1",
            },
        }

    else:
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
        sample_model_config = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_DT_MAX_CONTEXT_LEN": 256,
                "VLLM_DT_MAX_BATCH_SIZE": 2
            },
        }

    write_sample_model_config(tmp_path, sample_model_config)

    monkeypatch.setenv("VLLM_SPYRE_REQUIRE_PRECOMPILED_DECODERS", "1")
    monkeypatch.setenv("TORCH_SENDNN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "1")
    # Register the DISABLE_COMPILATION env variable with monkeypatch so that
    # it resets the value to its previous state after the
    # test as a cleanup
    monkeypatch.setenv("DISABLE_COMPILATION", "")

    with caplog_vllm_spyre.at_level(logging.INFO):
        _ = VllmConfig(model_config=ModelConfig(
            model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            max_model_len=256),
                       parallel_config=ParallelConfig(tensor_parallel_size=2),
                       scheduler_config=SchedulerConfig(max_num_seqs=2))
        assert "[PRECOMPILED_WARN] Setting DISABLE_COMPILATION" \
            in caplog_vllm_spyre.text

    assert "DISABLE_COMPILATION" in os.environ
    assert os.getenv("DISABLE_COMPILATION") == "true"


@pytest.mark.precompilation
@pytest.mark.parametrize("batch_type", ["sb", "cb"])
def test_handle_disable_compilation_catalog(caplog_vllm_spyre, monkeypatch,
                                            tmp_path, batch_type):
    """
    Test handle_disable_compilation for static and continuous batching.
    Note: since the validation here is only giving warning in case of mismatch,
    these tests will only fail if there is a bug in the logic
    """

    # Patch version to avoid test failures around version mismatch
    monkeypatch.setattr("vllm_spyre._version.version", "0.8.0")

    if batch_type == "sb":
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS", "128")
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS", "128")
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_BATCH_SIZES", "1")
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "0")
        sample_model_config1 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_SPYRE_WARMUP_PROMPT_LENS": "128",
                "VLLM_SPYRE_WARMUP_NEW_TOKENS": "128",
                "VLLM_SPYRE_WARMUP_BATCH_SIZES": "1",
            },
        }
        sample_model_config2 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_SPYRE_WARMUP_PROMPT_LENS": "256",
                "VLLM_SPYRE_WARMUP_NEW_TOKENS": "256",
                "VLLM_SPYRE_WARMUP_BATCH_SIZES": "1",
            },
        }

    else:
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
        sample_model_config1 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_DT_MAX_CONTEXT_LEN": 256,
                "VLLM_DT_MAX_BATCH_SIZE": 2
            },
        }
        sample_model_config2 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_DT_MAX_CONTEXT_LEN": 512,
                "VLLM_DT_MAX_BATCH_SIZE": 2
            },
        }

    sample_model_config = [sample_model_config1, sample_model_config2]

    write_sample_model_config(tmp_path,
                              sample_model_config,
                              filename=PRE_COMPILE_MODEL_CATALOG_FILENAME)

    monkeypatch.setenv("VLLM_SPYRE_REQUIRE_PRECOMPILED_DECODERS", "1")
    monkeypatch.setenv("TORCH_SENDNN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "1")
    # Register the DISABLE_COMPILATION env variable with monkeypatch so that
    # it resets the value to its previous state after the
    # test as a cleanup
    monkeypatch.setenv("DISABLE_COMPILATION", "")

    with caplog_vllm_spyre.at_level(logging.INFO):
        _ = VllmConfig(model_config=ModelConfig(
            model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            max_model_len=256),
                       parallel_config=ParallelConfig(tensor_parallel_size=2),
                       scheduler_config=SchedulerConfig(max_num_seqs=2))

        assert "[PRECOMPILED_WARN] Setting DISABLE_COMPILATION" \
            in caplog_vllm_spyre.text

    assert "DISABLE_COMPILATION" in os.environ
    assert os.getenv("DISABLE_COMPILATION") == "true"


@pytest.mark.precompilation
@pytest.mark.parametrize("batch_type", ["sb", "cb"])
def test_catalog_config_mismatch(caplog_vllm_spyre, monkeypatch, tmp_path,
                                 batch_type):
    """
    Test handle_disable_compilation for static and continuous batching
    and verify if we get proper error in case of mismatch catalog file
    """

    # Patch version to avoid test failures around version mismatch
    monkeypatch.setattr("vllm_spyre._version.version", "0.8.0")

    if batch_type == "sb":
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_PROMPT_LENS", "64")
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_NEW_TOKENS", "128")
        monkeypatch.setenv("VLLM_SPYRE_WARMUP_BATCH_SIZES", "1")
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "0")
        sample_model_config1 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_SPYRE_WARMUP_PROMPT_LENS": "128",
                "VLLM_SPYRE_WARMUP_NEW_TOKENS": "128",
                "VLLM_SPYRE_WARMUP_BATCH_SIZES": "1",
            },
        }
        sample_model_config2 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_SPYRE_WARMUP_PROMPT_LENS": "256",
                "VLLM_SPYRE_WARMUP_NEW_TOKENS": "256",
                "VLLM_SPYRE_WARMUP_BATCH_SIZES": "1",
            },
        }

    else:
        monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")
        sample_model_config1 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_DT_MAX_CONTEXT_LEN": 256,
                "VLLM_DT_MAX_BATCH_SIZE": 2
            },
        }
        sample_model_config2 = {
            "vllm_spyre_version": "0.8.0",
            "data": {
                "MODEL_NAME": "/models/granite-3.3-8b-instruct-FP8",
                "NUM_AIUS": 2,
                "VLLM_DT_MAX_CONTEXT_LEN": 512,
                "VLLM_DT_MAX_BATCH_SIZE": 2
            },
        }

    sample_model_config = [sample_model_config1, sample_model_config2]

    write_sample_model_config(tmp_path,
                              sample_model_config,
                              filename=PRE_COMPILE_MODEL_CATALOG_FILENAME)

    monkeypatch.setenv("VLLM_SPYRE_REQUIRE_PRECOMPILED_DECODERS", "1")
    monkeypatch.setenv("TORCH_SENDNN_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("TORCH_SENDNN_CACHE_ENABLE", "1")
    # Register the DISABLE_COMPILATION env variable with monkeypatch so that
    # it resets the value to its previous state after the
    # test as a cleanup
    monkeypatch.setenv("DISABLE_COMPILATION", "")

    with caplog_vllm_spyre.at_level(logging.WARNING):
        _ = VllmConfig(model_config=ModelConfig(
            model="ibm-ai-platform/micro-g3.3-8b-instruct-1b",
            max_model_len=64),
                       parallel_config=ParallelConfig(tensor_parallel_size=2),
                       scheduler_config=SchedulerConfig(max_num_seqs=2))
        assert "[PRECOMPILED_WARN]" in caplog_vllm_spyre.text
        assert "doesn't match any of the pre-compiled model " \
        "configurations. Catalog:" in caplog_vllm_spyre.text

    assert "DISABLE_COMPILATION" in os.environ
    assert os.getenv("DISABLE_COMPILATION") == "true"

import logging

import pytest
import yaml
from pytest import LogCaptureFixture

from vllm_spyre.config import runtime_config_validator
from vllm_spyre.config.runtime_config_validator import (
    validate_runtime_configuration as validate)


def setup_log_capture(caplog: LogCaptureFixture, level=logging.INFO):
    """
    Setup log capture for the test.
    """
    caplog.set_level(level)
    if caplog.handler not in runtime_config_validator.logger.handlers:
        runtime_config_validator.logger.addHandler(caplog.handler)


@pytest.mark.utils
@pytest.mark.cpu
def test_no_eager_validation(monkeypatch, caplog):
    """
    Ensure that model runtime config validation is skipped when not running on
    Spyre cards.
    """
    setup_log_capture(caplog, level=logging.INFO)
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "eager")
        validate("test/model")
        assert "validation bypassed" in caplog.text


@pytest.mark.utils
@pytest.mark.cpu
def test_model_not_supported(monkeypatch, caplog):
    """
    Ensure we can run model runtime config validation when (pretending to) run
    on Spyre cards.
    """
    setup_log_capture(caplog, level=logging.INFO)
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        validate("test/model")
        assert "Model 'test/model' is not supported" in caplog.text


@pytest.mark.utils
@pytest.mark.cpu
def test_exit_on_validation_error(monkeypatch, caplog):
    """
    If VLLM_SPYRE_EXIT_ON_UNSUPPORTED_RUNTIME_CONFIG is set, then model runtime
    configuration validation should exit with a ValueError.
    """
    setup_log_capture(caplog, level=logging.INFO)
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        m.setenv("VLLM_SPYRE_EXIT_ON_UNSUPPORTED_RUNTIME_CONFIG", "1")
        with pytest.raises(ValueError) as error:
            validate("test/model")
        assert "Model 'test/model' is not supported" in str(error.value)


@pytest.mark.utils
@pytest.mark.cpu
def test_model_runtime_configurations_file_is_valid(monkeypatch, caplog):
    """
    Validate that prompts are multiples of 64
    Validate that prompt + new_tokens <= max_model_len
    Validate that the batch size is <= a tested upper bound.
    """
    setup_log_capture(caplog, level=logging.INFO)
    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        validate("test/model")  # ensure configs got loaded
        mrcs = runtime_config_validator.model_runtime_configs
        assert len(mrcs) > 0
        for mrc in mrcs:
            for c in mrc.configs:
                assert c.tp_size in [1, 2, 4, 8, 16, 32]
                if c.cb:
                    assert c.warmup_shapes is None
                    assert c.max_model_len % 64 == 0
                    assert c.max_model_len <= 32 * 1024
                    assert c.max_num_seqs <= 16
                else:
                    assert c.warmup_shapes is not None
                    for ws in c.warmup_shapes:
                        assert ws[0] % 64 == 0
                        assert ws[0] <= 32 * 1024
                        assert ws[2] in [1, 2, 4, 8, 16, 32]


@pytest.mark.utils
@pytest.mark.cpu
def test_model_runtime_configurations(monkeypatch, caplog):
    """
    Verify that various example model runtime configurations can get validated
    against a small list of sample configurations.
    """
    test_configs = yaml.safe_load("""
    - model: "test/model"
      configs: [
        { cb: False, tp_size: 1, warmup_shapes: [[64, 20, 4], [128, 20, 2]] },
        { cb: False, tp_size: 1, warmup_shapes: [[256, 20, 1]] },
        { cb: False, tp_size: 2, warmup_shapes: [[64, 20, 4]] },
        { cb: True,  tp_size: 1, max_model_len: 1024, max_num_seqs: 16 },
        { cb: True,  tp_size: 4, max_model_len: 2048, max_num_seqs: 8 },
        { cb: True,  tp_size: 4, max_model_len: 4096, max_num_seqs: 4 },
        { cb: True,  tp_size: 4, max_model_len: 8192, max_num_seqs: 2 },
      ]
    """)
    runtime_config_validator.initialize_supported_configurations(test_configs)

    setup_log_capture(caplog, level=logging.INFO)

    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        m.setenv("VLLM_SPYRE_USE_CB", "1")
        assert validate("test/model", 4, 2048, 8)
        assert not validate("model/test", 4, 2048, 8)
        assert not validate("test/model", 4, 2049, 8)

    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        m.setenv("VLLM_SPYRE_USE_CB", "0")
        assert validate("test/model", 1, warmup_shapes=[[64, 20, 4]])
        assert validate("test/model", 1, warmup_shapes=[[128, 20, 2]])
        assert validate("test/model",
                        1,
                        warmup_shapes=[[64, 20, 4], [128, 20, 2]])
        assert validate("test/model",
                        1,
                        warmup_shapes=[[128, 20, 2], [64, 20, 4]])
        assert not validate("test/model",
                            1,
                            warmup_shapes=[[64, 20, 4], [128, 20, 2],
                                           [256, 20, 1]])

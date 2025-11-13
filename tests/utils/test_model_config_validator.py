import logging
from os import environ as env
from pathlib import Path

import pytest
import yaml
from pytest import LogCaptureFixture
from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from vllm.config import ModelConfig

from vllm_spyre import envs as envs_spyre
from vllm_spyre.config import runtime_config_validator
from vllm_spyre.config.runtime_config_validator import (
    find_known_models_by_model_config,
    get_supported_models_list,
)
from vllm_spyre.config.runtime_config_validator import validate_runtime_configuration as validate


class TestModelConfig(ModelConfig):
    def __init__(self, model: str, hf_config: PretrainedConfig = None):
        self.model = model
        self.hf_config = hf_config

    def __post_init__(self):
        pass

    def __repr__(self):
        return f"TestModelConfig({self.model})"


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
        validate(TestModelConfig("test/model"))
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
        validate(TestModelConfig("test/model"))
        assert "Found no matching model" in caplog.text


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
        validate(TestModelConfig("test/model"))  # ensure configs got loaded
        mrcs = runtime_config_validator.model_runtime_configs
        assert len(mrcs) > 0
        for mrc in mrcs:
            for c in mrc.configs:
                assert c.tp_size in [1, 2, 4, 8, 16, 32]
                if c.cb:
                    assert c.warmup_shapes is None
                    assert c.max_model_len % 64 == 0
                    assert c.max_model_len <= 32 * 1024
                    assert c.max_num_seqs <= 32
                else:
                    assert c.warmup_shapes is not None
                    for ws in c.warmup_shapes:
                        assert ws[0] % 64 == 0
                        assert ws[0] <= 32 * 1024
                        assert ws[2] in [1, 2, 4, 8, 16, 32, 64]


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
        { cb: True,  tp_size: 1, max_model_len: 1024, max_num_seqs: 16 },
        { cb: True,  tp_size: 4, max_model_len: 2048, max_num_seqs: 8 },
        { cb: True,  tp_size: 4, max_model_len: 4096, max_num_seqs: 4 },
        { cb: True,  tp_size: 4, max_model_len: 8192, max_num_seqs: 2 },
        { cb: False, tp_size: 1, warmup_shapes: [[64, 20, 4], [128, 20, 2]] },
        { cb: False, tp_size: 1, warmup_shapes: [[256, 20, 1]] },
        { cb: False, tp_size: 2, warmup_shapes: [[64, 20, 4]] },
      ]
    """)
    runtime_config_validator.initialize_supported_configurations(test_configs)

    setup_log_capture(caplog, level=logging.INFO)

    model = TestModelConfig("test/model")
    unknown_model = TestModelConfig("model/test")

    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")

        # continuous batching validations
        m.setenv("VLLM_SPYRE_USE_CB", "1")

        assert validate(model, 4, 2048, 8)
        assert not validate(unknown_model, 4, 2048, 8)
        # validate that individual values of a requested config can be less than
        # the upper bound of a supported config
        assert validate(model, 4, 1024, 8)
        assert validate(model, 4, 2048, 4)
        # validate that config parameters adhere to restrictions (n*64, 2^n)
        assert not validate(model, 3, 1024, 8)
        assert not validate(model, 4, 2047, 4)
        assert not validate(model, 4, 2048, 3)

        # static batching validations
        envs_spyre.override("VLLM_SPYRE_USE_CB", "0")

        assert validate(model, 1, warmup_shapes=[[64, 20, 4]])
        assert validate(model, 1, warmup_shapes=[[128, 20, 2]])
        # validate that warmup_shape values adhere to restrictions (n*64, 2^n)
        assert not validate(model, 1, warmup_shapes=[[63, 20, 4]])
        assert not validate(model, 1, warmup_shapes=[[64, 20, 3]])
        # validate that the sequence of warmup_shapes does not matter
        assert validate(model, 1, warmup_shapes=[[64, 20, 4], [128, 20, 2]])
        assert validate(model, 1, warmup_shapes=[[128, 20, 2], [64, 20, 4]])
        # validate that individual components of a warmup_shapes can be less
        # than the upper bound of a supported config
        assert validate(model, 1, warmup_shapes=[[128, 19, 2]])
        assert validate(model, 2, warmup_shapes=[[64, 19, 4]])
        assert validate(model, 2, warmup_shapes=[[64, 19, 2]])
        # validate that config parameters do not exceed upper bounds
        assert not validate(model, 1, warmup_shapes=[[128, 20, 4]])
        assert not validate(model, 2, warmup_shapes=[[64, 20, 4], [128, 20, 2]])
        assert not validate(model, 1, warmup_shapes=[[64, 20, 4], [128, 20, 2], [256, 20, 1]])

    # restore default configs for following tests
    runtime_config_validator.initialize_supported_configurations_from_file()


@pytest.mark.utils
@pytest.mark.cpu
def test_find_model_by_config(monkeypatch, caplog):
    """
    Verify that we can find models by ModelConfigs loaded from config files.
    This is important for the case where models are mounted to the local file
    system instead of being loaded/cached from HuggingFace.
    """
    model_configs_dir = Path(__file__).parent.parent / "fixtures" / "model_configs"

    setup_log_capture(caplog, level=logging.INFO)

    with monkeypatch.context() as m:
        m.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
        # m.setenv("HF_HUB_OFFLINE", "1")

        for model_id in get_supported_models_list():
            model_config_dir = model_configs_dir / model_id
            model_config_file = model_config_dir / "config.json"

            assert model_config_file.exists(), (
                f"Missing config file for model {model_id}."
                f" Use download_model_configs.py to download it."
            )

            if env.get("HF_HUB_OFFLINE", "0") == "0":
                # it takes up to 3 sec per model to load config from HF:
                #   vllm.config.ModelConfig.__post_init__():
                #     model_info, arch = self.registry.inspect_model_cls(...)
                model_config = ModelConfig(model=str(model_config_dir))
            else:
                hf_config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=model_config_file, local_files_only=True
                )
                model_config = TestModelConfig(model=str(model_config_dir), hf_config=hf_config)

            assert model_config.model != model_id

            models_found = find_known_models_by_model_config(model_config)
            assert len(models_found) > 0, (
                f"Could not find any known models that match the ModelConfig"
                f" for model `{model_id}`. Update the entry for `{model_id}`"
                f" in `vllm_spyre/config/known_model_configs.json` so that its"
                f" parameters are a subset of those in `{model_config_file}`."
            )
            assert len(models_found) < 2, (
                f"More than one model found. Add more distinguishing"
                f" parameters for models `{models_found}` in file"
                f" `vllm_spyre/config/known_model_configs.json`!"
            )
            assert models_found[0] == model_id

            validate(model_config)
            assert f"Model '{model_config.model}' is not a known" in caplog.text
            assert f"Found model '{model_id}'" in caplog.text

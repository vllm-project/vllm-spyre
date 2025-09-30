from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import init_logger

from vllm_spyre import envs as envs_spyre

_config_file = Path(__file__).parent / "supported_configurations.yaml"

logger = init_logger(__name__)

WarmupShapes = list[tuple[int, int, int]] | list[list[int]]


@dataclass(order=True)
class RuntimeConfiguration:
    cb: bool = False
    tp_size: int = 1
    max_model_len: int = 0
    max_num_seqs: int = 0
    warmup_shapes: WarmupShapes | None = field(compare=False, default=None)

    def __post_init__(self):
        if self.warmup_shapes is not None:
            self.warmup_shapes = [(ws[0], ws[1], ws[2])
                                  if isinstance(ws, list) else ws
                                  for ws in self.warmup_shapes]  # yapf: disable


@dataclass
class ModelRuntimeConfiguration:
    model: str
    configs: list[RuntimeConfiguration] | None = None
    ignore: bool = False

    def __post_init__(self):
        self.configs = [
            RuntimeConfiguration(**cfg) if isinstance(cfg, dict) else cfg
            for cfg in self.configs or []
        ]


model_runtime_configs: list[ModelRuntimeConfiguration] | None = None
ignored_models: set[str] = set()
runtime_configs_by_model: dict[str, list[RuntimeConfiguration]]


def load_config_yaml() -> list[dict[str, Any]]:
    with open(_config_file, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def initialize_supported_configurations(yaml_data: list[dict[str, Any]]):
    global model_runtime_configs, ignored_models, runtime_configs_by_model
    model_runtime_configs = [
        ModelRuntimeConfiguration(**config_dict) for config_dict in yaml_data
    ]
    ignored_models = {mrc.model for mrc in model_runtime_configs if mrc.ignore}
    runtime_configs_by_model = {
        mrc.model: mrc.configs or []
        for mrc in model_runtime_configs if not mrc.ignore
    }


def initialize_supported_configurations_from_file():
    yaml_data = load_config_yaml()
    initialize_supported_configurations(yaml_data)


def report_error(msg: str):
    if envs_spyre.VLLM_SPYRE_EXIT_ON_UNSUPPORTED_RUNTIME_CONFIG:
        raise ValueError(msg)
    else:
        logger.warning(msg)


def validate_runtime_configuration(
        model: str,
        tp_size: int = 0,
        max_model_len: int = 0,
        max_num_seqs: int = 0,
        warmup_shapes: WarmupShapes | None = None) -> bool:
    # we only validate runtime configurations when running on Spyre cards
    if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn":
        logger.info(
            "Model and runtime configuration validation bypassed for"
            " backend '%s'", envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND)
        return True

    global model_runtime_configs
    if model_runtime_configs is None:
        initialize_supported_configurations_from_file()

    if model in ignored_models:
        logger.info("Model '%s' is ignored", model)
        return True

    if model not in runtime_configs_by_model:
        report_error(f"Model '{model}' is not supported")
        return False

    use_cb = envs_spyre.VLLM_SPYRE_USE_CB

    requested_config = RuntimeConfiguration(
        cb=use_cb,
        tp_size=tp_size,
        max_model_len=max_model_len if use_cb else 0,
        max_num_seqs=max_num_seqs if use_cb else 0,
        warmup_shapes=warmup_shapes if not use_cb else None)

    supported_configs = runtime_configs_by_model.get(model, [])

    # Don't use `if requested_configuration not in supported_configurations:...`
    # since warmup shapes don't compare easily, exclude from dataclass __eq__
    # Instead, use filter here and do a set-compare for warmup_shapes separately
    matching_configs: list[RuntimeConfiguration] = list(
        filter(
            lambda supported_config: is_requested_config_supported(
                requested_config=requested_config,
                supported_config=supported_config),
            supported_configs,
        ))

    if len(matching_configs) == 0:
        report_error(f"The requested configuration is not supported for"
                     f" model '{model}': {str(requested_config)}")
        return False
    else:
        logger.info(
            "The requested configuration is supported for"
            " model '%s': %s", model, str(requested_config))
        return True


def is_requested_config_supported(
        requested_config: RuntimeConfiguration,
        supported_config: RuntimeConfiguration) -> bool:
    return requested_config <= supported_config and set(
        requested_config.warmup_shapes or []).issubset(
            set(supported_config.warmup_shapes or []))

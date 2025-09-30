from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from vllm.logger import init_logger

from vllm_spyre import envs as envs_spyre

_config_file = Path(__file__).parent / "supported_configurations.yaml"

logger = init_logger(__name__)

# warmup_shape = [prompt_length, new_tokens, batch_size]
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
    """
    Log a warning message of raise an error if the environment variable
    VLLM_SPYRE_EXIT_ON_UNSUPPORTED_RUNTIME_CONFIG is set.
    """
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
    """
    Verify if the requested model and configuration is supported by comparing
    the requested configuration to all the supported configurations.
    """
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
    """
    Check if the requested configuration is supported by comparing the requested
    configuration to all the supported configurations.
    """
    # Don't use `if requested_configuration not in supported_configurations:...`
    # since warmup shapes don't compare easily (excluded from dataclass __eq__)
    # Instead, use filter here and do a set-compare for warmup_shapes separately
    return (requested_config.cb == supported_config.cb
            and requested_config <= supported_config
            and (requested_config.cb or is_warmup_shapes_supported(
                requested_config, supported_config)))


def is_warmup_shapes_supported(requested_config: RuntimeConfiguration,
                               supported_config: RuntimeConfiguration) -> bool:
    """
    Check if the requested warmup_shapes are a subset of the supported
    warmup_shapes. If a singular warmup_shape is requested, check
    if its context length is less than or equal to the context length of a
    supported warmup_shapes with the same (or larger) batch size.
    """
    requested_shapes = set(requested_config.warmup_shapes or [])
    supported_shapes = set(supported_config.warmup_shapes or [])
    return (requested_shapes.issubset(supported_shapes)
            or is_context_length_supported(requested_shapes, supported_shapes))


def is_context_length_supported(requested_shapes: WarmupShapes,
                                supported_shapes: WarmupShapes) -> bool:
    """
    If a singular warmup_shape is requested, check if it's context length is
    less than or equal to the context length for any of the supported
    warmup_shapes with the same batch size (or larger supported batch size).
    (context length = prompt_length + new_tokens)
    """
    if len(requested_shapes) > 1:
        return False
    request_batch_size = list(requested_shapes)[0][2]
    supported_shapes_with_matching_batch_size = [
        ws for ws in supported_shapes if request_batch_size <= ws[2]
    ]
    return (
        len(supported_shapes_with_matching_batch_size) > 0 and
        (get_max_model_length(requested_shapes)
         <= get_max_model_length(supported_shapes_with_matching_batch_size)))


def get_max_model_length(warmup_shapes: WarmupShapes) -> int:
    """
    Return the maximum model length from the given warmup shapes.
    """
    # max_model_len = prompt_length + new_tokens
    # warmup_shape = [prompt_length, new_tokens, batch_size]
    return max([ws[0] + ws[1] for ws in warmup_shapes or []])

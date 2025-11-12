import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from pandas.io.json._normalize import nested_to_record as flatten
from vllm.config import ModelConfig
from vllm.logger import init_logger

from vllm_spyre import envs as envs_spyre

_supported_configs_file = Path(__file__).parent / "supported_configs.yaml"
_known_model_configs_file = Path(__file__).parent / "known_model_configs.json"

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
known_model_configs: dict[str, dict] | None = None


def load_known_model_configs_json() -> dict[str, Any]:
    with open(_known_model_configs_file, encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


def initialize_known_model_configurations_from_file():
    global known_model_configs
    known_model_configs = load_known_model_configs_json()


def load_supported_configs_yaml() -> list[dict[str, Any]]:
    with open(_supported_configs_file, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
    return yaml_data


def initialize_supported_configurations(yaml_data: list[dict[str, Any]]):
    global model_runtime_configs, ignored_models, runtime_configs_by_model
    model_runtime_configs = [ModelRuntimeConfiguration(**config_dict) for config_dict in yaml_data]
    ignored_models = {mrc.model for mrc in model_runtime_configs if mrc.ignore}
    runtime_configs_by_model = {
        mrc.model: mrc.configs or [] for mrc in model_runtime_configs if not mrc.ignore
    }


def initialize_supported_configurations_from_file():
    yaml_data = load_supported_configs_yaml()
    initialize_supported_configurations(yaml_data)


def get_supported_models_list() -> list[str]:
    global model_runtime_configs
    if model_runtime_configs is None:
        initialize_supported_configurations_from_file()
    public_models = [mrc.model for mrc in model_runtime_configs or [] if not mrc.ignore]
    return public_models


def verify_config_parameters(c: RuntimeConfiguration) -> bool:
    found_invalid_parameters = False

    def verify(msg: str, is_valid: bool):
        nonlocal found_invalid_parameters
        if not is_valid:
            found_invalid_parameters = True
            logger.warning(msg)

    def is_power_of_2(n: int) -> bool:
        return (n > 0) and (n & (n - 1) == 0)

    verify(
        f"'tensor_parallel_size' must be a power of 2, found {c.tp_size}", is_power_of_2(c.tp_size)
    )

    if c.cb:
        verify("'warmup_shapes' are not used for continuous batching", c.warmup_shapes is None)
        verify(
            f"'max_model_len' must be a multiple of 64, found {c.max_model_len}",
            c.max_model_len % 64 == 0,
        )
        verify(
            f"'max_num_seqs' must be a power of 2, found {c.max_num_seqs}",
            is_power_of_2(c.max_num_seqs),
        )
    else:
        verify(
            "at least one 'warmup_shapes' required for static batching",
            c.warmup_shapes is not None and len(c.warmup_shapes) > 0,
        )

        for i, ws in enumerate(c.warmup_shapes or []):
            # warmup_shape = [prompt_length, new_tokens, batch_size]
            verify(
                f"'prompt_length' must be a multiple of 64, found {ws[0]} in warmup_shapes[{i}]",
                ws[0] % 64 == 0,
            )
            verify(
                f"'batch_size' must be a power of 2, found {ws[2]} in warmup_shapes[{i}]",
                is_power_of_2(ws[2]),
            )

    return not found_invalid_parameters


def find_known_models_by_model_config(model_config: ModelConfig) -> list[str]:
    """
    Try to find a supported model by comparing the requested model config to
    the known model configurations. The known model configurations file only
    contains a minimal subset of model config parameters to distinguish
    between the supported models.
    """
    if known_model_configs is None:
        initialize_known_model_configurations_from_file()

    requested_config = model_config.hf_config.__dict__ if model_config.hf_config else {}

    # remove sub-dicts with integers as keys so we can flatten dictionaries
    requested_config.pop("id2label", None)

    # don't return quantized models if the requested config doesn't have it
    def is_quantized(config: dict) -> bool:
        return "quantization_config" in config

    matching_models = [
        model
        for model, config in (known_model_configs or {}).items()
        if flatten(config).items() <= flatten(requested_config).items()
        and (is_quantized(config) == is_quantized(requested_config))
    ]

    return matching_models


def validate_runtime_configuration(
    model_config: ModelConfig,
    tp_size: int = 0,
    max_model_len: int = 0,
    max_num_seqs: int = 0,
    warmup_shapes: WarmupShapes | None = None,
) -> bool:
    """
    Verify if the requested model and configuration is supported by comparing
    the requested configuration to all the supported configurations.
    """
    # we only validate runtime configurations when running on Spyre cards
    if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn":
        logger.info(
            "Model and runtime configuration validation bypassed for backend '%s'",
            envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND,
        )
        return True

    if model_runtime_configs is None:
        initialize_supported_configurations_from_file()

    model = model_config.model

    known_models = {mrc.model for mrc in model_runtime_configs or {}}

    if model not in known_models:
        logger.info(
            "Model '%s' is not a known model. Trying to find one with a matching ModelConfig.",
            model,
        )

        matching_models = find_known_models_by_model_config(model_config)

        if len(matching_models) == 1:
            model = matching_models[0]
            logger.info("Found model '%s' matching ModelConfig `%s`.", model, model_config)

        elif len(matching_models) == 0:
            logger.warning("Found no matching model for ModelConfig `%s`.", model_config)
            return False

        elif len(matching_models) > 1:
            logger.error(
                "Found several models matching the given ModelConfig."
                " Aborting model validation. vllm-Spyre developers"
                " need to update the known model configurations file"
                " to distinguish between the returned models."
                " Models found: [%s]. ModelConfig provided `%s`",
                matching_models,
                model_config,
            )
            return False

    if model in ignored_models:
        logger.info("Model '%s' is ignored", model)
        return True

    if model not in runtime_configs_by_model:
        logger.warning("Model '%s' is not supported", model)
        return False

    use_cb = envs_spyre.VLLM_SPYRE_USE_CB

    requested_config = RuntimeConfiguration(
        cb=use_cb,
        tp_size=tp_size,
        max_model_len=max_model_len if use_cb else 0,
        max_num_seqs=max_num_seqs if use_cb else 0,
        warmup_shapes=warmup_shapes if not use_cb else None,
    )

    if not verify_config_parameters(requested_config):
        return False

    supported_configs = runtime_configs_by_model.get(model, [])

    matching_configs: list[RuntimeConfiguration] = list(
        filter(
            lambda supported_config: is_requested_config_supported(
                requested_config=requested_config, supported_config=supported_config
            ),
            supported_configs,
        )
    )

    if len(matching_configs) == 0:
        logger.warning(
            "The requested configuration is not supported for model '%s': %s",
            model,
            str(requested_config),
        )
        return False
    else:
        logger.info(
            "The requested configuration is supported for model '%s': %s",
            model,
            str(requested_config),
        )
        return True


def is_requested_config_supported(
    requested_config: RuntimeConfiguration, supported_config: RuntimeConfiguration
) -> bool:
    """
    Check if the requested configuration is supported by comparing the requested
    configuration to all the supported configurations.
    """
    # Don't use `if requested_configuration not in supported_configurations:...`
    # since warmup shapes don't compare easily (excluded from dataclass __eq__)
    # Instead, use filter here and do a set-compare for warmup_shapes separately
    return (
        requested_config.cb == supported_config.cb
        and requested_config <= supported_config
        and (requested_config.cb or is_warmup_shapes_supported(requested_config, supported_config))
    )


def is_warmup_shapes_supported(
    requested_config: RuntimeConfiguration, supported_config: RuntimeConfiguration
) -> bool:
    """
    Check if the requested warmup_shapes are a subset of the supported
    warmup_shapes. If a single warmup_shape is requested, validate its context
    length.
    """
    requested_shapes = requested_config.warmup_shapes or []
    supported_shapes = supported_config.warmup_shapes or []
    return set(requested_shapes).issubset(set(supported_shapes)) or is_context_length_supported(
        requested_shapes, supported_shapes
    )


def is_context_length_supported(
    requested_shapes: WarmupShapes, supported_shapes: WarmupShapes
) -> bool:
    """
    If a single warmup_shape is requested, check if its context length is
    less than or equal to the context length of any supported warmup_shape
    with the same or larger batch size.
    (context length = prompt_length + new_tokens)
    """

    # TODO: what if more than one warmup shape is requested?
    if len(requested_shapes) > 1:
        return False

    request_batch_size = requested_shapes[0][2]
    shapes_with_same_batch_size = [
        (ws[0], ws[1], ws[2]) for ws in supported_shapes if request_batch_size <= ws[2]
    ]

    return len(shapes_with_same_batch_size) > 0 and (
        get_max_model_length(requested_shapes) <= get_max_model_length(shapes_with_same_batch_size)
    )


def get_max_model_length(warmup_shapes: WarmupShapes) -> int:
    """
    Return the maximum model length from the given warmup shapes.
    """

    # max_model_len = prompt_length + new_tokens
    # warmup_shape = [prompt_length, new_tokens, batch_size]

    return max([ws[0] + ws[1] for ws in warmup_shapes or []])

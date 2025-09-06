import platform
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import yaml
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.logger import init_logger

from vllm_spyre import envs as envs_spyre

_config_file = Path(__file__).parent / "supported_configurations.yaml"

logger = init_logger(__name__)
logger.info("Running on '%s'", platform.machine())

WarmupShapes = list[tuple[int, int, int]] | list[list[int]]


class PlatformName(Enum):
    AMD = "x86_64"
    ARM = "arm64"
    ZOS = "s390x"
    POWER = "ppc64le"


@dataclass
class RuntimeConfiguration:
    # TODO: ?platform? or use torch.device ("cpu") or DYNAMO_BACKEND instead?
    platform: PlatformName = PlatformName.AMD
    cb: bool = False
    tp_size: int = 1
    max_model_len: int = 0
    max_num_seqs: int = 0
    num_blocks: int = 0
    warmup_shapes: WarmupShapes | None = field(compare=False, default=None)

    def __post_init__(self):
        if isinstance(self.platform, str):
            self.platform = PlatformName(self.platform)
        if self.warmup_shapes is not None:
            self.warmup_shapes = [(ws[0], ws[1], ws[2])
                                  if isinstance(ws, list) else ws
                                  for ws in self.warmup_shapes]  # yapf: disable



@dataclass
class ModelRuntimeConfiguration:
    model: str
    configs: list[RuntimeConfiguration]

    def __post_init__(self):
        self.configs = [
            RuntimeConfiguration(**cfg) if isinstance(cfg, dict) else cfg
            for cfg in self.configs
        ]


model_runtime_configs: list[ModelRuntimeConfiguration] | None = None

runtime_configs_by_model: dict[str, list[RuntimeConfiguration]]


def initialize_supported_configurations_from_file():
    global model_runtime_configs, runtime_configs_by_model
    with open(_config_file, encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)
        model_runtime_configs = [
            ModelRuntimeConfiguration(**config_dict)
            for config_dict in yaml_data["runtime_configs"]
        ]
    runtime_configs_by_model = {
        mrc.model: mrc.configs
        for mrc in model_runtime_configs
    }


def get_sys_platform_name() -> PlatformName:
    machine = platform.machine()
    # TODO: remove hack: arm64 is x86_64 for local testing?
    #   should we use machine/arch, or torch.device("cpu"), or dynamo backend?
    if machine == "arm64":
        machine = "x86_64"  # "amd64"
    return PlatformName(machine)


def get_warmup_shapes_from_envs() -> WarmupShapes:
    prompt_lens = envs_spyre.VLLM_SPYRE_WARMUP_PROMPT_LENS or []
    new_tokens = envs_spyre.VLLM_SPYRE_WARMUP_NEW_TOKENS or []
    batch_sizes = envs_spyre.VLLM_SPYRE_WARMUP_BATCH_SIZES or []
    # fixed_prompt_length = warmup_shape[0] = 64
    # max_new_tokens = warmup_shape[1] = 20
    # batch_size = warmup_shape[2] = 1
    warmup_shapes = [
        (pl, nt, bs)
        for pl, nt, bs in zip(prompt_lens, new_tokens, batch_sizes)
    ]
    return warmup_shapes


def report_error(msg: str, raise_error: bool = False):
    if raise_error:
        raise ValueError(msg)
    else:
        logger.warning(msg)


def validate_runtime_configuration(model_config: ModelConfig,
                                   parallel_config: ParallelConfig,
                                   scheduler_config: SchedulerConfig,
                                   cache_config: CacheConfig,
                                   warmup_shapes: WarmupShapes | None = None,
                                   raise_error: bool = False):
    global model_runtime_configs
    if model_runtime_configs is None:
        initialize_supported_configurations_from_file()

    if model_config.model not in runtime_configs_by_model:
        report_error(f"Model {model_config.model} is not supported",
                     raise_error)

    use_cb = envs_spyre.VLLM_SPYRE_USE_CB

    # TODO: num_blocks = cpu or gpu blocks?
    requested_config = RuntimeConfiguration(
        platform=get_sys_platform_name(),
        cb=use_cb,
        tp_size=parallel_config.tensor_parallel_size,
        max_model_len=model_config.max_model_len if use_cb else 0,
        max_num_seqs=scheduler_config.max_num_seqs if use_cb else 0,
        num_blocks=cache_config.num_cpu_blocks or 0,
        warmup_shapes=warmup_shapes if not use_cb else None)

    supported_configs = runtime_configs_by_model.get(model_config.model, [])

    # Don't use `if requested_configuration not in supported_configurations:...`
    #   since warmup shapes don't compare easy, exclude from dataclass __eq__
    #   use filter and set-compare warmup_shapes separately
    matching_configs: list[RuntimeConfiguration] = (list(
        filter(lambda c: c == requested_config, supported_configs)))

    if len(matching_configs) == 0:
        report_error(
            f"The requested configuration is not supported for"
            f" model '{model_config.model}':"
            f" {str(requested_config)}", raise_error)

    if len(matching_configs) > 0 and not use_cb:
        supported_warmup_shapes = set([
            ws for config in matching_configs
            for ws in config.warmup_shapes or []
        ])

        requested_warmup_shapes = set(requested_config.warmup_shapes or [])

        if not requested_warmup_shapes.issubset(supported_warmup_shapes):
            report_error(
                f"The requested warmup_shapes are not supported"
                f" for model '{model_config.model}':"
                f" {str(list(requested_warmup_shapes))}", raise_error)

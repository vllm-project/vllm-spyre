# from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field
from pydantic.dataclasses import dataclass
from vllm.logger import init_logger

# from vllm.config.utils import config
from vllm_spyre.utils import install_patch

logger = init_logger(__name__)

if TYPE_CHECKING:
    from vllm.usage.usage_lib import SchedulerConfig, UsageContext
else:
    UsageContext = Any


def _patch_scheduler_config(mod):
    BaseSchedulerConfig: SchedulerConfig = mod.SchedulerConfig
    # create patched subclass to override Field defaults
    @dataclass
    class PatchedSchedulerConfig(BaseSchedulerConfig):
        DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 1024
        max_num_batched_tokens: int = Field(
            default=DEFAULT_MAX_NUM_BATCHED_TOKENS, ge=1)

    logger.debug("Applying patch to SchedulerConfig")
    mod.SchedulerConfig = PatchedSchedulerConfig


def _patch_arg_utils(mod):
    EngineArgs = getattr(mod, "EngineArgs", None)
    if EngineArgs is None:
        raise RuntimeError("EngineArgs has not been fully initialized")

    def _get_batch_defaults(
        cls, world_size: int
    ) -> tuple[dict[UsageContext | None, int], dict[UsageContext | None, int]]:
        from vllm.usage.usage_lib import UsageContext

        default_max_num_batched_tokens: dict[UsageContext | None, int]
        default_max_num_seqs: dict[UsageContext | None, int]

        # TODO: different defaults depending on torch backend
        default_max_num_batched_tokens = {
            UsageContext.LLM_CLASS: 1024,
            UsageContext.OPENAI_API_SERVER: 1024,
        }
        default_max_num_seqs = {
            UsageContext.LLM_CLASS: 4,
            UsageContext.OPENAI_API_SERVER: 4,
        }

        return default_max_num_batched_tokens, default_max_num_seqs

    logger.debug("Applying patch to EngineArgs")
    EngineArgs.get_batch_defaults = _get_batch_defaults


def patch_at_import():
    install_patch("vllm.config.scheduler", _patch_scheduler_config)


def patch_arg_utils():
    install_patch("vllm.engine.arg_utils", _patch_arg_utils)

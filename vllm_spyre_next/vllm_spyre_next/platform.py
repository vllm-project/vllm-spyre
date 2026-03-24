import sys
from typing import TYPE_CHECKING
from string import Template
import multiprocessing

# When running this plugin on a Mac, we assume it's for local development
# purposes. However, due to a compatibility issue with vLLM, which overrides
# the Triton module with a placeholder, vLLM may fail to load on macOS. To
# mitigate this issue, we can safely remove the Triton module (if imported)
# and rely on PyTorch to handle the absence of Triton, ensuring fine execution
# in eager mode.
if sys.platform.startswith("darwin"):
    if sys.modules.get("triton"):
        del sys.modules["triton"]

from vllm.logger import init_logger
from vllm.platforms import PlatformEnum
from vllm.platforms.cpu import CpuPlatform

if TYPE_CHECKING:
    # NB: We can't eagerly import many things from vllm since vllm.config
    # will import this file. These would lead to circular imports
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class TorchSpyrePlatform(CpuPlatform):
    _enum = PlatformEnum.OOT

    # "spyre" device_name no longer worked due to https://github.com/vllm-project/vllm/pull/16464
    device_name: str = "cpu"
    device_type: str = "cpu"

    # Custom ops dispatch key. Currently CPU because:
    # - Integer tensors (input_ids, positions) stay on CPU (Spyre int64
    #   copy path is broken)
    # - In eager mode, all data flows on CPU (fallback ops, RMSNorm,
    #   SiluAndMul all execute on CPU)
    # When Spyre compilation is enabled and tensors flow on device,
    # this should change to "PrivateUse1".
    dispatch_key: str = "CPU"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "torch-spyre"

    @classmethod
    def log_server_boot(cls, vllm_config: VllmConfig) -> None:
        # Only log in main process (not in TP workers)
        if multiprocessing.current_process().name != "MainProcess":
            return

        # yapf: disable
        logo_template = Template(
            template="\n       ${w}█     █     █▄   ▄█${r}       ${red}▄█▀▀█▄${r}  ${orange}█▀▀▀█▄${r}  ${yellow}█   █${r}  ${green}█▀▀▀█▄${r}  ${blue}█▀▀▀▀${r}\n" # noqa: E501
            " ${o}▄▄${r} ${b}▄█${r} ${w}█     █     █ ▀▄▀ █${r}       ${red}▀▀▄▄▄${r}   ${orange}█▄▄▄█▀${r}  ${yellow}▀▄ ▄▀${r}  ${green}█▄▄▄█▀${r}  ${blue}█▄▄▄${r}   version ${w}%s${r}\n" # noqa: E501
            "  ${o}█${r}${b}▄█▀${r} ${w}█     █     █     █${r}            ${red}█${r}  ${orange}█${r}        ${yellow}▀█▀${r}   ${green}█ ▀█▄${r}   ${blue}█${r}      model   ${w}%s${r}\n" # noqa: E501
            "   ${b}▀▀${r}  ${w}▀▀▀▀▀ ▀▀▀▀▀ ▀     ▀${r}       ${red}▀▄▄▄█▀${r}  ${orange}█${r}         ${yellow}█${r}    ${green}█   ▀█${r}  ${blue}█▄▄▄▄${r}\n" # noqa: E501
        )
        # yapf: enable
        colors = {
            "w": "\033[97;1m",  # white
            "o": "\033[93m",  # orange
            "b": "\033[94m",  # blue
            "r": "\033[0m",  # reset
            "red": "\033[91m",  # red (rainbow start)
            "orange": "\033[38;5;208m",  # orange
            "yellow": "\033[93m",  # yellow
            "green": "\033[92m",  # green
            "blue": "\033[94m",  # blue (rainbow end)
        }

        message = logo_template.substitute(colors)

        from vllm_spyre_next import _version

        model_name = vllm_config.model_config.model

        logger.info(message, _version.version, model_name)

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend,
        attn_selector_config,
        num_heads=None,
    ) -> str:
        """Use Spyre attention backend that wraps CPU attention with
        Spyre<->CPU device transfers."""
        return ("vllm_spyre_next.attention_backend"
                ".SpyreCPUAttentionBackend")

    @classmethod
    def apply_config_platform_defaults(cls, vllm_config: VllmConfig) -> None:
        """Set Spyre-specific config defaults before vLLM's defaulting logic.

        Called early in VllmConfig.__post_init__, BEFORE compilation mode
        and custom_ops defaults are applied. Setting CompilationMode.NONE here
        ensures:
        - vLLM's @support_torch_compile won't activate (avoids dynamic shapes
          / SymInt which Spyre can't handle)
        - custom_ops defaults to "all" (since mode == NONE), enabling
          forward_oot dispatch for OOT-registered layers
        Spyre compilation is handled separately in _compile_for_spyre().
        """
        from vllm.config import CompilationMode
        vllm_config.compilation_config.mode = CompilationMode.NONE

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cls.log_server_boot(vllm_config)

        # ---- worker ----
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            worker_class = "vllm_spyre_next.v1.worker.spyre_worker.TorchSpyreWorker"
            logger.info("Loading worker from: %s", worker_class)
            parallel_config.worker_cls = worker_class

        # ---- scheduler ----
        scheduler_config = vllm_config.scheduler_config
        scheduler_class = "vllm.v1.core.sched.scheduler.Scheduler"
        logger.info("Loading scheduler from: %s", scheduler_class)
        scheduler_config.scheduler_cls = scheduler_class

        # call CpuPlatform.check_and_update_config()
        super().check_and_update_config(vllm_config)

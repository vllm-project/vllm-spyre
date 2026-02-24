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
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        cls.log_server_boot(vllm_config)

        # ---- worker ----
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            # "auto" defaults to the CPUWorker as we inherit from the CpuPlatform
            worker_class = "vllm.v1.worker.cpu_worker.CPUWorker"
            # if a torch spyre specific worker class is needed it can be loaded with
            # worker_class = "vllm_spyre_next.v1.worker.spyre_worker.TorchSpyreWorker"
            logger.info("Loading worker from: %s", worker_class)
            parallel_config.worker_cls = worker_class

        # ---- model runner ----
        # A custom model runner has to be added to a potential TorchSpyreWorker class:
        # TorchSpyreWorker.model_runner = TorchSpyreModelRunner (see SpyreWorker for reference)
        # The default vllm.v1.worker.cpu_worker.CPUWorker uses
        # vllm.v1.worker.cpu_model_runner.CPUModelRunner

        # ---- scheduler ----
        scheduler_config = vllm_config.scheduler_config
        # default scheduler
        scheduler_class = "vllm.v1.core.sched.scheduler.Scheduler"
        # if a torch spyre specific scheduler class is needed it can be loaded with
        # scheduler_class = "vllm_spyre_next.v1.core.scheduler.TorchSpyreScheduler"
        logger.info("Loading scheduler from: %s", scheduler_class)
        scheduler_config.scheduler_cls = scheduler_class

        # ---- attention backend ----
        # A custom attention backend can be registered with get_attn_backend_cls()
        # see copied code from vllm/platforms/cpu.CpuPlatform illustrating the default
        # TorchSDPABackend used for vLLM CPU execution

        # @classmethod
        # def get_attn_backend_cls(cls, selected_backend: _Backend, head_size: int,
        #                      dtype: torch.dtype, kv_cache_dtype: Optional[str],
        #                      block_size: int, use_v1: bool,
        #                      use_mla: bool) -> str:
        #     if selected_backend and selected_backend != _Backend.TORCH_SDPA:
        #         logger.info("Cannot use %s backend on CPU.", selected_backend)
        #     logger.info("Using Torch SDPA backend.")
        #     return "vllm.attention.backends.torch_sdpa.TorchSDPABackend"

        # call CpuPlatform.check_and_update_config()
        super().check_and_update_config(vllm_config)

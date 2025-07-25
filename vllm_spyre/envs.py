import os
from typing import TYPE_CHECKING, Any, Callable, Optional

from vllm.logger import init_logger

if TYPE_CHECKING:
    VLLM_SPYRE_DYNAMO_BACKEND: str = "sendnn"
    VLLM_SPYRE_WARMUP_PROMPT_LENS: Optional[list[int]] = None
    VLLM_SPYRE_WARMUP_NEW_TOKENS: Optional[list[int]] = None
    VLLM_SPYRE_WARMUP_BATCH_SIZES: Optional[list[int]] = None
    VLLM_SPYRE_USE_CB: bool = False
    VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED: int = 0
    VLLM_SPYRE_PERF_METRIC_LOGGING_DIR: str = "/tmp"
    VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER: bool = False
    # Prompt logprobs are behind a flag because they're only supported for
    # static batching and require passing back the hidden states for the full
    # prefill on every request. This could incur a heavy performance penalty in
    # many cases, so it should only be enabled when prompt_logprobs are required
    # for experimentation purposes.
    VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS: bool = False

logger = init_logger(__name__)


def _backend_backwards_compat() -> str:
    val = os.getenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
    if val == "sendnn_decoder":
        logger.warning_once(
            "Using 'sendnn_decoder' for "
            "VLLM_SPYRE_DYNAMO_BACKEND is deprecated. Use 'sendnn' instead")
        val = 'sendnn'
    return val


# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Defines the prompt lengths the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in static batching
    # mode (VLLM_SPYRE_USE_CB=0).
    "VLLM_SPYRE_WARMUP_PROMPT_LENS":
    lambda: [
        int(p) for p in os.getenv(key='VLLM_SPYRE_WARMUP_PROMPT_LENS',
                                  default='64').split(',')
    ],
    # Defines the max output tokens the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in static batching
    # mode (VLLM_SPYRE_USE_CB=0).
    "VLLM_SPYRE_WARMUP_NEW_TOKENS":
    lambda: [
        int(d) for d in os.getenv(key='VLLM_SPYRE_WARMUP_NEW_TOKENS',
                                  default='20').split(',')
    ],
    # Defines the batch sizes the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in static batching
    # mode (VLLM_SPYRE_USE_CB=0).
    "VLLM_SPYRE_WARMUP_BATCH_SIZES":
    lambda: [
        int(b) for b in os.getenv(key='VLLM_SPYRE_WARMUP_BATCH_SIZES',
                                  default='1').split(',')
    ],

    # Defines the backend that torch.compile will use when using Spyre
    # Available options:
    # - "sendnn": Compile for execution on Spyre hardware
    # - "inductor": Compile for execution on CPU (for debug and testing)
    # - "eager": Skip compile entirely (for debug and testing)
    #
    # - "sendnn_decoder": Deprecated in favor of "sendnn"
    "VLLM_SPYRE_DYNAMO_BACKEND":
    _backend_backwards_compat,

    # If set, use the V1 continuous batching implementation. Otherwise, static
    # batching mode will be enabled.
    "VLLM_SPYRE_USE_CB":
    lambda: bool(int(os.getenv("VLLM_SPYRE_USE_CB", "0"))),

    # Enable performance metric logging. This captures startup information
    # such as warmup times, and loading times. It is turned off by default.
    "VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED":
    lambda: int(os.getenv("VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED", 0)),

    # Directory to write performance metric logging files. By default,
    # logs are written to /tmp.
    "VLLM_SPYRE_PERF_METRIC_LOGGING_DIR":
    lambda: os.getenv("VLLM_SPYRE_PERF_METRIC_LOGGING_DIR", "/tmp"),

    # If set, override the signal handler for vllm-spyre on
    # vLLM V1 + torch_sendnn backend to be able to gracefully
    # shutdown the engine.
    "VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER":
    lambda: bool(int(os.getenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1"))),

    # If set, enables the `prompt_logprobs` sampling parameter.
    # By default, prompt_logprobs aren't supported
    "VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS":
    lambda: bool(int(os.getenv("VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS", "0"))),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

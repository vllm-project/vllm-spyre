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
    VLLM_SPYRE_UPDATE_THREAD_CONFIG: bool = True
    VLLM_SPYRE_MAX_LOAD_PROCESSES: int = 0
    VLLM_SPYRE_WORKER_LOG_REDIRECT_DIR: str = ""
    VLLM_SPYRE_GLOO_TIMEOUT_MINUTES: int = 60

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

    # If set, enables the joining of a new sequence even if its prompt length
    # is exceeding the tkv of the current decode batch. As this shifts all the
    # sequences in the decode batch to the right (increasing the tkv), there is
    # also a potential performance decrease coming with this. The switch allows
    # to test the feature on realistic workloads before enabling it by default.
    "VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION":
    lambda: bool(int(os.getenv("VLLM_SPYRE_ENABLE_PREFILL_OPTIMIZATION", "0"))
                 ),

    # scheduling heuristic: prefill vs decode prioritization
    # Prefills using up to VLLM_SPYRE_N_TOKENS_PREFILL_PRIO tokens will always
    # be prioritized. If limit is exceeded, decodes are prioritized.
    "VLLM_SPYRE_N_TOKENS_PREFILL_PRIO":
    lambda: int(os.getenv("VLLM_SPYRE_N_TOKENS_PREFILL_PRIO", "-1")),

    # scheduling heuristic: maximal waiting (blocking) time for prefill
    # Prefills waiting longer than VLLM_SPYRE_MAX_WAITING_TIME_SECONDS
    # seconds will have priority after the current decode batch has finished.
    "VLLM_SPYRE_MAX_WAITING_TIME_SECONDS":
    lambda: float(os.getenv("VLLM_SPYRE_MAX_WAITING_TIME_SECONDS", "inf")),

    # Allow vllm-spyre to update env vars related to multi-threading (eg. OMP)
    # based on the detected CPU cores and server configuration
    "VLLM_SPYRE_UPDATE_THREAD_CONFIG":
    lambda: bool(int(os.getenv("VLLM_SPYRE_UPDATE_THREAD_CONFIG", "1"))),

    # If set, limit the number of concurrent processes loading/compiling
    # large models or models with larger context lengths to limit
    # memory usage.
    # Set to 0 to allow any number of processes
    "VLLM_SPYRE_MAX_LOAD_PROCESSES":
    lambda: int(os.getenv("VLLM_SPYRE_MAX_LOAD_PROCESSES", "0")),

    # If set, redirects all stdout and stderr from worker processes to files
    # within this director. This is useful for debugging card-specific errors
    # in multi-AIU setups, but should never be enabled in production settings.
    # This removes all output from stdout and stderr for the worker processes.
    "VLLM_SPYRE_WORKER_LOG_REDIRECT_DIR":
    lambda: os.getenv("VLLM_SPYRE_WORKER_LOG_REDIRECT_DIR", ""),

    # If set, overrides the default (30 minutes) timeout for
    #  torch.distributed.init_process_group
    "VLLM_SPYRE_GLOO_TIMEOUT_MINUTES":
    lambda: int(os.getenv("VLLM_SPYRE_GLOO_TIMEOUT_MINUTES", "60"))
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

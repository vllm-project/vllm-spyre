import os
from typing import TYPE_CHECKING, Any, Callable

from vllm.logger import init_logger

if TYPE_CHECKING:
    VLLM_SPYRE_DYNAMO_BACKEND: str = "sendnn"
    VLLM_SPYRE_WARMUP_PROMPT_LENS: list[int] | None = None
    VLLM_SPYRE_WARMUP_NEW_TOKENS: list[int] | None = None
    VLLM_SPYRE_WARMUP_BATCH_SIZES: list[int] | None = None
    VLLM_SPYRE_USE_CB: bool = False
    VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED: int = 0
    VLLM_SPYRE_PERF_METRIC_LOGGING_DIR: str = "/tmp"
    VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER: bool = False
    VLLM_SPYRE_USE_CHUNKED_PREFILL: bool = False
    VLLM_SPYRE_CP_INTERLEAVE_STEPS: bool = True
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
    VLLM_SPYRE_REQUIRE_PRECOMPILED_DECODERS: bool = False
    VLLM_SPYRE_SIMPLE_COMPILE_BACKEND: str = "inductor"
    VLLM_SPYRE_NUM_CPUS: int = 0

logger = init_logger(__name__)

_cache: dict[str, Any] = {}


def override(name: str, value: str) -> None:
    if name not in environment_variables:
        raise ValueError(f"The variable {name} is not a known setting and cannot be overridden")
    os.environ[name] = value
    _cache[name] = environment_variables[name]()


def clear_env_cache():
    _cache.clear()


def _backend_backwards_compat() -> str:
    val = os.getenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")
    if val == "sendnn_decoder":
        logger.warning_once(
            "Using 'sendnn_decoder' for "
            "VLLM_SPYRE_DYNAMO_BACKEND is deprecated. Use 'sendnn' instead"
        )
        val = "sendnn"
    return val


# --8<-- [start:env-vars-definition]
environment_variables: dict[str, Callable[[], Any]] = {
    # Defines the prompt lengths the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in static batching
    # mode (VLLM_SPYRE_USE_CB=0).
    "VLLM_SPYRE_WARMUP_PROMPT_LENS": lambda: [
        int(p) for p in os.getenv(key="VLLM_SPYRE_WARMUP_PROMPT_LENS", default="64").split(",")
    ],
    # Defines the max output tokens the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in static batching
    # mode (VLLM_SPYRE_USE_CB=0).
    "VLLM_SPYRE_WARMUP_NEW_TOKENS": lambda: [
        int(d) for d in os.getenv(key="VLLM_SPYRE_WARMUP_NEW_TOKENS", default="20").split(",")
    ],
    # Defines the batch sizes the Spyre accelerator should be prepared
    # for, formatted as comma separated list. Only applicable in static batching
    # mode (VLLM_SPYRE_USE_CB=0).
    "VLLM_SPYRE_WARMUP_BATCH_SIZES": lambda: [
        int(b) for b in os.getenv(key="VLLM_SPYRE_WARMUP_BATCH_SIZES", default="1").split(",")
    ],
    # Defines the backend that torch.compile will use when using Spyre
    # Available options:
    # - "sendnn": Compile for execution on Spyre hardware
    # - "inductor": Compile for execution on CPU (for debug and testing)
    # - "eager": Skip compile entirely (for debug and testing)
    #
    # - "sendnn_decoder": Deprecated in favor of "sendnn"
    "VLLM_SPYRE_DYNAMO_BACKEND": _backend_backwards_compat,
    # If set, use the V1 continuous batching implementation. Otherwise, static
    # batching mode will be enabled.
    "VLLM_SPYRE_USE_CB": lambda: bool(int(os.getenv("VLLM_SPYRE_USE_CB", "0"))),
    # Enable performance metric logging. This captures startup information
    # such as warmup times, and loading times.
    # When `--disable-log-stats=False` is used, this will log timing metrics
    # about every finished request into a .jsonl file. These are the same
    # metrics that are available in prometheus format on the /metrics endpoint,
    # but it is sometime helpful to view them disaggregated to debug performance
    # problems. This logging is not designed to be performant, and should not be
    # enabled in production settings.
    # It is turned off by default.
    "VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED": lambda: int(
        os.getenv("VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED", 0)
    ),
    # Directory to write performance metric logging files. By default,
    # logs are written to /tmp.
    "VLLM_SPYRE_PERF_METRIC_LOGGING_DIR": lambda: os.getenv(
        "VLLM_SPYRE_PERF_METRIC_LOGGING_DIR", "/tmp"
    ),
    # If set, override the signal handler for vllm-spyre on
    # vLLM V1 + torch_sendnn backend to be able to gracefully
    # shutdown the engine.
    "VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER": lambda: bool(
        int(os.getenv("VLLM_SPYRE_OVERRIDE_SIGNALS_HANDLER", "1"))
    ),
    # If set, enables the `prompt_logprobs` sampling parameter.
    # Currently, prompt_logprobs aren't supported
    "VLLM_SPYRE_ENABLE_PROMPT_LOGPROBS": lambda: False,
    # Allow vllm-spyre to update env vars related to multi-threading (eg. OMP)
    # based on the detected CPU cores and server configuration
    "VLLM_SPYRE_UPDATE_THREAD_CONFIG": lambda: bool(
        int(os.getenv("VLLM_SPYRE_UPDATE_THREAD_CONFIG", "1"))
    ),
    # If set, limit the number of concurrent processes loading/compiling
    # large models or models with larger context lengths to limit
    # memory usage.
    # Set to 0 to allow any number of processes
    "VLLM_SPYRE_MAX_LOAD_PROCESSES": lambda: int(os.getenv("VLLM_SPYRE_MAX_LOAD_PROCESSES", "0")),
    # If set, redirects all stdout and stderr from worker processes to files
    # within this director. This is useful for debugging card-specific errors
    # in multi-AIU setups, but should never be enabled in production settings.
    # This removes all output from stdout and stderr for the worker processes.
    "VLLM_SPYRE_WORKER_LOG_REDIRECT_DIR": lambda: os.getenv(
        "VLLM_SPYRE_WORKER_LOG_REDIRECT_DIR", ""
    ),
    # If set, overrides the default (30 minutes) timeout for
    #  torch.distributed.init_process_group
    "VLLM_SPYRE_GLOO_TIMEOUT_MINUTES": lambda: int(
        os.getenv("VLLM_SPYRE_GLOO_TIMEOUT_MINUTES", "60")
    ),
    # If set, this will require use of pre-compiled models and
    # disable compilation for decoders
    "VLLM_SPYRE_REQUIRE_PRECOMPILED_DECODERS": lambda: bool(
        int(os.getenv("VLLM_SPYRE_REQUIRE_PRECOMPILED_DECODERS", "0"))
    ),
    # Simple compile backend for some dynamically compiled operations, like
    # gathering logprobs in the sampler.
    # Defaults to eager, iductor can be used if python headers and a compiler
    # are available.
    "VLLM_SPYRE_SIMPLE_COMPILE_BACKEND": lambda: os.getenv(
        "VLLM_SPYRE_SIMPLE_COMPILE_BACKEND", "inductor"
    ),
    # Configures the number of CPUs used when determining multi-threading
    # configurations
    # Set to 0 to have vllm-spyre attempt to detect the CPU count
    "VLLM_SPYRE_NUM_CPUS": lambda: int(os.getenv("VLLM_SPYRE_NUM_CPUS", "0")),
    # Feature Flag
    # If set, use the V1 chunked prefill implementation. Otherwise, normal
    # single prefill is used.
    "VLLM_SPYRE_USE_CHUNKED_PREFILL": lambda: bool(
        int(os.getenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "0"))
    ),
    # Feature Flag
    # Works only with chunked prefill enabled. If set, prefill steps are
    # interleaved with a decode step
    "VLLM_SPYRE_CP_INTERLEAVE_STEPS": lambda: bool(
        int(os.getenv("VLLM_SPYRE_CP_INTERLEAVE_STEPS", "1"))
    ),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    if name in _cache:
        return _cache[name]

    # caching and lazy evaluation of environment variables
    if name in environment_variables:
        value = environment_variables[name]()
        _cache[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

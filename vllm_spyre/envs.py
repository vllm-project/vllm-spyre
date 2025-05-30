import os
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    VLLM_SPYRE_DYNAMO_BACKEND: str = "sendnn_decoder"
    VLLM_SPYRE_WARMUP_PROMPT_LENS: Optional[list[int]] = None
    VLLM_SPYRE_WARMUP_NEW_TOKENS: Optional[list[int]] = None
    VLLM_SPYRE_WARMUP_BATCH_SIZES: Optional[list[int]] = None
    VLLM_SPYRE_USE_CB: bool = False
    VLLM_SPYRE_RM_PADDED_BLOCKS: bool = False
    VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED: int = 0
    VLLM_SPYRE_PERF_METRIC_LOGGING_DIR: str = "/tmp"

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
    # - "sendnn_decoder": Compile for execution on Spyre hardware for
    #   decoder models
    # - "sendnn": Compile for execution on Spyre hardware for
    #   encoder models
    # - "inductor": Compile for execution on CPU (for debug and testing)
    # - "eager": Skip compile entirely (for debug and testing)
    "VLLM_SPYRE_DYNAMO_BACKEND":
    lambda: os.getenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn_decoder"),

    # If set, use the V1 continuous batching implementation. Otherwise, static
    # batching mode will be enabled.
    "VLLM_SPYRE_USE_CB":
    lambda: bool(int(os.getenv("VLLM_SPYRE_USE_CB", "0"))),

    # If set, remove redundant (left) padded blocks. Only applicable in
    # continuous batching mode.
    "VLLM_SPYRE_RM_PADDED_BLOCKS":
    lambda: bool(int(os.getenv("VLLM_SPYRE_RM_PADDED_BLOCKS", "0"))),

    # Enable performance metric logging. This captures startup information
    # such as warmup times, and loading times. It is turned off by default.
    "VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED":
    lambda: int(os.getenv("VLLM_SPYRE_PERF_METRIC_LOGGING_ENABLED", 0)),

    # Directory to write performance metric logging files. By default,
    # logs are written to /tmp.
    "VLLM_SPYRE_PERF_METRIC_LOGGING_DIR":
    lambda: os.getenv("VLLM_SPYRE_PERF_METRIC_LOGGING_DIR", "/tmp"),
}
# --8<-- [end:env-vars-definition]


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

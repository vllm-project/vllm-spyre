import os
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    VLLM_SPYRE_DYNAMO_BACKEND: str = "sendnn_decoder"
    VLLM_SPYRE_WARMUP_PROMPT_LENS: Optional[List[int]] = None
    VLLM_SPYRE_WARMUP_NEW_TOKENS: Optional[List[int]] = None
    VLLM_SPYRE_WARMUP_BATCH_SIZES: Optional[List[int]] = None

environment_variables: Dict[str, Callable[[], Any]] = {
    # Defines the prompt lengths the Spyre accelerator should be prepared
    # for, formatted as comma separated list.
    "VLLM_SPYRE_WARMUP_PROMPT_LENS":
    lambda: [
        int(p) for p in os.getenv(key='VLLM_SPYRE_WARMUP_PROMPT_LENS',
                                  default='64').split(',')
    ],
    # Defines the max output tokens the Spyre accelerator should be prepared
    # for, formatted as comma separated list.
    "VLLM_SPYRE_WARMUP_NEW_TOKENS":
    lambda: [
        int(d) for d in os.getenv(key='VLLM_SPYRE_WARMUP_NEW_TOKENS',
                                  default='20').split(',')
    ],
    # Defines the batch sizes the Spyre accelerator should be prepared
    # for, formatted as comma separated list.
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
    # - "eager": Skip compile entirely (for debug and testing
    "VLLM_SPYRE_DYNAMO_BACKEND":
    lambda: os.getenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn_decoder"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())

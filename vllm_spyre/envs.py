import os
from typing import TYPE_CHECKING, Any, Callable, Dict

if TYPE_CHECKING:
    VLLM_SPYRE_DYNAMO_BACKEND: str = "sendnn_decoder"

environment_variables: Dict[str, Callable[[], Any]] = {

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

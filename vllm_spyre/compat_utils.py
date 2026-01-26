import inspect
from dataclasses import fields
from functools import lru_cache
from typing import Callable


def dataclass_fields(cls: type) -> list[str]:
    return [f.name for f in fields(cls)]


@lru_cache
def has_argument(func: Callable, param_name: str) -> bool:
    # Checks the signature of a method and returns true iff the method accepts
    # a parameter named `$param_name`.
    # `lru_cache` is used because inspect + for looping is pretty slow. This
    # should not be invoked in the critical path.
    signature = inspect.signature(func)
    for param in signature.parameters.values():
        if (
            param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and param.name == param_name
        ):
            return True
    return False


# Backwards compatibility for vLLM < 0.11.0
# TODO: Remove this when dropping support for vLLM v0.10.2
from vllm import SamplingParams as SamplingParamsBase  # noqa: E402

try:
    from vllm.sampling_params import StructuredOutputsParams

    _PARAM_NAME = "structured_outputs"
except ImportError:
    from vllm.sampling_params import GuidedDecodingParams as StructuredOutputsParams  # type: ignore

    _PARAM_NAME = "guided_decoding"


class _SamplingParamsWrapper:
    """Internal wrapper class for SamplingParams.

    TODO: Remove this class when dropping support for vLLM v0.10.2
    """

    def __init__(self, params):
        object.__setattr__(self, "_params", params)

    @property
    def structured_outputs(self) -> StructuredOutputsParams | None:
        """Get the structured outputs parameter."""
        return getattr(self._params, _PARAM_NAME, None)

    @structured_outputs.setter
    def structured_outputs(self, value: StructuredOutputsParams | None) -> None:
        """Set the structured outputs parameter."""
        setattr(self._params, _PARAM_NAME, value)

    def __getattr__(self, name):
        """Delegate all other attribute access to the wrapped params."""
        return getattr(self._params, name)

    def __setattr__(self, name, value):
        """Delegate all other attribute setting to the wrapped params."""
        if name == "_params":
            object.__setattr__(self, name, value)
        else:
            setattr(self._params, name, value)


def SamplingParamsCompat(params_or_first_arg=None, **kwargs) -> _SamplingParamsWrapper:
    """Factory function that wraps SamplingParams for parameter name compatibility.

    In vLLM < 0.11.0, the parameter was called 'guided_decoding'.
    In vLLM >= 0.11.0, it was renamed to 'structured_outputs'.

    This function can be used in two ways:
    1. As a wrapper: SamplingParamsCompat(existing_params)
    2. As a constructor: SamplingParamsCompat(max_tokens=20, structured_outputs=...)

    TODO: Remove this function when dropping support for vLLM v0.10.2
    """

    # If called with an existing SamplingParams instance, wrap it
    if params_or_first_arg is not None and isinstance(params_or_first_arg, SamplingParamsBase):
        return _SamplingParamsWrapper(params_or_first_arg)

    # Otherwise, create a new SamplingParams with parameter name translation
    if "structured_outputs" in kwargs:
        # Translate structured_outputs to the correct parameter name
        kwargs[_PARAM_NAME] = kwargs.pop("structured_outputs")

    # Handle the case where params_or_first_arg might be a positional argument
    if params_or_first_arg is not None:
        # Assume it's a positional argument for SamplingParams
        params = SamplingParamsBase(params_or_first_arg, **kwargs)
    else:
        params = SamplingParamsBase(**kwargs)

    return _SamplingParamsWrapper(params)

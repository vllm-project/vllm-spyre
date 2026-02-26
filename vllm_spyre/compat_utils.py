import inspect
from dataclasses import fields
from functools import lru_cache
from typing import Callable
from packaging.version import Version
import torch
import transformers
import functools
from typing import Any
from types import ModuleType


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


def is_pytorch_lt_2_8() -> bool:
    return Version(torch.__version__) < Version("2.8.0")


def maybe_patch_torch_2_7():
    # Workaround issue with torch 2.7.1 https://github.com/pytorch/pytorch/issues/160886
    # For now, we just disable the replacement of the linear layers
    if is_pytorch_lt_2_8():
        import vllm.model_executor.models.transformers.base as transformer_utils

        @functools.wraps(transformer_utils.replace_linear_class)
        def replace_linear_class(
            linear: Any,
            style: Any = "replicate",
            quant_config: Any = None,
            *,
            prefix: str = "",
        ) -> Any:
            return linear

        transformer_utils.replace_linear_class = replace_linear_class  # ty: ignore


def is_transformers_lt_5() -> bool:
    return Version(transformers.__version__) < Version("5.0.0")


def maybe_patch_transformers_4_57(patch_backend: bool = False):
    if is_transformers_lt_5():
        if patch_backend:
            from vllm.model_executor.models.transformers.base import Base

            @functools.wraps(Base.check_version)
            def check_version(cls, min_version: str, feature: str):
                pass

            Base.check_version = check_version  # ty: ignore

        def patch_model(model_module: ModuleType, attn_classes_attr: str, class_names: list[str]):
            attn_classes = getattr(model_module, attn_classes_attr)
            attn_classes["vllm"] = attn_classes["sdpa"]

            for class_name in class_names:
                model_class = getattr(model_module, class_name)
                model_class.is_causal = False
                model_class._supports_attention_backend = True

        patch_model(
            transformers.models.bert.modeling_bert, "BERT_SELF_ATTENTION_CLASSES", ["BertModel"]
        )
        patch_model(
            transformers.models.roberta.modeling_roberta,
            "ROBERTA_SELF_ATTENTION_CLASSES",
            ["RobertaModel", "RobertaForMaskedLM", "RobertaForSequenceClassification"],
        )

        patch_model(
            transformers.models.xlm_roberta.modeling_xlm_roberta,
            "XLM_ROBERTA_SELF_ATTENTION_CLASSES",
            ["XLMRobertaModel", "XLMRobertaForMaskedLM", "XLMRobertaForSequenceClassification"],
        )

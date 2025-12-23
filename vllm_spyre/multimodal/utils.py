"""
Super hacky utils for multimodal model stuff.
"""
import torch
from fms import models
from fms.utils import serialization
from fms.utils.config import ModelConfig
from transformers import LlavaNextConfig

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    MultiModalBatchedField,
    MultiModalFieldElem,
    PlaceholderRange,
)

MULITMODAL_ARCHITECTURES = ["llava_next"]


def is_multimodal(fms_obj, fms_mm_registry):
    """Used to check in an object is multimodal; This could be a
    ModelConfig or a model object, depending on the wrapping partial
    registry."""
    if fms_mm_registry is None:
        return False
    for mm_type in fms_mm_registry:
        if isinstance(fms_obj, mm_type):
            return True
    return False


def resolve_multimodal_vocab_size(fms_config: ModelConfig):
    """Parse the FMS config for the underlying LLM's src vocab size.
    Currently this is only for Llava Next / granite vision, but as
    this is likely to vary by model, we abstract it here.
    """
    if not isinstance(fms_config, ModelConfig):
        raise TypeError(
            "Provided config is of type %s, not an FMS ModelConfig",
            type(fms_config))
    if isinstance(fms_config, models.llava_next.LlavaNextConfig):
        return fms_config.text_config.src_vocab_size
    raise ValueError(
        "Unable to resolve vocab size for multimodal config of type %s",
        type(fms_config))


def unwrap_mm_kv_cache_opts(fms_config, hf_config):
    """Unwrap options to be passed for the kv cache from the underlying
    text configs and return the resulting dictionary, which is used to
    .update() the common kv cache opts that don't need unwrapping.
    
    Currently this only handles llava next with granite llms, because
    the LLM is abstracted in HF Transformers, but hardcoded to granite in
    FMS. TODO: refactor this and ContinuousBatchingFmsModel's init
    logic to make it generic for LLMs so that we can abstract it here.
    """
    if hf_config.model_type != "llava_next" or hf_config.text_config.model_type != "granite":
        raise TypeError(
            "Currently multimodal only supports granite vision with granite llms"
        )

    kv_cache_specs = {}
    kv_cache_specs['num_layers'] = hf_config.text_config.num_hidden_layers
    kv_cache_specs['head_dim'] = getattr(
        fms_config.text_config, "head_dim",
        hf_config.text_config.hidden_size //
        hf_config.text_config.num_attention_heads)
    return kv_cache_specs


def get_mm_specific_load_overrides(model_config):
    """Get any overrides needed for fixing compile with current multimodal models.
    This is technically not specific to multimodal, but currently surfaces for 2b
    variants of granite 3.x LLMs, which is all of the granite vision models, so we
    put it here.
    """
    # head_dim expansion is required for current granite vision models.
    get_model_kwargs = {}
    if isinstance(model_config, LlavaNextConfig):
        # TODO: we should probably only do this for 2b granite models
        serialization.extend_adapter(
            "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"])
        get_model_kwargs = {
            "override_hf_pretrained_config": True,
            "text_config": {
                "head_dim": 128
            },
        }
    return get_model_kwargs


def get_multimodal_warmup_features(valid_token_ids):
    # Multimodal models take embeddings as inputs
    # since we merge multimodal features with them;
    # these take priority over input toks and skip
    # embed in the FMS model.
    mm_features = get_llava_next_image_features()
    warmup_input_ids = get_llava_next_text_features()
    warmup_embeds_tensor = torch.rand((3, warmup_input_ids.shape[-1], 4096))

    # Input features for the visual encoder
    return mm_features, warmup_input_ids, warmup_embeds_tensor


def get_llava_next_image_features():
    # This is the minimal (small) image case in granite vision;
    # The maximal shape has 11 tiles. Careful to handle the offsets
    # correctly if this is modified!
    pixel_values_shape = [3, 3, 384, 384]
    # Image size (px, not tokens)
    image_sizes = torch.tensor([118, 177], dtype=torch.int64)
    # offsets wrt text prompt
    mm_position = PlaceholderRange(offset=42, length=1836)

    mm_features = [
        MultiModalFeatureSpec(
            data=MultiModalKwargsItem({
                "pixel_values":
                MultiModalFieldElem(modality="image",
                                    key="pixel_values",
                                    data=torch.rand(
                                        pixel_values_shape,
                                        dtype=torch.bfloat16,
                                    ),
                                    field=MultiModalBatchedField()),
                "image_sizes":
                MultiModalFieldElem(modality="image",
                                    key="image_sizes",
                                    data=image_sizes,
                                    field=MultiModalBatchedField())
            }),
            modality="image",
            identifier="MM-warmup-llava-next",
            mm_position=mm_position,
        )
    ]
    return mm_features


def get_llava_next_text_features():
    # TODO make this less hacky, build dynamically, don't hardcode image token
    img_toks = [49155] * 1836
    return torch.tensor([
        46, 110, 2946, 28318, 203, 51, 11210, 3733, 312, 39489, 1256, 461, 600,
        5549, 31251, 629, 21488, 47330, 32, 886, 47330, 13344, 17247, 30,
        16360, 30, 461, 7743, 659, 19969, 372, 322, 1256, 1182, 10017, 32, 203,
        46, 110, 496, 28318, 203
    ] + img_toks + [203, 7628, 458, 1778, 32, 203, 46, 110, 17594, 28318, 203])

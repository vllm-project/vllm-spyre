"""
Super hacky utils for multimodal model stuff.
"""
from fms import models
from fms.utils import serialization
from fms.utils.config import ModelConfig
from transformers import LlavaNextConfig

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
        raise TypeError("Provided config is of type %s, not an FMS ModelConfig", type(fms_config))
    if isinstance(fms_config, models.llava_next.LlavaNextConfig):
        return fms_config.text_config.src_vocab_size
    raise ValueError("Unable to resolve vocab size for multimodal config of type %s", type(fms_config))

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
        raise TypeError("Currently multimodal only supports granite vision with granite llms")

    kv_cache_specs = {}
    kv_cache_specs['num_layers'] = hf_config.text_config.num_hidden_layers
    kv_cache_specs['head_dim'] = getattr(
        fms_config.text_config, "head_dim",
        hf_config.text_config.hidden_size // hf_config.text_config.num_attention_heads)
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
            "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"]
        )
        get_model_kwargs = {
            "override_hf_pretrained_config": True,
            "text_config": {"head_dim": 128},
        }
    return get_model_kwargs

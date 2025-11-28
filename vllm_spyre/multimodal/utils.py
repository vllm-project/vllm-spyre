"""
Super hacky utils for multimodal model stuff.
"""
from fms import models
from fms.utils.config import ModelConfig

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

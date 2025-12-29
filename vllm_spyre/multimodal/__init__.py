from typing import Optional

import transformers

from vllm_spyre.multimodal.mm_mappings import LlavaNextMMUtils, MMUtilsBase

# Maps transformers classes to the corresponding utils
MM_HF_CFG_REGISTRY = {
    transformers.LlavaNextConfig: LlavaNextMMUtils,
}


def get_mm_specific_load_overrides(hf_config: transformers.PretrainedConfig):
    # Ensure the model is multimodal, otherwise we have no overrides
    cfg_type = type(hf_config)
    if cfg_type not in MM_HF_CFG_REGISTRY:
        return {}
    return MM_HF_CFG_REGISTRY[cfg_type].get_mm_specific_load_overrides(
        hf_config)


def maybe_get_mm_utils(fms_config, hf_config) -> Optional[MMUtilsBase]:
    """Create an instance of the corresponding multimodal model's utils
    if one exists; if it doesn't, the model is not multimodal.
    """
    if type(hf_config) in MM_HF_CFG_REGISTRY:
        util_cls = MM_HF_CFG_REGISTRY[type(hf_config)]
        return util_cls(
            fms_config=fms_config,
            hf_config=hf_config,
        )

    return None  # Not a multimodal model

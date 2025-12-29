from typing import Optional

import transformers
from fms import models

from vllm_spyre.multimodal.mm_mappings import LlavaNextMMUtils, MMUtilsBase

MM_CFG_MAPPING = {
    # FMS configs
    models.llava_next.LlavaNextConfig:
    LlavaNextMMUtils,
    # Analogous mappings for Transformers configs
    transformers.LlavaNextConfig:
    LlavaNextMMUtils,
}


def get_mm_specific_load_overrides(hf_config: transformers.PretrainedConfig):
    # Ensure the model is multimodal, otherwise we have no overrides
    cfg_type = type(hf_config)
    if cfg_type not in MM_CFG_MAPPING:
        return {}
    return MM_CFG_MAPPING[cfg_type].get_mm_specific_load_overrides(hf_config)


def maybe_get_mm_utils(fms_config, hf_config) -> Optional[MMUtilsBase]:
    """Create an instance of the corresponding multimodal model's utils
    if one exists; if it doesn't, the model is not multimodal.
    """
    fms_config_cls = type(fms_config)
    hf_config_cls = type(hf_config)

    if fms_config_cls in MM_CFG_MAPPING:
        util_cls = MM_CFG_MAPPING[fms_config_cls]
    elif hf_config_cls in MM_CFG_MAPPING:
        util_cls = MM_CFG_MAPPING[hf_config_cls]
    else:
        return None  # Not multimodal
    return util_cls(fms_config=fms_config, hf_config=hf_config)

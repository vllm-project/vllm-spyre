from fms import models
from functools import partial
from vllm_spyre.multimodal.mm_model_info import MultiModalMappingInfo
from vllm_spyre.multimodal.utils import is_multimodal
# Multimodal architectures; currently this maps FMS architectures
# to additional info needed to process multimodal inputs.
FMS_MM_REGISTRY = {
    models.llava_next.LlavaNext: MultiModalMappingInfo(special_token_map={"image": "<image>"})
}

is_multimodal = partial(is_multimodal, fms_mm_registry=FMS_MM_REGISTRY)

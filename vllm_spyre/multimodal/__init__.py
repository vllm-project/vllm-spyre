from fms import models
from functools import partial
from vllm_spyre.multimodal.mm_model_info import MultiModalMappingInfo
from vllm_spyre.multimodal.utils import (
    is_multimodal,
    resolve_multimodal_vocab_size,
)

# TODO - we can definitely consolidate and combine some of this stuff,
# but for now, leaving as is, and just trying to contain as many mm
# details within this package as possible for the first pass.

# Multimodal architectures; currently this maps FMS architectures
# to additional info needed to process multimodal inputs.
FMS_MM_REGISTRY = {
    models.llava_next.LlavaNext: MultiModalMappingInfo(
        special_token_map={"image": "<image>"},
    )
}

# Maps FMS ModelConfig classes to the corresponding
# FMS model class this is mostly used for convenience
# for getting things like the source vocab.
FMS_MM_CFG_TO_ARCH = {
    models.llava_next.LlavaNextConfig: models.llava_next.LlavaNext,
}

# is_multimodal_model = partial(is_multimodal, fms_mm_registry=FMS_MM_REGISTRY)
is_multimodal_config = partial(is_multimodal, fms_mm_registry=FMS_MM_CFG_TO_ARCH)

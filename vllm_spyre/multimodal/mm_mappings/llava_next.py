import torch
from fms.utils import serialization
from fms.utils.config import ModelConfig
from transformers import PretrainedConfig

from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalKwargsItem,
    MultiModalBatchedField,
    MultiModalFieldElem,
    PlaceholderRange,
)

from vllm_spyre.multimodal.mm_mappings.base import MMUtilsBase


class LlavaNextMMUtils(MMUtilsBase):

    @staticmethod
    def _validate_configs(fms_config: ModelConfig,
                          hf_config: PretrainedConfig):
        """Ensure that configs are properly typed. Additional validation, e.g.,
        validating subconfig attrs should generally be done within subclasses.
        """
        MMUtilsBase._validate_configs(fms_config, hf_config)
        if hf_config.model_type != "llava_next" or hf_config.text_config.model_type != "granite":
            raise TypeError(
                "Currently llava next / granite vision only supports granite LLMs!"
            )

    def unwrap_mm_kv_cache_opts(self):
        """Unwrap options to be passed for the kv cache from the underlying
        text configs and return the resulting dictionary, which is used to
        .update() the common kv cache opts that don't need unwrapping.
        """
        kv_cache_specs = {}
        kv_cache_specs[
            'num_layers'] = self.hf_config.text_config.num_hidden_layers
        kv_cache_specs['head_dim'] = getattr(
            self.fms_config.text_config, "head_dim",
            self.hf_config.text_config.hidden_size //
            self.hf_config.text_config.num_attention_heads)
        return kv_cache_specs

    @staticmethod
    def get_mm_specific_load_overrides(hf_config: PretrainedConfig):
        """Get any overrides needed for initializing the FMS model from the transformers
        config. For this model, we need to fix the head_dim, which currently surfaces
        as a problem for all 2b variants of granite 3.x LLMs when running through FMS.
        
        TODO: If additional variants of granite vision are added, or broader llava
        next support is added, we should be sure to handle it properly here.
        """
        serialization.extend_adapter(
            "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"])
        return {
            "override_hf_pretrained_config": True,
            "text_config": {
                "head_dim": 128
            },
        }

    def get_multimodal_warmup_features(self, *args, **kwargs):
        # TODO split this up and just get the features directly post refactor
        # Multimodal models take embeddings as inputs
        # since we merge multimodal features with them;
        # these take priority over input toks and skip
        # embed in the FMS model.
        mm_features = self.get_llava_next_image_features()
        warmup_input_ids = self.get_llava_next_text_features()
        warmup_embeds_tensor = torch.rand(
            (3, warmup_input_ids.shape[-1], 4096))

        # Input features for the visual encoder
        return mm_features, warmup_input_ids, warmup_embeds_tensor

    def get_llava_next_image_features(self):
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

    def get_llava_next_text_features(self):
        # TODO make this less hacky, build dynamically, don't hardcode image token
        img_toks = [self.get_multimodal_token_id()] * 1  #1836
        return torch.tensor(
            [
                46, 110, 2946, 28318, 203, 51, 11210, 3733, 312, 39489, 1256,
                461, 600, 5549, 31251, 629, 21488, 47330, 32, 886, 47330,
                13344, 17247, 30, 16360, 30, 461, 7743, 659, 19969, 372, 322,
                1256, 1182, 10017, 32, 203, 46, 110, 496, 28318, 203
            ] + img_toks +
            [203, 7628, 458, 1778, 32, 203, 46, 110, 17594, 28318, 203])

    def get_multimodal_token(self) -> str:
        return "<image>"  # TODO pull from cfg; this is mostly for convenience

    def get_multimodal_token_id(self) -> int:
        return 49155  #TODO pull from cfg

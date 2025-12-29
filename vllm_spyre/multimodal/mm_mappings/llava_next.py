import torch
from fms.utils import serialization
from fms.utils.config import ModelConfig
from transformers import PretrainedConfig
from vllm.multimodal.inputs import (MultiModalBatchedField,
                                    MultiModalFeatureSpec, MultiModalFieldElem,
                                    MultiModalKwargsItem, PlaceholderRange)

from vllm_spyre.multimodal.mm_mappings import MMUtilsBase


class LlavaNextMMUtils(MMUtilsBase):

    @staticmethod
    def _validate_configs(fms_config: ModelConfig,
                          hf_config: PretrainedConfig):
        """Ensure that configs are properly typed. Additional validation, e.g.,
        validating subconfig attrs should generally be done within subclasses.
        """
        MMUtilsBase._validate_configs(fms_config, hf_config)
        if (hf_config.model_type != "llava_next"
                or hf_config.text_config.model_type != "granite"):
            raise TypeError("llava next currently only supports granite LLMs!")

    def unwrap_mm_kv_cache_opts(self):
        """Unwrap options to be passed for the kv cache from the underlying
        text configs and return the resulting dictionary, which is used to
        .update() the common kv cache opts that don't need unwrapping.
        """
        kv_cache_specs = {}
        # NOTE: this is granite LLM specific, since the only llava next
        # variant supported in FMS is currently granite vision.
        kv_cache_specs[
            'num_layers'] = self.hf_config.text_config.num_hidden_layers
        kv_cache_specs['head_dim'] = getattr(
            self.fms_config.text_config, "head_dim",
            self.hf_config.text_config.hidden_size //
            self.hf_config.text_config.num_attention_heads)
        return kv_cache_specs

    @staticmethod
    def get_mm_specific_load_overrides(hf_config: PretrainedConfig):
        """Get any overrides needed for initializing the FMS model from the
        transformers config. For this model, we need to fix the head_dim, which
        currently surfaces as a problem for all 2b variants of granite 3.x LLMs
        when running through FMS.
        
        TODO: If additional variants of granite vision are added, or broader
        llava next support is added in FMS, handle it properly here.
        """
        serialization.extend_adapter(
            "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"])
        return {
            "override_hf_pretrained_config": True,
            "text_config": {
                "head_dim": 128
            },
        }

    @staticmethod
    def get_maybe_mm_embeddings(
        fms_model: torch.nn.Module,
        input_ids: torch.Tensor,
        mm_features: list[MultiModalFeatureSpec],
        is_decode: bool,
    ) -> torch.Tensor:
        """Get the text or multimodal embeddings for Llava Next using
        the (potentially compiled) FMS model.
        """
        fms_kwargs = {"use_cache": True}
        mm_spec_keys = ["pixel_values", "image_sizes"]

        # Only merge multimodal features in prefill; nothing mm in decode
        if mm_features:
            if len(mm_features) != 1:
                raise ValueError(
                    "Currently we assume we only embed one mm request at a time"
                )
            mm_spec = mm_features[0].data
            if mm_spec is not None:
                # NOTE: This should be pretty safe as it's dependent on the
                # vLLM/HF processor objects, but we check it anyway to be safe
                # for now, since transformers 5.0 is just around the corner.
                if any(k not in mm_spec for k in mm_spec_keys):
                    raise KeyError(
                        f"Llava Next requires kwargs: {mm_spec_keys}")

                fms_kwargs["pixel_values"] = mm_spec["pixel_values"].data
                image_sizes = mm_spec["image_sizes"].data

                # Careful about this; if it's 1D, we'll a tensor of shape
                # [x, y], which will break in a weird way in image packing,
                # since it assumes it's 2D and will get sad about getting
                # an int instead of an iterable
                if image_sizes.ndim == 1:
                    image_sizes = image_sizes.unsqueeze(0)
                fms_kwargs["image_sizes"] = image_sizes

        # NOTE: use_cache is actually not used here, but currently it's
        # required. Also, the value of iteration for decode as long as it's > 0.
        input_embeds, _ = fms_model.prepare_inputs_for_generation(
            iteration=0 if not is_decode else 1,
            input_ids=input_ids,
            kwargs=fms_kwargs)
        return input_embeds

    def get_warmup_mm_features(self):
        """Get warmup features for Llava Next / Granite Vision."""
        # This is the minimal (small) image case in granite vision;
        # The maximal shape has 11 tiles. Careful to handle the offsets
        # correctly if this is modified!
        tile_size = self.hf_config.vision_config.image_size
        img_size = [118, 177]
        # TODO: Make this more dynamic and calc based on vLLM utils
        pixel_values_shape = [3, 3, tile_size, tile_size]
        # Image size (px, not tokens)
        image_sizes = torch.tensor(img_size, dtype=torch.int64)
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

    def get_warmup_tokens(self) -> torch.Tensor:
        # TODO make this less hacky, build dynamically
        img_toks = [self.get_multimodal_token_id()] * 1836
        return torch.tensor(
            [
                46, 110, 2946, 28318, 203, 51, 11210, 3733, 312, 39489, 1256,
                461, 600, 5549, 31251, 629, 21488, 47330, 32, 886, 47330,
                13344, 17247, 30, 16360, 30, 461, 7743, 659, 19969, 372, 322,
                1256, 1182, 10017, 32, 203, 46, 110, 496, 28318, 203
            ] + img_toks +
            [203, 7628, 458, 1778, 32, 203, 46, 110, 17594, 28318, 203])

    def get_warmup_embeds_tensor(self, num_requests=3) -> torch.Tensor:
        warmup_input_ids = self.get_warmup_tokens()
        emb_dim = self.hf_config.text_config.hidden_size
        seq_len = warmup_input_ids.shape[-1]
        return torch.rand((num_requests, seq_len, emb_dim))

    def get_multimodal_token_id(self) -> int:
        return self.hf_config.image_token_index

import torch
from fms.utils import serialization
from fms.utils.config import ModelConfig
from transformers import PretrainedConfig
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

from vllm_spyre.multimodal.mm_mappings import MMUtilsBase, MMWarmupInputs


class Mistral3MMUtils(MMUtilsBase):
    @staticmethod
    def _validate_configs(fms_config: ModelConfig, hf_config: PretrainedConfig):
        """Ensure that configs are properly typed. Additional validation, e.g.,
        validating subconfig attrs should generally be done within subclasses.
        """
        MMUtilsBase._validate_configs(fms_config, hf_config)
        if hf_config.model_type != "mistral3" or hf_config.text_config.model_type != "mistral":
            # HF config maps mistral3 model_type to pixtral
            raise TypeError("mistral3 currently only supports mistral LLMs!")

    def unwrap_mm_kv_cache_opts(self):
        """Unwrap options to be passed for the kv cache from the underlying
        text configs and return the resulting dictionary, which is used to
        .update() the common kv cache opts that don't need unwrapping.
        """
        kv_cache_specs = {}
        # NOTE: this is mistral LLM specific, since the only mistral3
        # variant supported in FMS is currently pixtral.
        kv_cache_specs["num_layers"] = self.hf_config.text_config.num_hidden_layers
        kv_cache_specs["head_dim"] = getattr(
            self.fms_config.text_config, "head_dim", self.hf_config.text_config.head_dim
        )
        return kv_cache_specs

    @staticmethod
    def get_maybe_mm_embeddings(
        fms_model: torch.nn.Module,
        input_ids: torch.Tensor,
        mm_features: list[MultiModalFeatureSpec],
        is_decode: bool,
    ) -> torch.Tensor:
        """Get the text or multimodal embeddings for mistral3 using
        the (potentially compiled) FMS model.
        """
        fms_kwargs = {"use_cache": True}
        mm_spec_keys = ["pixel_values", "image_sizes"]

        # Only merge multimodal features in prefill; nothing mm in decode
        if mm_features:

            if len(mm_features) != 1:
                raise ValueError("Currently we assume we only embed one mm request at a time")
            mm_spec = mm_features[0].data
            if mm_spec is not None:
                # NOTE: This should be pretty safe as it's dependent on the
                # vLLM/HF processor objects, but we check it anyway to be safe
                # for now, since transformers 5.0 is just around the corner.
                if any(k not in mm_spec for k in mm_spec_keys):
                    raise KeyError(f"Mistral3 requires kwargs: {mm_spec_keys}")

                pixel_values = mm_spec["pixel_values"].data
                # FMS vision tower expects pixel_values with batch dimension
                # If squeezed during spec building, add it back
                if pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                fms_kwargs["pixel_values"] = pixel_values

                # Use the processor's image_sizes which tracks the logical image dimensions
                # This is used by the projector to correctly split/merge patches
                image_sizes_tensor = mm_spec["image_sizes"].data
                if image_sizes_tensor.ndim == 1:
                    # Single image: convert to list of tuples
                    image_sizes = [(image_sizes_tensor[0].item(), image_sizes_tensor[1].item())]
                else:
                    # Multiple images
                    image_sizes = [(h.item(), w.item()) for h, w in image_sizes_tensor]
                fms_kwargs["image_sizes"] = image_sizes

        # The value of iteration does not matter for decode as long as it's > 0
        input_embeds, _ = fms_model.prepare_inputs_for_generation(
            iteration=0 if not is_decode else 1, input_ids=input_ids, kwargs=fms_kwargs
        )  # ty: ignore[call-non-callable]
        return input_embeds

    def get_warmup_inputs(self, req_count: int) -> MMWarmupInputs:
        """Get the inputs to the huggingface processor to create the warmup
        features or feature shapes.
        """
        # Warmup text is just an image token
        dummy_tokens = [self.hf_processor.decode(self.get_multimodal_token_id())]

        # Warmup with the minimal nontrivial case (4x4 patch); note that mistral
        # positionally encodes the image directly and does not break into tiles
        # like many VLMs.
        # Note: spatial_merge_size for mistral is 2, in FMS currently, we do
        # squeeze(0) on image features in _get_image_features function
        # before splitting, which means, if we only have 1 patch, and 1st dim is is 1, we get
        # incorrect dimension of image_features
        side_dim = self.hf_config.vision_config.patch_size * 4
        dummy_img = torch.zeros((3, side_dim, side_dim), dtype=torch.uint8)

        proc_res = self.hf_processor(
            text=dummy_tokens,
            images=dummy_img,
            return_tensors="pt",
        )

        seq_len = proc_res.input_ids.shape[-1]
        # Get the input tokens and embeddings; currently embeddings are used,
        # but tokens are still required for the interfaces to be happy.
        warmup_input_ids = proc_res.input_ids.squeeze(0)
        emb_dim = self.hf_config.text_config.hidden_size
        warmup_embeds = torch.rand((seq_len, emb_dim))
        # Get the multimodal features spec
        warmup_mm_features = self._build_multimodal_spec(proc_res)

        return MMWarmupInputs(
            input_ids=[warmup_input_ids.tolist()] * req_count,
            input_embeds=[warmup_embeds] * req_count,
            mm_features=warmup_mm_features,
        )

    def _build_multimodal_spec(self, proc_res):
        """Given output of the processor on warmup data, build MM features.

        NOTE: Currently assuming single image inputs for warmup, since we just
        use the minimal case.
        """
        # HF Processing will add image break / end tokens etc, so we need to make sure
        # offsets correspond to the image tokens, and not the delimiter toks
        # https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/pixtral/processing_pixtral.py#L192
        input_ids = proc_res.input_ids.squeeze(0)

        img_tok_mask = input_ids == self.get_multimodal_token_id()
        # Get the position of image tokens
        img_token_positions = torch.where(input_ids == self.get_multimodal_token_id())[0]
        img_start = img_token_positions[0].item()  # First image token position
        num_img_toks = torch.sum(img_tok_mask).item()

        # Multimodal features / feature spec
        mm_position = PlaceholderRange(offset=img_start, length=num_img_toks)
        mm_data = {
            "pixel_values": proc_res.pixel_values.squeeze(axis=0),
            "image_sizes": proc_res.image_sizes.squeeze(axis=0),
        }
        mm_fields = MultiModalKwargsItem(
            {
                mm_key: MultiModalFieldElem(
                    modality="image", key=mm_key, data=mm_data, field=MultiModalBatchedField()
                )
                for mm_key, mm_data in mm_data.items()
            }
        )

        return [
            MultiModalFeatureSpec(
                data=mm_fields,
                modality="image",
                identifier="MM-warmup-mistral3",
                mm_position=mm_position,
            )
        ]

    def get_multimodal_token_id(self) -> int:
        return self.hf_config.image_token_index

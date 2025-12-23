ALLOWED_MULTIMODAL_KEYS = ["image"]  # TODO expand


class MultiModalMappingInfo:

    def __init__(self, special_token_map):
        """Initializes multimodal mapping information for this FMS model;
        generally this includes things like mapping the multimodal token
        etc.
        """
        self._validate_modalities(special_token_map.keys())
        self.special_token_map = special_token_map

    def _validate_modalities(self, special_token_keys):
        for key in special_token_keys:
            assert key in ALLOWED_MULTIMODAL_KEYS

    def resolve_multimodal_shapes(self):
        # Adapts the prompt to be used for warmup to include the multimodal features;
        # we need to do this since practically, things like <image> are essentially
        # expanded to be their actual feature size, as opposed to a single token.
        # As such, when profiling, we should replace any multimodal tokens with the
        # maximum number of features per multimiodal input for that model.
        #
        # E.g., in the case of llava next / granite vision, each <image> tag should be
        # expanded to the case in which we have the maximum number of tiles resolved
        # through anyres.
        #
        # NOTE: this is basically what we want to accomplish through dummy inputs in:
        # https://github.com/vllm-project/vllm/blob/main/docs/contributing/model/multimodal.md?plain=1
        #
        # This is still TODO, and for now I am just setting shape sizes to be 8k,
        # which is bigger than the max features per image (~5k) in granite vision
        raise NotImplementedError(
            "multimodal warmup shape resolution not implemented")

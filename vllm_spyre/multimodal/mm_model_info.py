ALLOWED_MULTIMODAL_KEYS = ["image"] # TODO expand

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

    def _get_warmup_data(self):
        # Map the model architecture and config to the correct
        # warmup configuration based on multimodal
        raise NotImplementedError()

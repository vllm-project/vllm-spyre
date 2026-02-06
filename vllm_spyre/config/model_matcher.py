"""Pattern-based model matching."""

from typing import Any

from vllm.logger import init_logger

from vllm_spyre.config.model_config import ArchitecturePattern

logger = init_logger(__name__)


class ModelMatcher:
    """Pattern-based model matching for identifying models from HF configs."""

    def _validate_quantization_config(
        self, model_name: str, config_value: Any, pattern_value: dict
    ) -> bool:
        """Validate quantization_config dictionary match.

        Args:
            model_name: Model name for logging purposes
            config_value: Actual quantization config from HF config
            pattern_value: Expected quantization config pattern

        Returns:
            True if quantization config matches, False otherwise
        """
        if not isinstance(config_value, dict):
            logger.debug(
                "Model '%s': quantization_config type mismatch: config=%s, pattern=%s",
                model_name,
                type(config_value),
                type(pattern_value),
            )
            return False

        for key, value in pattern_value.items():
            if config_value.get(key) != value:
                logger.debug(
                    "Model '%s': quantization_config['%s'] mismatch: config=%s, pattern=%s",
                    model_name,
                    key,
                    config_value.get(key),
                    value,
                )
                return False

        return True

    def _validate_attribute(
        self, hf_config: Any, model_name: str, attr_name: str, pattern_value: Any
    ) -> bool:
        """Validate a single attribute match between config and pattern.

        Args:
            hf_config: HuggingFace model configuration object
            model_name: Model name for logging purposes
            attr_name: Name of the attribute to validate
            pattern_value: Expected value for the attribute (never None)

        Returns:
            True if the attribute matches, False otherwise
        """
        if not hasattr(hf_config, attr_name):
            logger.debug(
                "Model '%s': HF config missing attribute '%s' required by pattern",
                model_name,
                attr_name,
            )
            return False

        config_value = getattr(hf_config, attr_name)

        if attr_name == "quantization_config" and isinstance(pattern_value, dict):
            return self._validate_quantization_config(model_name, config_value, pattern_value)

        if config_value != pattern_value:
            logger.debug(
                "Model '%s': Attribute '%s' mismatch: config=%s, pattern=%s",
                model_name,
                attr_name,
                config_value,
                pattern_value,
            )
            return False

        return True

    def matches(self, hf_config: Any, pattern: ArchitecturePattern) -> bool:
        """Check if HF config matches architecture pattern.

        Args:
            hf_config: HuggingFace model configuration object
            pattern: Architecture pattern to match against

        Returns:
            True if the config matches the pattern, False otherwise
        """
        model_name = pattern.model_name

        if not hasattr(hf_config, "model_type"):
            logger.debug("Model '%s': HF config missing 'model_type' attribute", model_name)
            return False

        if hf_config.model_type != pattern.model_type:
            logger.debug(
                "Model '%s': Model type mismatch: config=%s, pattern=%s",
                model_name,
                hf_config.model_type,
                pattern.model_type,
            )
            return False

        for attr_name, pattern_value in pattern.attributes.items():
            if not self._validate_attribute(hf_config, model_name, attr_name, pattern_value):
                return False

        logger.debug("Model '%s': HF config matches pattern", model_name)
        return True


# Made with Bob

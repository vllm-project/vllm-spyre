"""Pattern-based model matching."""

from typing import Any

from vllm.logger import init_logger

from vllm_spyre.config.model_config import ArchitecturePattern

logger = init_logger(__name__)


class ModelMatcher:
    """Pattern-based model matching for identifying models from HF configs."""

    def matches(self, hf_config: Any, pattern: ArchitecturePattern) -> bool:
        """Check if HF config matches architecture pattern.

        Args:
            hf_config: HuggingFace model configuration object
            pattern: Architecture pattern to match against

        Returns:
            True if the config matches the pattern, False otherwise
        """
        # Check model_type first (required)
        if not hasattr(hf_config, "model_type"):
            logger.debug("HF config missing 'model_type' attribute")
            return False

        if hf_config.model_type != pattern.model_type:
            logger.debug(
                "Model type mismatch: config=%s, pattern=%s",
                hf_config.model_type,
                pattern.model_type,
            )
            return False

        # Check optional attributes
        for attr_name in [
            "num_hidden_layers",
            "max_position_embeddings",
            "hidden_size",
            "vocab_size",
            "num_key_value_heads",
            "num_attention_heads",
            "num_experts_per_tok",
        ]:
            pattern_value = getattr(pattern, attr_name)
            if pattern_value is None:
                continue  # Skip if not specified in pattern

            if not hasattr(hf_config, attr_name):
                logger.debug(
                    "HF config missing attribute '%s' required by pattern %s", attr_name, pattern.model_type
                )
                return False

            config_value = getattr(hf_config, attr_name)
            if config_value != pattern_value:
                logger.debug(
                    "Attribute '%s' mismatch: config=%s, pattern=%s",
                    attr_name,
                    config_value,
                    pattern_value,
                )
                return False

        logger.debug("HF config matches pattern for model_type=%s", pattern.model_type)
        return True

# Made with Bob

"""Model configuration registry."""

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from vllm.logger import init_logger

from vllm_spyre.config.model_config import ModelConfig
from vllm_spyre.config.model_matcher import ModelMatcher

if TYPE_CHECKING:
    from vllm.config import ModelConfig as VllmModelConfig

    from vllm_spyre.config.configurators.model_configurator import ModelConfigurator

logger = init_logger(__name__)


class ModelConfigRegistry:
    """Singleton registry for model configurations.

    This registry manages model configurations loaded from YAML files
    and provides methods to match models and retrieve configurators.
    """

    _instance: "ModelConfigRegistry | None" = None
    _initialized: bool = False

    def __init__(self):
        """Initialize the registry."""
        self._models: dict[str, ModelConfig] = {}
        self._configurators: dict[str, "ModelConfigurator"] = {}
        self._matcher = ModelMatcher()

    @classmethod
    def get_instance(cls) -> "ModelConfigRegistry":
        """Get singleton instance.

        Returns:
            The singleton ModelConfigRegistry instance
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def initialize(self, config_path: Path | None = None) -> None:
        """Load configurations from YAML file.

        Args:
            config_path: Path to model_configs.yaml file. If None, uses default location.
        """
        if self._initialized:
            logger.debug("Registry already initialized, skipping")
            return

        if config_path is None:
            config_path = Path(__file__).parent / "model_configs.yaml"

        if not config_path.exists():
            logger.warning(
                "Model configuration file not found at %s. Registry will be empty.",
                config_path,
            )
            self._initialized = True
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            for model_name, model_data in data.get("models", {}).items():
                model_config = ModelConfig.from_dict(model_name, model_data)
                self.register_model(model_config)

            self._initialized = True
            logger.info("Loaded %d model configurations from %s", len(self._models), config_path)
        except Exception as e:
            logger.error("Failed to load model configurations from %s: %s", config_path, e, exc_info=True)
            self._initialized = True  # Mark as initialized even on error to avoid retry loops

    def register_model(self, model_config: ModelConfig) -> None:
        """Register a model configuration.

        Args:
            model_config: The model configuration to register
        """
        self._models[model_config.name] = model_config
        logger.debug("Registered model: %s", model_config.name)

    def get_model_config(self, model_name: str) -> ModelConfig | None:
        """Get configuration for a specific model.

        Args:
            model_name: Name of the model

        Returns:
            ModelConfig if found, None otherwise
        """
        return self._models.get(model_name)

    def find_matching_model(self, vllm_model_config: "VllmModelConfig") -> str | None:
        """Find model name by matching HF config.

        Args:
            vllm_model_config: vLLM model configuration containing HF config

        Returns:
            Model name if a match is found, None otherwise
        """
        hf_config = vllm_model_config.hf_config
        if hf_config is None:
            logger.debug("No HF config available for matching")
            return None

        for model_name, model_config in self._models.items():
            if self._matcher.matches(hf_config, model_config.architecture):
                logger.info(
                    "Matched model '%s' to configuration '%s'",
                    vllm_model_config.model,
                    model_name,
                )
                return model_name

        logger.debug(
            "No matching model configuration found for '%s'", vllm_model_config.model
        )
        return None

    def get_configurator(self, model_name: str) -> "ModelConfigurator":
        """Get or create configurator for a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelConfigurator instance for the model

        Raises:
            ValueError: If model is not registered
        """
        if model_name in self._configurators:
            return self._configurators[model_name]

        model_config = self.get_model_config(model_name)
        if model_config is None:
            raise ValueError(f"Model '{model_name}' not registered")

        configurator = self._create_configurator(model_config)
        self._configurators[model_name] = configurator
        return configurator

    def _create_configurator(self, model_config: ModelConfig) -> "ModelConfigurator":
        """Create configurator instance.

        Args:
            model_config: Model configuration

        Returns:
            ModelConfigurator instance
        """
        from vllm_spyre.config.configurators.model_configurator import ModelConfigurator

        logger.debug("Creating configurator for model %s", model_config.name)
        return ModelConfigurator(model_config)

    def list_models(self) -> list[str]:
        """List all registered model names.

        Returns:
            List of model names
        """
        return list(self._models.keys())


def get_model_registry() -> ModelConfigRegistry:
    """Get the global model registry instance.

    This is a convenience function that ensures the registry is initialized.

    Returns:
        The initialized ModelConfigRegistry instance
    """
    registry = ModelConfigRegistry.get_instance()
    if not registry._initialized:
        registry.initialize()
    return registry

# Made with Bob

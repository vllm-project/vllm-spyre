"""Model configurator for applying model-specific configurations."""

import os
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_spyre import envs as envs_spyre

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_spyre.config.model_config import DeviceConfig, ModelConfig

logger = init_logger(__name__)


class ModelConfigurator:
    """Configurator that handles all model configurations.

    This configurator applies device configurations including:
    - Environment variables
    - GPU block overrides (with version-aware logic)
    - Chunked prefill configurations

    All features are optional and driven by the device_config in YAML.
    """

    def __init__(self, model_config: "ModelConfig"):
        """Initialize configurator with model configuration.

        Args:
            model_config: The model configuration to use
        """
        self.model_config = model_config

    def configure(self, vllm_config: "VllmConfig") -> None:
        """Apply device configurations.

        Args:
            vllm_config: The vLLM configuration to modify
        """
        tp_size = vllm_config.parallel_config.world_size
        device_config = self.get_device_config(tp_size)

        if device_config is None:
            logger.debug(
                "No device configuration for model '%s' with TP=%d",
                self.model_config.name,
                tp_size,
            )
            return

        # Apply environment variables
        for key, value in device_config.env_vars.items():
            self.set_env_var(key, value, override=False)

        # Handle num_gpu_blocks_override with version check
        self._configure_gpu_blocks(device_config, vllm_config, tp_size)

        # Handle chunked prefill configuration
        self._configure_chunked_prefill(device_config, vllm_config, tp_size)

        logger.info(
            "Applied configuration for model '%s' with TP=%d",
            self.model_config.name,
            tp_size,
        )

    def set_env_var(
        self, key: str, value: Any, override: bool = False, log_level: str = "info"
    ) -> bool:
        """Set environment variable with logging.

        Args:
            key: Environment variable name
            value: Value to set
            override: Whether to override existing value
            log_level: Logging level ('info', 'warning', 'debug')

        Returns:
            True if variable was set, False if skipped
        """
        str_value = str(value)
        existing = os.getenv(key)

        if existing is not None and not override:
            if existing != str_value:
                logger.warning(
                    "%s was set to %s, not overriding to model default of %s",
                    key,
                    existing,
                    str_value,
                )
            return False

        os.environ[key] = str_value
        log_func = getattr(logger, log_level)
        log_func("Set %s = %s", key, str_value)
        return True

    def get_device_config(self, tp_size: int) -> "DeviceConfig | None":
        """Get device configuration for tensor parallel size.

        Args:
            tp_size: Tensor parallel size

        Returns:
            DeviceConfig if available for this TP size, None otherwise
        """
        return self.model_config.device_configs.get(tp_size)

    def _configure_gpu_blocks(
        self, device_config, vllm_config: "VllmConfig", tp_size: int
    ) -> None:
        """Configure GPU blocks with optional version-aware logic.

        Args:
            device_config: Device configuration containing block override settings
            vllm_config: The vLLM configuration to modify
            tp_size: Tensor parallel size
        """
        blocks_config = device_config.num_gpu_blocks_override
        if blocks_config is None:
            return

        # Determine which override to use
        if isinstance(blocks_config, int):
            blocks_override = blocks_config
        else:
            # Version-aware selection for torch_sendnn
            from vllm_spyre.platform import SpyrePlatform

            if (
                SpyrePlatform.sendnn_configured()
                and (0, 0, 0) < SpyrePlatform.sendnn_version() < (1, 0, 3)
            ):
                blocks_override = blocks_config.get("torch_sendnn_lt_1_0_3")
            else:
                blocks_override = blocks_config.get("default")

        if blocks_override is None:
            return

        # Apply override if not already set
        if vllm_config.cache_config.num_gpu_blocks_override is None:
            vllm_config.cache_config.num_gpu_blocks_override = blocks_override
            logger.info(
                "Model %s with TP=%d: Overriding KV Cache blocks to %d",
                self.model_config.name,
                tp_size,
                blocks_override,
            )
        elif vllm_config.cache_config.num_gpu_blocks_override != blocks_override:
            logger.warning(
                "--num-gpu-blocks-override was set to %d, not using model default of %d",
                vllm_config.cache_config.num_gpu_blocks_override,
                blocks_override,
            )

    def _configure_chunked_prefill(
        self, device_config, vllm_config: "VllmConfig", tp_size: int
    ) -> None:
        """Configure chunked prefill settings if present.

        Args:
            device_config: Device configuration containing chunked prefill settings
            vllm_config: The vLLM configuration to modify
            tp_size: Tensor parallel size
        """
        cp_config = device_config.chunked_prefill_config
        if cp_config is None:
            return

        if not envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL:
            return

        if envs_spyre.VLLM_SPYRE_DYNAMO_BACKEND != "sendnn":
            return

        if os.getenv("VLLM_DT_CHUNK_LEN") is not None:
            return

        max_batched_tokens = cp_config.get("max_num_batched_tokens")
        if max_batched_tokens:
            logger.info(
                "Model %s with TP=%d and chunked prefill: "
                "Setting --max-num-batched-tokens %d",
                self.model_config.name,
                tp_size,
                max_batched_tokens,
            )
            vllm_config.scheduler_config.max_num_batched_tokens = max_batched_tokens

# Made with Bob

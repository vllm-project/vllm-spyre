"""Model configurator for applying model-specific configurations."""

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger

from vllm_spyre import envs as envs_spyre

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_spyre.config.model_config import DeviceConfig, ModelConfig

logger = init_logger(__name__)


@dataclass
class ConfigurationSummary:
    """Summary of configuration changes applied by the configurator.

    Attributes:
        model_name: Name of the model being configured
        tp_size: Tensor parallel size
        env_vars: Dictionary of environment variables that were checked/set
        num_blocks: num_gpu_blocks_override value, if configured
        chunk_size: max_num_batched_tokens for Chunked Prefill, if configured
    """

    model_name: str
    tp_size: int
    env_vars: dict[str, str] = field(default_factory=dict)
    num_blocks: int | None = None
    chunk_size: int | None = None


class ModelConfigurator:
    """Configurator that handles all model configurations.

    This configurator applies device configurations including:
    - Environment variables
    - GPU block overrides (with version-aware logic)
    - Chunked prefill configurations

    All features are optional and driven by the device_config in YAML.
    """

    def __init__(self, model_config: "ModelConfig", device_config: "DeviceConfig | None" = None):
        """Initialize configurator with model configuration and optional device config.

        Args:
            model_config: The model configuration to use
            device_config: Optional device configuration (from matching CB config)
        """
        self.model_config = model_config
        self.device_config = device_config

    def configure(self, vllm_config: "VllmConfig") -> ConfigurationSummary:
        """Apply device configurations.

        Args:
            vllm_config: The vLLM configuration to modify

        Returns:
            ConfigurationSummary with all configuration settings checked/applied
        """
        tp_size = vllm_config.parallel_config.world_size

        summary = ConfigurationSummary(
            model_name=self.model_config.name,
            tp_size=tp_size,
        )

        if self.device_config is None:
            logger.debug(
                "No device configuration for model '%s' with TP=%d",
                self.model_config.name,
                tp_size,
            )
            return summary

        # Apply environment variables and track them
        for key, value in self.device_config.env_vars.items():
            self.set_env_var(key, value, override=False)
            summary.env_vars[key] = str(value)

        # Handle num_gpu_blocks_override with version check
        blocks_override = self._configure_gpu_blocks(self.device_config, vllm_config)
        if blocks_override is not None:
            summary.num_blocks = blocks_override

        # Handle chunked prefill configuration
        cp_tokens = self._configure_chunked_prefill(self.device_config, vllm_config)
        if cp_tokens is not None:
            summary.chunk_size = cp_tokens

        return summary

    def set_env_var(
        self, key: str, value: Any, override: bool = False, log_level: str = "debug"
    ) -> bool:
        """Set environment variable with logging.

        Args:
            key: Environment variable name
            value: Value to set
            override: Whether to override existing value
            log_level: Logging level ('info', 'warning', 'debug')

        Returns:
            True if variable was set, False if skipped

        Raises:
            RuntimeError: If VLLM_SPYRE_REQUIRE_KNOWN_CONFIG is set and existing value conflicts
        """
        str_value = str(value)
        existing = os.getenv(key)

        if existing is not None and not override:
            if existing != str_value:
                if envs_spyre.VLLM_SPYRE_REQUIRE_KNOWN_CONFIG:
                    raise RuntimeError(
                        f"Model '{self.model_config.name}' requires {key}={str_value}, "
                        f"but it was already set to {existing}. "
                        f"VLLM_SPYRE_REQUIRE_KNOWN_CONFIG is enabled."
                    )
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

    def _configure_gpu_blocks(self, device_config, vllm_config: "VllmConfig") -> int | None:
        """Configure GPU blocks with optional version-aware logic.

        Args:
            device_config: Device configuration containing block override settings
            vllm_config: The vLLM configuration to modify

        Returns:
            The num blocks override value that was configured, or None if not configured

        Raises:
            RuntimeError: If VLLM_SPYRE_REQUIRE_KNOWN_CONFIG is set and user override conflicts
        """
        blocks_config = device_config.num_gpu_blocks_override
        if blocks_config is None:
            return None

        # Determine which override to use
        if isinstance(blocks_config, int):
            num_blocks_override = blocks_config
        else:
            # Version-aware selection for torch_sendnn
            from vllm_spyre.platform import SpyrePlatform

            sendnn_version = SpyrePlatform.sendnn_version()
            if (
                SpyrePlatform.sendnn_configured()
                and sendnn_version is not None
                and (0, 0, 0) < sendnn_version < (1, 0, 3)
            ):
                num_blocks_override = blocks_config.get("torch_sendnn_lt_1_0_3")
            else:
                num_blocks_override = blocks_config.get("default")

        if num_blocks_override is None:
            return None

        # Apply override if not already set
        if vllm_config.cache_config.num_gpu_blocks_override is None:
            vllm_config.cache_config.num_gpu_blocks_override = num_blocks_override
            logger.debug(
                "Set num_gpu_blocks_override=%d for model %s",
                num_blocks_override,
                self.model_config.name,
            )
            return num_blocks_override
        elif vllm_config.cache_config.num_gpu_blocks_override != num_blocks_override:
            user_value = vllm_config.cache_config.num_gpu_blocks_override
            if envs_spyre.VLLM_SPYRE_REQUIRE_KNOWN_CONFIG:
                raise RuntimeError(
                    f"Expected runtime config for '{self.model_config.name}' requires "
                    f"num_gpu_blocks_override={num_blocks_override}, but user set "
                    f"--num-gpu-blocks-override={user_value}. "
                    f"VLLM_SPYRE_REQUIRE_KNOWN_CONFIG is enabled."
                )
            logger.warning(
                "--num-gpu-blocks-override was set to %d, not using model default of %d",
                user_value,
                num_blocks_override,
            )
            return num_blocks_override

        return num_blocks_override

    def _configure_chunked_prefill(
        self,
        device_config,
        vllm_config: "VllmConfig",
    ) -> int | None:
        """Configure chunked prefill settings if present.

        Args:
            device_config: Device configuration containing chunked prefill settings
            vllm_config: The vLLM configuration to modify
            tp_size: Tensor parallel size

        Returns:
            The max_num_batched_tokens value that was configured, or None if not configured
        """
        cp_config = device_config.chunked_prefill_config
        if cp_config is None:
            return None

        if not envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL:
            return None

        if os.getenv("VLLM_DT_CHUNK_LEN") is not None:
            return None

        max_batched_tokens = cp_config.get("max_num_batched_tokens")
        if max_batched_tokens:
            logger.debug(
                "Set max_num_batched_tokens=%d for model %s (chunked prefill)",
                max_batched_tokens,
                self.model_config.name,
            )
            vllm_config.scheduler_config.max_num_batched_tokens = max_batched_tokens
            return max_batched_tokens

        return None


# Made with Bob

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
class ConfigValue:
    """Tracks a configuration value with override information.

    Attributes:
        expected: The expected/default value from model configuration
        actual: The actual value that was applied (may differ if overridden)
    """

    expected: str | int | None
    actual: str | int | None

    def was_overridden(self) -> bool:
        """Check if the actual value differs from the expected value."""
        return self.expected != self.actual

    def __eq__(self, other: object) -> bool:
        """Compare ConfigValue with another value using the actual value.

        This allows backward compatibility with code that compares directly
        to values without accessing .actual attribute.
        """
        if isinstance(other, ConfigValue):
            return self.actual == other.actual
        return self.actual == other

    def __hash__(self) -> int:
        """Make ConfigValue hashable based on actual value."""
        return hash(self.actual)


@dataclass
class ConfigurationSummary:
    """Summary of configuration changes applied by the configurator.

    Attributes:
        model_name: Name of the model being configured
        tp_size: Tensor parallel size
        env_vars: Dictionary of environment variables with override tracking
        num_blocks: num_gpu_blocks_override value with override tracking, if configured
        chunk_size: max_num_batched_tokens for Chunked Prefill with override tracking, if configured
    """

    model_name: str
    tp_size: int
    env_vars: dict[str, ConfigValue] = field(default_factory=dict)
    num_blocks: ConfigValue | None = None
    chunk_size: ConfigValue | None = None

    def format_log_message(self) -> str:
        """Format the configuration summary as a multi-line log message.

        Returns:
            Formatted string suitable for logging with logger.info()
        """

        def format_config_line(name: str, config_value: ConfigValue) -> str:
            if config_value.was_overridden():
                return f"  {name}={config_value.actual} ⚠ (expected: {config_value.expected})"
            return f"  {name}={config_value.actual} ✓"

        def generate_lines():
            yield f"Applied registry configuration for '{self.model_name}' (TP={self.tp_size}):"

            if self.env_vars:
                yield "  Environment variables:"
                for key, config_value in self.env_vars.items():
                    yield f"  {format_config_line(key, config_value)}"

            if self.num_blocks is not None:
                yield format_config_line("num_gpu_blocks_override", self.num_blocks)

            if self.chunk_size is not None:
                yield format_config_line("max_num_batched_tokens", self.chunk_size)

        lines = list(generate_lines())
        if len(lines) == 1:
            lines.append("  no device-specific configs")

        return "\n".join(lines)


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
            config_value = self.set_env_var(key, value, override=False)
            summary.env_vars[key] = config_value

        # Handle num_gpu_blocks_override with version check
        blocks_override = self._configure_gpu_blocks(self.device_config, vllm_config)
        if blocks_override is not None:
            summary.num_blocks = blocks_override

        # Handle chunked prefill configuration
        cp_tokens = self._configure_chunked_prefill(self.device_config, vllm_config)
        if cp_tokens is not None:
            summary.chunk_size = cp_tokens

        return summary

    def _validate_config_override(
        self,
        config_name: str,
        config_value: ConfigValue,
        error_context: str,
    ) -> None:
        """Validate that actual config value matches expected value.

        Handles the common pattern of checking if a user-provided value conflicts
        with the expected model configuration value, and enforces VLLM_SPYRE_REQUIRE_KNOWN_CONFIG.

        Args:
            config_name: Name of the configuration parameter (for error messages)
            config_value: ConfigValue with expected and actual values
            error_context: Additional context for error messages (e.g., "it was already set to X")

        Raises:
            RuntimeError: If VLLM_SPYRE_REQUIRE_KNOWN_CONFIG is set and values conflict
        """
        if config_value.was_overridden():
            if envs_spyre.VLLM_SPYRE_REQUIRE_KNOWN_CONFIG:
                raise RuntimeError(
                    f"Model '{self.model_config.name}' configures "
                    f"{config_name}={config_value.expected}, "
                    f"but {error_context}. "
                    f"VLLM_SPYRE_REQUIRE_KNOWN_CONFIG is enabled."
                )
            logger.warning(
                "%s was set to %s, not using model default of %s",
                config_name,
                config_value.actual,
                config_value.expected,
            )

    def set_env_var(
        self, key: str, value: Any, override: bool = False, log_level: str = "debug"
    ) -> ConfigValue:
        """Set environment variable with logging.

        Args:
            key: Environment variable name
            value: Value to set
            override: Whether to override existing value
            log_level: Logging level ('info', 'warning', 'debug')

        Returns:
            ConfigValue tracking expected and actual values

        Raises:
            RuntimeError: If VLLM_SPYRE_REQUIRE_KNOWN_CONFIG is set and existing value conflicts
        """
        str_value = str(value)
        existing = os.getenv(key)

        if existing is not None and not override:
            config_value = ConfigValue(expected=str_value, actual=existing)
            if config_value.was_overridden():
                self._validate_config_override(
                    config_name=key,
                    config_value=config_value,
                    error_context=f"it was already set to {existing}",
                )
            return config_value

        os.environ[key] = str_value
        log_func = getattr(logger, log_level)
        log_func("Set %s = %s", key, str_value)
        return ConfigValue(expected=str_value, actual=str_value)

    def _configure_gpu_blocks(self, device_config, vllm_config: "VllmConfig") -> ConfigValue | None:
        """Configure GPU blocks with optional version-aware logic.

        Args:
            device_config: Device configuration containing block override settings
            vllm_config: The vLLM configuration to modify

        Returns:
            ConfigValue tracking expected and actual num_blocks, or None if not configured

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
            return ConfigValue(expected=num_blocks_override, actual=num_blocks_override)

        # User already set a value - validate it
        user_value = vllm_config.cache_config.num_gpu_blocks_override
        config_value = ConfigValue(expected=num_blocks_override, actual=user_value)
        self._validate_config_override(
            config_name="num_gpu_blocks_override",
            config_value=config_value,
            error_context=f"user set --num-gpu-blocks-override={user_value}",
        )
        return config_value

    def _configure_chunked_prefill(
        self,
        device_config,
        vllm_config: "VllmConfig",
    ) -> ConfigValue | None:
        """Configure chunked prefill settings if present.

        Args:
            device_config: Device configuration containing chunked prefill settings
            vllm_config: The vLLM configuration to modify
            tp_size: Tensor parallel size

        Returns:
            ConfigValue tracking expected and actual chunk_size, or None if not configured
        """
        cp_config = device_config.chunked_prefill_config
        if cp_config is None:
            return None

        if not envs_spyre.VLLM_SPYRE_USE_CHUNKED_PREFILL:
            return None

        max_batched_tokens = cp_config.get("max_num_batched_tokens")
        if not max_batched_tokens:
            return None

        # Check if user already set VLLM_DT_CHUNK_LEN
        user_chunk_len = os.getenv("VLLM_DT_CHUNK_LEN")
        if user_chunk_len is not None:
            try:
                user_value = int(user_chunk_len)
                logger.debug(
                    "VLLM_DT_CHUNK_LEN already set to %d, not using model default of %d",
                    user_value,
                    max_batched_tokens,
                )
                return ConfigValue(expected=max_batched_tokens, actual=user_value)
            except (ValueError, TypeError):
                logger.warning(
                    "VLLM_DT_CHUNK_LEN was set to invalid value %s, ignoring setting",
                    user_chunk_len,
                )
                return None

        logger.debug(
            "Set max_num_batched_tokens=%d for model %s (chunked prefill)",
            max_batched_tokens,
            self.model_config.name,
        )
        vllm_config.scheduler_config.max_num_batched_tokens = max_batched_tokens
        return ConfigValue(expected=max_batched_tokens, actual=max_batched_tokens)


# Made with Bob

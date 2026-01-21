"""Data structures for model configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArchitecturePattern:
    """Pattern for matching model architectures.

    Attributes:
        model_name: The model name/identifier (e.g., 'ibm-granite/granite-3.3-8b-instruct')
        model_type: The model type (e.g., 'granite', 'llama', 'roberta')
        attributes: Dictionary of arbitrary attributes to match against HF config.
                   Keys are attribute names, values are expected values (None means optional).
                   Common attributes include: num_hidden_layers, max_position_embeddings,
                   hidden_size, vocab_size, num_key_value_heads, num_attention_heads,
                   num_experts_per_tok, quantization_config, etc.
    """

    model_name: str
    model_type: str
    attributes: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, model_name: str, data: dict[str, Any]) -> "ArchitecturePattern":
        """Create ArchitecturePattern from dictionary.

        Args:
            model_name: The model name/identifier
            data: Dictionary containing architecture pattern data

        Returns:
            ArchitecturePattern instance
        """
        # Extract model_type (required)
        model_type = data["model_type"]

        # All other keys become attributes
        attributes = {k: v for k, v in data.items() if k != "model_type"}

        return cls(
            model_name=model_name,
            model_type=model_type,
            attributes=attributes,
        )


@dataclass
class DeviceConfig:
    """Device-specific configuration for a model.

    Attributes:
        tp_size: Tensor parallel size this config applies to
        env_vars: Environment variables to set
        num_gpu_blocks_override: Override for GPU blocks (can be int or dict with version keys)
        chunked_prefill_config: Configuration for chunked prefill
    """

    tp_size: int
    env_vars: dict[str, Any] = field(default_factory=dict)
    num_gpu_blocks_override: dict[str, int] | int | None = None
    chunked_prefill_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, tp_size: int, data: dict[str, Any]) -> "DeviceConfig":
        """Create DeviceConfig from dictionary."""
        return cls(
            tp_size=tp_size,
            env_vars=data.get("env_vars", {}),
            num_gpu_blocks_override=data.get("num_gpu_blocks_override"),
            chunked_prefill_config=data.get("chunked_prefill_config"),
        )


@dataclass
class StaticBatchingConfig:
    """Static batching configuration.

    Attributes:
        tp_size: Tensor parallel size
        warmup_shapes: Warmup shapes as (prompt_length, new_tokens, batch_size) tuples
    """

    tp_size: int
    warmup_shapes: list[tuple[int, int, int]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StaticBatchingConfig":
        """Create StaticBatchingConfig from dictionary."""
        # Convert list of lists to list of tuples
        warmup_shapes = [tuple(ws) for ws in data["warmup_shapes"]]
        return cls(
            tp_size=data["tp_size"],
            warmup_shapes=warmup_shapes,
        )


@dataclass
class ContinuousBatchingConfig:
    """Continuous batching configuration.

    Attributes:
        tp_size: Tensor parallel size
        max_model_len: Maximum model length
        max_num_seqs: Maximum number of sequences
    """

    tp_size: int
    max_model_len: int
    max_num_seqs: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuousBatchingConfig":
        """Create ContinuousBatchingConfig from dictionary."""
        return cls(
            tp_size=data["tp_size"],
            max_model_len=data["max_model_len"],
            max_num_seqs=data["max_num_seqs"],
        )


# Type alias for runtime configs
RuntimeConfig = StaticBatchingConfig | ContinuousBatchingConfig


@dataclass
class ModelConfig:
    """Complete model configuration.

    Attributes:
        name: Model name/identifier
        architecture: Architecture pattern for matching
        static_batching_configs: List of static batching configurations
        continuous_batching_configs: List of continuous batching configurations
        device_configs: Device-specific configurations keyed by TP size (only for CB)
    """

    name: str
    architecture: ArchitecturePattern
    static_batching_configs: list[StaticBatchingConfig] = field(default_factory=list)
    continuous_batching_configs: list[ContinuousBatchingConfig] = field(default_factory=list)
    device_configs: dict[int, DeviceConfig] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary (typically from YAML).

        Args:
            name: Model name
            data: Dictionary containing model configuration

        Returns:
            ModelConfig instance
        """
        # Parse architecture (pass model name for better logging)
        architecture = ArchitecturePattern.from_dict(name, data["architecture"])

        # Parse supported configs - separate into static and continuous batching
        static_configs = []
        continuous_configs = []

        for cfg in data.get("supported_configs", []):
            if "warmup_shapes" in cfg:
                # Static batching config
                static_configs.append(StaticBatchingConfig.from_dict(cfg))
            elif "max_model_len" in cfg and "max_num_seqs" in cfg:
                # Continuous batching config
                continuous_configs.append(ContinuousBatchingConfig.from_dict(cfg))
            else:
                raise ValueError(
                    f"Invalid config for model {name}: must have either 'warmup_shapes' "
                    f"(static batching) or 'max_model_len' and 'max_num_seqs' (continuous batching)"
                )

        # Parse device configs (only relevant for continuous batching)
        device_configs = {}
        for tp_size_str, device_data in data.get("device_configs", {}).items():
            tp_size = int(tp_size_str)
            device_configs[tp_size] = DeviceConfig.from_dict(tp_size, device_data)

        return cls(
            name=name,
            architecture=architecture,
            static_batching_configs=static_configs,
            continuous_batching_configs=continuous_configs,
            device_configs=device_configs,
        )


# Made with Bob

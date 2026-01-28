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
    """Continuous batching configuration with optional device config.

    Attributes:
        tp_size: Tensor parallel size
        max_model_len: Maximum model length
        max_num_seqs: Maximum number of sequences
        device_config: Optional device-specific configuration (nested)
    """

    tp_size: int
    max_model_len: int
    max_num_seqs: int
    device_config: "DeviceConfig | None" = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ContinuousBatchingConfig":
        """Create ContinuousBatchingConfig from dictionary.

        The device_config field is optional and will be None if not present.
        """
        device_config = None
        if "device_config" in data:
            # tp_size is inherited from parent config
            device_config = DeviceConfig.from_dict(
                tp_size=data["tp_size"], data=data["device_config"]
            )

        return cls(
            tp_size=data["tp_size"],
            max_model_len=data["max_model_len"],
            max_num_seqs=data["max_num_seqs"],
            device_config=device_config,
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
            (each may have its own device_config)
    """

    name: str
    architecture: ArchitecturePattern
    static_batching_configs: list[StaticBatchingConfig] = field(default_factory=list)
    continuous_batching_configs: list[ContinuousBatchingConfig] = field(default_factory=list)

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

        # Parse static batching configs
        static_configs = []
        for cfg in data.get("static_batching_configs", []):
            static_configs.append(StaticBatchingConfig.from_dict(cfg))

        # Parse continuous batching configs (with nested device configs)
        continuous_configs = []
        for cfg in data.get("continuous_batching_configs", []):
            continuous_configs.append(ContinuousBatchingConfig.from_dict(cfg))

        # Validate no duplicate CB configs
        cb_signatures = set()
        for cfg in continuous_configs:
            signature = (cfg.tp_size, cfg.max_model_len, cfg.max_num_seqs)
            if signature in cb_signatures:
                raise ValueError(
                    f"Duplicate runtime configuration for model '{name}': "
                    f"tp_size={cfg.tp_size}, max_model_len={cfg.max_model_len}, "
                    f"max_num_seqs={cfg.max_num_seqs}"
                )
            cb_signatures.add(signature)

        # Validate no duplicate static configs
        static_signatures = set()
        for cfg in static_configs:
            # Sort warmup shapes for comparison (order shouldn't matter)
            signature = (cfg.tp_size, tuple(sorted(cfg.warmup_shapes)))
            if signature in static_signatures:
                raise ValueError(
                    f"Duplicate runtime configuration for model '{name}': "
                    f"tp_size={cfg.tp_size}, warmup_shapes={cfg.warmup_shapes}"
                )
            static_signatures.add(signature)

        # Validate at least one runtime configuration exists
        if not static_configs and not continuous_configs:
            raise ValueError(
                f"Model '{name}' must have at least one runtime configuration "
                f"(either static_batching_configs or continuous_batching_configs)"
            )

        return cls(
            name=name,
            architecture=architecture,
            static_batching_configs=static_configs,
            continuous_batching_configs=continuous_configs,
        )


# Made with Bob

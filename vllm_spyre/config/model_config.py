"""Data structures for model configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArchitecturePattern:
    """Pattern for matching model architectures.
    
    Attributes:
        model_type: The model type (e.g., 'granite', 'llama', 'roberta')
        num_hidden_layers: Number of hidden layers (optional)
        max_position_embeddings: Maximum position embeddings (optional)
        hidden_size: Hidden size dimension (optional)
        vocab_size: Vocabulary size (optional)
        num_key_value_heads: Number of key-value heads (optional)
        num_attention_heads: Number of attention heads (optional)
        num_experts_per_tok: Number of experts per token for MoE models (optional)
    """

    model_type: str
    num_hidden_layers: int | None = None
    max_position_embeddings: int | None = None
    hidden_size: int | None = None
    vocab_size: int | None = None
    num_key_value_heads: int | None = None
    num_attention_heads: int | None = None
    num_experts_per_tok: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArchitecturePattern":
        """Create ArchitecturePattern from dictionary."""
        return cls(
            model_type=data["model_type"],
            num_hidden_layers=data.get("num_hidden_layers"),
            max_position_embeddings=data.get("max_position_embeddings"),
            hidden_size=data.get("hidden_size"),
            vocab_size=data.get("vocab_size"),
            num_key_value_heads=data.get("num_key_value_heads"),
            num_attention_heads=data.get("num_attention_heads"),
            num_experts_per_tok=data.get("num_experts_per_tok"),
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
class RuntimeConfig:
    """Runtime configuration (from supported_configs.yaml).
    
    Attributes:
        cb: Whether continuous batching is enabled
        tp_size: Tensor parallel size
        max_model_len: Maximum model length (for CB mode)
        max_num_seqs: Maximum number of sequences (for CB mode)
        warmup_shapes: Warmup shapes as (prompt_length, new_tokens, batch_size) tuples
    """

    cb: bool
    tp_size: int
    max_model_len: int = 0
    max_num_seqs: int = 0
    warmup_shapes: list[tuple[int, int, int]] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuntimeConfig":
        """Create RuntimeConfig from dictionary."""
        warmup_shapes = None
        if "warmup_shapes" in data:
            # Convert list of lists to list of tuples
            warmup_shapes = [tuple(ws) for ws in data["warmup_shapes"]]

        return cls(
            cb=data["cb"],
            tp_size=data["tp_size"],
            max_model_len=data.get("max_model_len", 0),
            max_num_seqs=data.get("max_num_seqs", 0),
            warmup_shapes=warmup_shapes,
        )


@dataclass
class ModelConfig:
    """Complete model configuration.
    
    Attributes:
        name: Model name/identifier
        architecture: Architecture pattern for matching
        supported_configs: List of supported runtime configurations
        device_configs: Device-specific configurations keyed by TP size
    """

    name: str
    architecture: ArchitecturePattern
    supported_configs: list[RuntimeConfig]
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
        # Parse architecture
        architecture = ArchitecturePattern.from_dict(data["architecture"])

        # Parse supported configs
        supported_configs = [
            RuntimeConfig.from_dict(cfg) for cfg in data.get("supported_configs", [])
        ]

        # Parse device configs
        device_configs = {}
        for tp_size_str, device_data in data.get("device_configs", {}).items():
            tp_size = int(tp_size_str)
            device_configs[tp_size] = DeviceConfig.from_dict(tp_size, device_data)

        return cls(
            name=name,
            architecture=architecture,
            supported_configs=supported_configs,
            device_configs=device_configs,
        )

# Made with Bob

"""Tests for model configuration data structures."""

import pytest

from vllm_spyre.config.model_config import (
    ArchitecturePattern,
    ContinuousBatchingConfig,
    DeviceConfig,
    ModelConfig,
    StaticBatchingConfig,
)

pytestmark = pytest.mark.skip_global_cleanup


class TestArchitecturePattern:
    """Tests for ArchitecturePattern dataclass."""

    def test_create_minimal_pattern(self):
        """Test creating pattern with only required fields."""
        pattern = ArchitecturePattern(model_name="test-model", model_type="llama")
        assert pattern.model_name == "test-model"
        assert pattern.model_type == "llama"
        assert pattern.attributes == {}

    def test_create_full_pattern(self):
        """Test creating pattern with all fields."""
        quant_config = {"format": "float-quantized"}
        attributes = {
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "hidden_size": 4096,
            "vocab_size": 49152,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "num_experts_per_tok": 2,
            "quantization_config": quant_config,
        }
        pattern = ArchitecturePattern(
            model_name="granite-model",
            model_type="granite",
            attributes=attributes,
        )
        assert pattern.model_name == "granite-model"
        assert pattern.model_type == "granite"
        assert pattern.attributes["num_hidden_layers"] == 32
        assert pattern.attributes["max_position_embeddings"] == 4096
        assert pattern.attributes["hidden_size"] == 4096
        assert pattern.attributes["vocab_size"] == 49152
        assert pattern.attributes["num_key_value_heads"] == 8
        assert pattern.attributes["num_attention_heads"] == 32
        assert pattern.attributes["num_experts_per_tok"] == 2
        assert pattern.attributes["quantization_config"] == quant_config

    def test_from_dict_minimal(self):
        """Test creating pattern from minimal dict."""
        data = {"model_type": "roberta"}
        pattern = ArchitecturePattern.from_dict("roberta-model", data)
        assert pattern.model_name == "roberta-model"
        assert pattern.model_type == "roberta"
        assert pattern.attributes == {}

    def test_from_dict_full(self):
        """Test creating pattern from complete dict."""
        data = {
            "model_type": "granite",
            "num_hidden_layers": 32,
            "max_position_embeddings": 4096,
            "hidden_size": 4096,
            "vocab_size": 49152,
            "num_key_value_heads": 8,
            "num_attention_heads": 32,
            "num_experts_per_tok": 2,
            "quantization_config": {"format": "float-quantized"},
        }
        pattern = ArchitecturePattern.from_dict("granite-model", data)
        assert pattern.model_name == "granite-model"
        assert pattern.model_type == "granite"
        assert pattern.attributes["num_hidden_layers"] == 32
        assert pattern.attributes["max_position_embeddings"] == 4096
        assert pattern.attributes["hidden_size"] == 4096
        assert pattern.attributes["vocab_size"] == 49152
        assert pattern.attributes["num_key_value_heads"] == 8
        assert pattern.attributes["num_attention_heads"] == 32
        assert pattern.attributes["num_experts_per_tok"] == 2
        assert pattern.attributes["quantization_config"] == {"format": "float-quantized"}


class TestDeviceConfig:
    """Tests for DeviceConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating device config with only required fields."""
        config = DeviceConfig(tp_size=1)
        assert config.tp_size == 1
        assert config.env_vars == {}
        assert config.num_gpu_blocks_override is None
        assert config.chunked_prefill_config is None

    def test_create_with_env_vars(self):
        """Test creating device config with environment variables."""
        env_vars = {"SOME_ENV_VAR": "some_value"}
        config = DeviceConfig(tp_size=2, env_vars=env_vars)
        assert config.tp_size == 2
        assert config.env_vars == env_vars

    def test_create_with_gpu_blocks_int(self):
        """Test creating device config with integer GPU blocks override."""
        config = DeviceConfig(tp_size=1, num_gpu_blocks_override=1000)
        assert config.num_gpu_blocks_override == 1000

    def test_create_with_gpu_blocks_dict(self):
        """Test creating device config with dict GPU blocks override."""
        blocks_config = {"default": 1000, "torch_sendnn_lt_1_0_3": 800}
        config = DeviceConfig(tp_size=1, num_gpu_blocks_override=blocks_config)
        assert config.num_gpu_blocks_override == blocks_config

    def test_create_with_chunked_prefill(self):
        """Test creating device config with chunked prefill config."""
        cp_config = {"max_num_batched_tokens": 512}
        config = DeviceConfig(tp_size=1, chunked_prefill_config=cp_config)
        assert config.chunked_prefill_config == cp_config

    def test_from_dict_minimal(self):
        """Test creating device config from minimal dict."""
        config = DeviceConfig.from_dict(tp_size=1, data={})
        assert config.tp_size == 1
        assert config.env_vars == {}

    def test_from_dict_full(self):
        """Test creating device config from complete dict."""
        data = {
            "env_vars": {"SOME_ENV_VAR": "some_value"},
            "num_gpu_blocks_override": {"default": 1000, "torch_sendnn_lt_1_0_3": 800},
            "chunked_prefill_config": {"max_num_batched_tokens": 512},
        }
        config = DeviceConfig.from_dict(tp_size=2, data=data)
        assert config.tp_size == 2
        assert config.env_vars == data["env_vars"]
        assert config.num_gpu_blocks_override == data["num_gpu_blocks_override"]
        assert config.chunked_prefill_config == data["chunked_prefill_config"]


class TestStaticBatchingConfig:
    """Tests for StaticBatchingConfig dataclass."""

    def test_create_config(self):
        """Test creating static batching config."""
        warmup_shapes = [(64, 20, 4), (128, 40, 2)]
        config = StaticBatchingConfig(tp_size=1, warmup_shapes=warmup_shapes)
        assert config.tp_size == 1
        assert config.warmup_shapes == warmup_shapes

    def test_from_dict(self):
        """Test creating static batching config from dict."""
        data = {
            "tp_size": 1,
            "warmup_shapes": [[64, 20, 4], [128, 40, 2]],
        }
        config = StaticBatchingConfig.from_dict(data)
        assert config.tp_size == 1
        assert config.warmup_shapes == [(64, 20, 4), (128, 40, 2)]


class TestContinuousBatchingConfig:
    """Tests for ContinuousBatchingConfig dataclass."""

    def test_create_config(self):
        """Test creating continuous batching config."""
        config = ContinuousBatchingConfig(tp_size=1, max_model_len=2048, max_num_seqs=256)
        assert config.tp_size == 1
        assert config.max_model_len == 2048
        assert config.max_num_seqs == 256

    def test_from_dict(self):
        """Test creating continuous batching config from dict."""
        data = {
            "tp_size": 2,
            "max_model_len": 4096,
            "max_num_seqs": 128,
        }
        config = ContinuousBatchingConfig.from_dict(data)
        assert config.tp_size == 2
        assert config.max_model_len == 4096
        assert config.max_num_seqs == 128


class TestModelConfig:
    """Tests for ModelConfig dataclass."""

    def test_create_minimal_config(self):
        """Test creating model config with minimal fields."""
        architecture = ArchitecturePattern(model_name="test-model", model_type="llama")
        config = ModelConfig(name="test-model", architecture=architecture)
        assert config.name == "test-model"
        assert config.architecture == architecture
        assert config.static_batching_configs == []
        assert config.continuous_batching_configs == []
        assert config.device_configs == {}

    def test_create_with_configs(self):
        """Test creating model config with configurations."""
        architecture = ArchitecturePattern(model_name="granite-model", model_type="granite")
        static_config = StaticBatchingConfig(tp_size=1, warmup_shapes=[(64, 20, 4)])
        cb_config = ContinuousBatchingConfig(tp_size=1, max_model_len=2048, max_num_seqs=256)
        device_config = DeviceConfig(tp_size=1, env_vars={"TEST": "value"})

        config = ModelConfig(
            name="granite-model",
            architecture=architecture,
            static_batching_configs=[static_config],
            continuous_batching_configs=[cb_config],
            device_configs={1: device_config},
        )
        assert config.name == "granite-model"
        assert len(config.static_batching_configs) == 1
        assert len(config.continuous_batching_configs) == 1
        assert 1 in config.device_configs
        assert config.device_configs[1] == device_config

    def test_from_dict_minimal(self):
        """Test creating model config from minimal dict."""
        data = {
            "architecture": {"model_type": "llama"},
            "supported_configs": [],
        }
        config = ModelConfig.from_dict(name="test-model", data=data)
        assert config.name == "test-model"
        assert config.architecture.model_type == "llama"
        assert config.architecture.model_name == "test-model"
        assert config.static_batching_configs == []
        assert config.continuous_batching_configs == []
        assert config.device_configs == {}

    def test_from_dict_with_supported_configs(self):
        """Test creating model config from dict with supported configs."""
        data = {
            "architecture": {"model_type": "granite"},
            "supported_configs": [
                {"tp_size": 1, "warmup_shapes": [[64, 20, 4]]},
                {"tp_size": 1, "max_model_len": 2048, "max_num_seqs": 256},
            ],
        }
        config = ModelConfig.from_dict(name="granite-model", data=data)
        assert len(config.static_batching_configs) == 1
        assert len(config.continuous_batching_configs) == 1
        assert config.static_batching_configs[0].tp_size == 1
        assert config.static_batching_configs[0].warmup_shapes == [(64, 20, 4)]
        assert config.continuous_batching_configs[0].tp_size == 1
        assert config.continuous_batching_configs[0].max_model_len == 2048

    def test_from_dict_with_device_configs(self):
        """Test creating model config from dict with device configs."""
        data = {
            "architecture": {"model_type": "granite"},
            "supported_configs": [],
            "device_configs": {
                "1": {
                    "env_vars": {"VLLM_ATTENTION_BACKEND": "FLASH_ATTN"},
                    "num_gpu_blocks_override": 1000,
                },
                "2": {
                    "env_vars": {"VLLM_ATTENTION_BACKEND": "XFORMERS"},
                    "num_gpu_blocks_override": 500,
                },
            },
        }
        config = ModelConfig.from_dict(name="granite-model", data=data)
        assert 1 in config.device_configs
        assert 2 in config.device_configs
        assert config.device_configs[1].tp_size == 1
        assert config.device_configs[2].tp_size == 2
        assert config.device_configs[1].num_gpu_blocks_override == 1000
        assert config.device_configs[2].num_gpu_blocks_override == 500

    def test_from_dict_complete(self):
        """Test creating model config from complete dict."""
        data = {
            "architecture": {
                "model_type": "granite",
                "num_hidden_layers": 32,
                "hidden_size": 4096,
            },
            "supported_configs": [
                {"tp_size": 1, "warmup_shapes": [[64, 20, 4]]},
                {"tp_size": 2, "max_model_len": 4096, "max_num_seqs": 128},
            ],
            "device_configs": {
                "1": {
                    "env_vars": {"TEST_VAR": "test_value"},
                    "num_gpu_blocks_override": {"default": 1000, "torch_sendnn_lt_1_0_3": 800},
                    "chunked_prefill_config": {"max_num_batched_tokens": 512},
                },
            },
        }
        config = ModelConfig.from_dict(name="granite-8b", data=data)
        assert config.name == "granite-8b"
        assert config.architecture.model_type == "granite"
        assert config.architecture.model_name == "granite-8b"
        assert config.architecture.attributes["num_hidden_layers"] == 32
        assert len(config.static_batching_configs) == 1
        assert len(config.continuous_batching_configs) == 1
        assert config.static_batching_configs[0].warmup_shapes == [(64, 20, 4)]
        assert config.continuous_batching_configs[0].max_model_len == 4096
        assert 1 in config.device_configs
        assert config.device_configs[1].env_vars == {"TEST_VAR": "test_value"}
        assert config.device_configs[1].chunked_prefill_config == {"max_num_batched_tokens": 512}


# Made with Bob

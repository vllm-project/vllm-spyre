# Model Configuration System

This directory contains the new model configuration system for vLLM-Spyre, which provides a clean, extensible way to manage model-specific configurations.

## Overview

The configuration system replaces hard-coded model logic with a declarative YAML-based approach, making it easy to add new models and maintain existing configurations.

### Key Components

1. **`model_configs.yaml`**: Declarative model definitions with architecture patterns, runtime configs, and device settings
2. **`model_registry.py`**: Singleton registry that loads and manages model configurations
3. **`model_matcher.py`**: Pattern-based matching to identify models from HuggingFace configs
4. **`model_config.py`**: Data structures (ModelConfig, ArchitecturePattern, DeviceConfig, RuntimeConfig)
5. **`configurators/`**: Code-based logic for applying configurations
   - `base.py`: Abstract base class with helper methods
   - `default.py`: Universal configurator handling all model types

## Architecture

```
┌─────────────────────┐
│  model_configs.yaml │  ← Declarative configuration
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ModelConfigRegistry│  ← Singleton registry
│  - Loads YAML       │
│  - Matches models   │
│  - Creates config   │
└──────────┬──────────┘
           │
           ├─► ModelMatcher (pattern matching)
           │
           └─► DefaultConfigurator (apply configs)
                    │
                    ├─► Environment variables
                    ├─► GPU block overrides
                    └─► Chunked prefill settings
```

## Adding a New Model

Simply add an entry to `model_configs.yaml`:

```yaml
models:
  your-org/your-model:
    architecture:
      model_type: llama  # Required: HF model type
      num_hidden_layers: 32  # Optional: for precise matching
      vocab_size: 128256  # Optional: for precise matching
      # Add other HF config attributes as needed

    supported_configs:
      # Continuous batching configuration
      - cb: true
        tp_size: 1
        max_model_len: 8192
        max_num_seqs: 16

      # Static batching configuration
      - cb: false
        tp_size: 1
        warmup_shapes: [[2048, 1024, 16]]  # [prompt_len, new_tokens, batch_size]

    # Optional: device-specific configurations
    device_configs:
      4:  # Tensor parallel size 4
        env_vars:
          VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
          FLEX_HDMA_P2PSIZE: 268435456
        num_gpu_blocks_override:
          torch_sendnn_lt_1_0_3: 2080  # Version-specific override
          default: 8192
        chunked_prefill_config:
          max_num_batched_tokens: 1024
```

**That's it!** No code changes needed for most models.

## Configuration Schema

### Architecture Pattern

Defines how to match a model from its HuggingFace config. Only `model_type` is required; other fields are optional and used for more precise matching.

```yaml
architecture:
  model_type: granite          # Required: HF model type
  num_hidden_layers: 40        # Optional: number of layers
  max_position_embeddings: 131072  # Optional: max positions
  hidden_size: 4096            # Optional: hidden dimension
  vocab_size: 49159            # Optional: vocabulary size
  num_key_value_heads: 8       # Optional: KV heads
  num_attention_heads: 32      # Optional: attention heads
  num_experts_per_tok: 0       # Optional: for MoE models
```

### Runtime Configurations

Defines supported runtime modes (continuous batching or static batching):

```yaml
supported_configs:
  # Continuous batching mode
  - cb: true
    tp_size: 4
    max_model_len: 32768
    max_num_seqs: 32

  # Static batching mode
  - cb: false
    tp_size: 1
    warmup_shapes:
      - [2048, 1024, 16]  # prompt_length, new_tokens, batch_size
      - [4096, 512, 8]
```

### Device Configurations

Optional device-specific settings keyed by tensor parallel size:

```yaml
device_configs:
  4:  # TP size 4
    # Environment variables to set
    env_vars:
      VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
      FLEX_HDMA_P2PSIZE: 268435456

    # GPU block override (simple or version-aware)
    num_gpu_blocks_override: 8192  # Simple override
    # OR
    num_gpu_blocks_override:
      torch_sendnn_lt_1_0_3: 2080  # For torch_sendnn < 1.0.3
      default: 8192                 # For other versions

    # Chunked prefill configuration
    chunked_prefill_config:
      max_num_batched_tokens: 1024
```

## How It Works

### 1. Model Matching

When vLLM loads a model, the registry:
1. Extracts the HuggingFace config
2. Compares it against all registered architecture patterns
3. Finds the matching model configuration

```python
# In platform.py
from vllm_spyre.config.model_registry import get_model_registry

registry = get_model_registry()
model_name = registry.find_matching_model(model_config)
```

### 2. Configuration Application

Once matched, the configurator:
1. Gets the device config for the current TP size
2. Applies environment variables
3. Configures GPU blocks (with version awareness)
4. Sets up chunked prefill (if applicable)

```python
if model_name:
    configurator = registry.get_configurator(model_name)
    configurator.configure(vllm_config)
```

### 3. Optional Features

All device configuration features are optional:
- **No device_configs**: Model works with defaults
- **No env_vars**: No environment variables set
- **No num_gpu_blocks_override**: Uses vLLM's default calculation
- **No chunked_prefill_config**: Uses command-line settings

## API Reference

### ModelConfigRegistry

```python
from vllm_spyre.config.model_registry import get_model_registry

registry = get_model_registry()

# Find model by HF config
model_name = registry.find_matching_model(vllm_model_config)

# Get model configuration
model_config = registry.get_model_config(model_name)

# Get configurator
configurator = registry.get_configurator(model_name)

# List all registered models
models = registry.list_models()
```

### DefaultConfigurator

The universal configurator handles all models:

```python
class DefaultConfigurator(ModelConfigurator):
    def configure(self, vllm_config: VllmConfig) -> None:
        """Apply device configurations"""

    def _configure_gpu_blocks(self, device_config, vllm_config, tp_size) -> None:
        """Configure GPU blocks with version-aware logic"""

    def _configure_chunked_prefill(self, device_config, vllm_config, tp_size) -> None:
        """Configure chunked prefill settings"""
```

Helper methods from base class:

```python
# Set environment variable with logging
self.set_env_var("KEY", "value", override=False)

# Get device config for TP size
device_config = self.get_device_config(tp_size)
```

## Migration from Legacy Code

### Before (Hard-coded in platform.py)

```python
def configure_granite_3_8b(cls, vllm_config: VllmConfig):
    if parallel_config.world_size != 4:
        return

    tkv_128k = 128 * 1024
    if not os.getenv("VLLM_DT_MAX_BATCH_TKV_LIMIT"):
        os.environ["VLLM_DT_MAX_BATCH_TKV_LIMIT"] = str(tkv_128k)
    # ... more hard-coded logic
```

### After (Declarative YAML)

```yaml
ibm-granite/granite-3.3-8b-instruct:
  device_configs:
    4:
      env_vars:
        VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
```

## Benefits

1. **Extensibility**: Add new models by editing YAML only
2. **Maintainability**: Centralized configuration, no scattered code
3. **Testability**: Easy to test configurations in isolation
4. **Documentation**: YAML serves as self-documenting configuration
5. **Simplicity**: Single configurator handles all models
6. **Flexibility**: Optional features can be omitted

## Testing

Unit tests should cover:

```bash
# Test model matching
pytest tests/config/test_model_matcher.py

# Test registry functionality
pytest tests/config/test_model_registry.py

# Test configurator logic
pytest tests/config/test_configurators.py

# Integration tests
pytest tests/config/test_integration.py
```

## Future Enhancements

Potential improvements:
- YAML schema validation
- Configuration inheritance (model families)
- Dynamic model registration API
- Configuration versioning
- Performance profiling per configuration

## Examples

### Simple Embedding Model

```yaml
sentence-transformers/all-roberta-large-v1:
  architecture:
    model_type: roberta
    num_hidden_layers: 24
    vocab_size: 50265

  supported_configs:
    - cb: false
      tp_size: 1
      warmup_shapes: [[128, 0, 8]]
```

### Complex Generation Model

```yaml
ibm-granite/granite-3.3-8b-instruct:
  architecture:
    model_type: granite
    num_hidden_layers: 40
    max_position_embeddings: 131072
    hidden_size: 4096
    vocab_size: 49159
    num_key_value_heads: 8
    num_attention_heads: 32

  supported_configs:
    - cb: true
      tp_size: 4
      max_model_len: 32768
      max_num_seqs: 32

  device_configs:
    4:
      env_vars:
        VLLM_DT_MAX_BATCH_TKV_LIMIT: 131072
        FLEX_HDMA_P2PSIZE: 268435456
      num_gpu_blocks_override:
        torch_sendnn_lt_1_0_3: 2080
        default: 8192
      chunked_prefill_config:
        max_num_batched_tokens: 1024
```

## Troubleshooting

### Model Not Matched

If your model isn't being matched:
1. Check the HF config attributes: `print(model_config.hf_config.__dict__)`
2. Ensure `model_type` matches exactly
3. Add more specific attributes to narrow the match
4. Check logs for matching attempts

### Configuration Not Applied

If configuration isn't being applied:
1. Verify the TP size matches a device_config key
2. Check that environment variables aren't already set
3. Review logs for configuration application messages
4. Ensure YAML syntax is correct

### Version-Specific Issues

For version-aware configurations:
1. Check torch_sendnn version: `SpyrePlatform.sendnn_version()`
2. Verify version comparison logic in configurator
3. Add logging to debug version selection

## Support

For questions or issues:
1. Check this README
2. Review existing model configurations in `model_configs.yaml`
3. Examine the configurator code in `configurators/default.py`
4. Consult the vLLM-Spyre team

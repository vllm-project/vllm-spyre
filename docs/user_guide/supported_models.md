# Supported Models

The vLLM Spyre plugin relies on model code implemented by the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack/tree/main/fms/models).

## Configurations

The following models have been verified to run on vLLM Spyre with the listed
configurations.

### Decoder Models

| Model                     | AIUs | Context Length | Batch Size |
|---------------------------|------|----------------|------------|
| [Granite-3.3-8b][]        | 1    | 3072           | 16         |
| [Granite-3.3-8b][]        | 4    | 32768          | 32         |
| [Granite-3.3-8b (FP8)][]  | 1    | 3072           | 16         |
| [Granite-3.3-8b (FP8)][]  | 4    | 32768          | 32         |
| [Llama-3.1-8B-Instruct][] | 1    | 3072           | 16         |
| [Llama-3.1-8B-Instruct][] | 4    | 32768          | 32         |

### Encoder Models

| Model                                     | AIUs | Context Length | Batch Size |
|-------------------------------------------|------|----------------|------------|
| [Granite-Embedding-125m (English)][]      | 1    | 512            | 1          |
| [Granite-Embedding-125m (English)][]      | 1    | 512            | 64         |
| [Granite-Embedding-278m (Multilingual)][] | 1    | 512            | 1          |
| [Granite-Embedding-278m (Multilingual)][] | 1    | 512            | 64         |
| [BAAI/BGE-Reranker (v2-m3)][]             | 1    | 8192           | 1          |
| [BAAI/BGE-Reranker (Large)][]             | 1    | 512            | 1          |
| [BAAI/BGE-Reranker (Large)][]             | 1    | 512            | 64         |
| [Multilingual-E5-large][]                 | 1    | 512            | 64         |

[Granite-3.3-8b]: https://huggingface.co/ibm-granite/granite-3.3-8b-instruct
[Llama-3.1-8B-Instruct]: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
[Granite-3.3-8b (FP8)]: https://huggingface.co/ibm-granite/granite-3.3-8b-instruct-FP8
[Granite-Embedding-125m (English)]: https://huggingface.co/ibm-granite/granite-embedding-125m-english
[Granite-Embedding-278m (Multilingual)]: https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual
[BAAI/BGE-Reranker (v2-m3)]: https://huggingface.co/BAAI/bge-reranker-v2-m3
[BAAI/BGE-Reranker (Large)]: https://huggingface.co/BAAI/bge-reranker-large
[Multilingual-E5-large]: https://huggingface.co/intfloat/multilingual-e5-large

## Model Configuration

The Spyre engine uses a model registry to manage model-specific configurations. Model configurations
are defined in <gh-file:vllm_spyre/config/model_configs.yaml> and include:

- Architecture patterns for model matching
- Device-specific configurations (environment variables, GPU block overrides)
- Supported runtime configurations (static batching warmup shapes, continuous batching parameters)

When a model is loaded, the registry automatically matches it to the appropriate configuration and
applies model-specific settings.

### Supported Configurations

The following configurations are supported for each model:

```yaml
--8<-- "vllm_spyre/config/model_configs.yaml:supported-model-runtime-configuration"
```

### Configuration Validation

By default, the Spyre engine will log warnings if a requested model or configuration is not found
in the registry. To enforce strict validation and fail if an unknown configuration is requested,
set the environment variable:

```bash
export VLLM_SPYRE_REQUIRE_KNOWN_CONFIG=1
```

When this flag is enabled, the engine will raise a `RuntimeError` if:

- The model cannot be matched to a known configuration
- The requested runtime parameters are not in the supported configurations list

See the [Configuration Guide](configuration.md) for more details on model configuration.

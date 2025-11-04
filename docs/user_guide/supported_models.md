# Supported Models

The vLLM Spyre plugin relies on model code implemented by the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack/tree/main/fms/models).

## Configurations

The following models have been verified to run on vLLM Spyre with the listed
configurations.

### Decoder Models

**_Static Batching:_**

| Model                | AIUs | Prompt Length | New Tokens | Batch Size |
|----------------------|------|---------------|------------|------------|
| [Granite-3.3-8b][]   | 4    | 7168          | 1024       | 4          |

**_Continuous Batching:_**

| Model                     | AIUs | Context Length | Batch Size |
|---------------------------|------|----------------|------------|
| [Granite-3.3-8b][]        | 1    | 3072           | 16         |
| [Granite-3.3-8b][]        | 4    | 32768          | 32         |
| [Granite-3.3-8b (FP8)][]  | 1    | 3072           | 16         |
| [Granite-3.3-8b (FP8)][]  | 4    | 32768          | 32         |

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
[Granite-3.3-8b (FP8)]: https://huggingface.co/ibm-granite/granite-3.3-8b-instruct-FP8
[Granite-Embedding-125m (English)]: https://huggingface.co/ibm-granite/granite-embedding-125m-english
[Granite-Embedding-278m (Multilingual)]: https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual
[BAAI/BGE-Reranker (v2-m3)]: https://huggingface.co/BAAI/bge-reranker-v2-m3
[BAAI/BGE-Reranker (Large)]: https://huggingface.co/BAAI/bge-reranker-large
[Multilingual-E5-large]: https://huggingface.co/intfloat/multilingual-e5-large

## Runtime Validation

At runtime, the Spyre engine validates the requested model and configurations against the list
of supported models and configurations based on the entries in the file
<gh-file:vllm_spyre/config/supported_configs.yaml>. If a requested model or configuration
is not found, a warning message will be logged.

```python
--8<-- "vllm_spyre/config/supported_configs.yaml:supported-model-runtime-configurations"
```

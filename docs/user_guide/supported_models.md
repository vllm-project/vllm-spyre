# Supported Models

The vLLM Spyre plugin relies on model code implemented by the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack/tree/main/fms/models).

| Model Family | Supported |
| ------------ | --------- |
| Granite      | ✅        |
| LLaMA        | ✅        |
| RoBERTa      | ✅        |

## Configurations for Pre-compiled Models

The following pre-compiled models have been verified to run on vLLM Spyre with
the listed configurations.

### Granite-3.3-8b-instruct (Precision: 16 bit)

- **Static Batching**:

| Platform  | AIUs | Prompt Length | New Tokens | Batch Size | Use Case |
|-----------|------|---------------|------------|------------|----------|
| `ppc64le` | 1    | 2048          | 1024       | 16         | EE       |
| `ppc64le` | 4    | 6144          | 2048       | 1          | RAG      |
| `s390x`   | 4    | 7168          | 1024       | 1          | RAG      |

- **Continuous Batching**:

| Platform  | AIUs | Context Length | Batch Size | Comments         |
|-----------|------|----------------|------------|------------------|
| `amd64`   | 1    | 8192           | 4          |                  |
| `amd64`   | 2    | 8192           | 4          |                  |
| `amd64`   | 4    | 16384          | 4          | `FLEX_DEVICE=PF` |
| `amd64`   | 4    | 8192           | 4          |                  |
| `amd64`   | 4    | 8192           | 4          |                  |
| `s390x`   | 1    | 8192           | 4          |                  |
| `s390x`   | 2    | 8192           | 4          |                  |
| `s390x`   | 4    | 16384          | 4          | `FLEX_DEVICE=PF` |
| `s390x`   | 4    | 16384          | 4          | `FLEX_DEVICE=VF` |
| `s390x`   | 4    | 8192           | 4          |                  |
| `s390x`   | 4    | 8192           | 4          |                  |
| `s390x`   | 4    | 8192           | 4          | `FLEX_DEVICE=PF` |

### Granite-3.3-8b-instruct-FP8 (Precision: 8 bit)

- **Continuous Batching**:

| Platform | AIUs | Context Length | Batch Size | FLEX Device |
|----------|------|----------------|------------|-------------|
| `amd64`  | 4    | 16384          | 4          | `PF`        |
| `s390x`  | 4    | 8192           | 4          | `PF`        |
| `s390x`  | 4    | 16384          | 4          | `PF`        |
| `s390x`  | 4    | 16384          | 4          | `VF`        |

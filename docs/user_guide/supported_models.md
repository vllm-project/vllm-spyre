# Supported Models

The vLLM Spyre plugin relies on model code implemented by the [Foundation Model Stack](https://github.com/foundation-model-stack/foundation-model-stack/tree/main/fms/models).

| Model Family | Supported |
| ------------ | --------- |
| Granite      | ✅        |
| LLaMA        | ✅        |
| RoBERTa      | ✅        |

## Configurations

The following models have been verified to run on vLLM Spyre with the listed
configurations.

### Decoder Models

**_Static Batching:_**

| Model          |  Platform  | AIUs | Prompt Length | New Tokens | Batch Size | Source (remove this column) |
|----------------|------------|------|---------------|------------|------------|-----------------------------|
| Granite-3.3-8b |  `ppc64le` | 1    | 2048          | 1024       | 16         | Gaurav 9/3                  |
| Granite-3.3-8b |  `ppc64le` | 4    | 2048          | 1024       | 16         | AIU docs                    |
| Granite-3.3-8b |  `ppc64le` | 4    | 6144          | 2048       | 1          | AIU docs                    |
| Granite-3.3-8b |  `s390x`   | 4    | 7168          | 1024       | 1          | AIU docs                    |

**_Continuous Batching:_**

| Model                | Platform  | AIUs | Context Length | Batch Size | Source (remove this column) |
|----------------------|-----------|------|----------------|------------|-----------------------------|
| Granite-3.3-8b       | `amd64`   | 4    | 8192           | 4          | AIU docs - remove?          |
| Granite-3.3-8b       | `amd64`   | 4    | 16384          | 4          | AIU docs - remove?          |
| Granite-3.3-8b       | `ppc64le` | 1    | 3072           | 16         | Gaurav 9/3 - remove?        |
| Granite-3.3-8b       | `s390x`   | 1    | 3072           | 16         | PELE                        |
| Granite-3.3-8b       | `s390x`   | 1    | 8192           | 4          | AIU docs - remove?          |
| Granite-3.3-8b       | `s390x`   | 2    | 8192           | 4          | AIU docs - remove?          |
| Granite-3.3-8b       | `s390x`   | 4    | 16384          | 4          | AIU docs - remove?          |
| Granite-3.3-8b       | `s390x`   | 4    | 8192           | 4          | AIU docs - remove?          |
| Granite-3.3-8b       | `s390x`   | 4    | 32768          | 32         | PELE                        |
| Granite-3.3-8b (FP8) | `amd64`   | 4    | 16384          | 4          | AIU docs - remove?          |
| Granite-3.3-8b (FP8) | `s390x`   | 1    | 4096           | 32         | PELE                        |
| Granite-3.3-8b (FP8) | `s390x`   | 1    | 8192           | 16         | PELE                        |
| Granite-3.3-8b (FP8) | `s390x`   | 1    | 16384          | 8          | PELE                        |
| Granite-3.3-8b (FP8) | `s390x`   | 1    | 32768          | 4          | PELE                        |
| Granite-3.3-8b (FP8) | `s390x`   | 1    | 32768          | 32         | PELE                        |
| Granite-3.3-8b (FP8) | `s390x`   | 4    | 8192           | 4          | AIU docs - remove?          |
| Granite-3.3-8b (FP8) | `s390x`   | 4    | 16384          | 4          | AIU docs - remove?          |

### Encoder Models

| Model                               | Platform  | AIUs | Context Length | Batch Size | Source (remove this column) |
|-------------------------------------|-----------|------|----------------|------------|-----------------------------|
| Granite-Embedding-125m-English      | `s390x`   | 1    | 512            | 1          | PELE                        |
| Granite-Embedding-125m-English      | `s390x`   | 1    | 512            | 64         | PELE                        |
| granite-embedding-278m-multilingual | `s390x`   | 1    | 512            | 1          | PELE                        |
| granite-embedding-278m-multilingual | `s390x`   | 1    | 512            | 64         | PELE                        |
| BAAI/bge-reranker-v2-m3             | `s390x`   | 1    | 2048           | 1          | PELE                        |
| BAAI/bge-reranker-v2-m3             | `s390x`   | 1    | 4096           | 1          | PELE                        |
| BAAI/bge-reranker-v2-m3             | `s390x`   | 1    | 8192           | 1          | PELE                        |
| BAAI/bge-reranker-large             | `s390x`   | 1    | 512            | 1          | PELE                        |
| BAAI/bge-reranker-large             | `s390x`   | 1    | 512            | 64         | PELE                        |

## Model Files

| Model                               | Download                                                                           |
|-------------------------------------|------------------------------------------------------------------------------------|
| Granite-3.3-8b                      | [Download](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)             |
| Granite-3.3-8b (FP8)                |                                                                                    |
| Granite-Embedding-125m-English      | [Download](https://huggingface.co/ibm-granite/granite-embedding-125m-english)      |
| Granite-Embedding-278m-Multilingual | [Download](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) |
| BAAI/bge-reranker-v2-m3             | [Download](https://huggingface.co/BAAI/bge-reranker-v2-m3)                         |
| BAAI/bge-reranker-large             | [Download](https://huggingface.co/BAAI/bge-reranker-large)                         |

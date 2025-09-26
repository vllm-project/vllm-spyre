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

| Model          | AIUs | Prompt Length | New Tokens | Batch Size |
|----------------|------|---------------|------------|------------|
| Granite-3.3-8b | 4    | 7168          | 1024       | 4          |

**_Continuous Batching:_**

| Model                | AIUs | Context Length | Batch Size |
|----------------------|------|----------------|------------|
| Granite-3.3-8b       | 1    | 3072           | 16         |
| Granite-3.3-8b       | 4    | 32768          | 32         |
| Granite-3.3-8b (FP8) | 1    | 3072           | 16         |
| Granite-3.3-8b (FP8) | 4    | 32768          | 32         |


### Encoder Models

| Model                   | AIUs | Context Length | Batch Size |
|-------------------------|------|----------------|------------|
| Granite-Emb-125m-Eng    | 1    | 512            | 1          |
| Granite-Emb-125m-Eng    | 1    | 512            | 64         |
| Granite-Emb-278m-Multi  | 1    | 512            | 1          |
| Granite-Emb-278m-Multi  | 1    | 512            | 64         |
| BAAI/BGE-Reranker-v2-m3 | 1    | 8192           | 1          |
| BAAI/BGE-Reranker-Large | 1    | 512            | 1          |
| BAAI/BGE-Reranker-Large | 1    | 512            | 64         |

## Model Files

| Model                   | Download                                                                           |
|-------------------------|------------------------------------------------------------------------------------|
| Granite-3.3-8b          | [Download](https://huggingface.co/ibm-granite/granite-3.3-8b-instruct)             |
| Granite-3.3-8b (FP8)    |                                                                                    |
| Granite-Emb-125m-Eng    | [Download](https://huggingface.co/ibm-granite/granite-embedding-125m-english)      |
| Granite-Emb-278m-Multi  | [Download](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) |
| BAAI/BGE-Reranker-v2-m3 | [Download](https://huggingface.co/BAAI/bge-reranker-v2-m3)                         |
| BAAI/BGE-Reranker-Large | [Download](https://huggingface.co/BAAI/bge-reranker-large)                         |

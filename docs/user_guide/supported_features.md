# Supported Features

This table summarize the status of features on Spyre. By default, those features are planned to be developed using vLLM engine V1.

| Feature                       | Status | Note |
|-------------------------------|--------|------|
| Chunked Prefill               |   🗓️   |      |
| Automatic Prefix Caching      |   🗓️   |      |
| LoRA                          |   🗓️   |      |
| Prompt Adapter                |   ⛔   | Being deprecated in vLLM [vllm#13981](https://github.com/vllm-project/vllm/issues/13981) |
| Speculative Decoding          |   🗓️   |      |
| Guided Decoding               |   🗓️   |      |
| Pooling                       |   ⚠️   | Works with V0. V1 still being developed in vLLM [vllm#18052](https://github.com/vllm-project/vllm/issues/18052) |
| Enc-dec                       |   ⛔   | No plans for now |
| Multi Modality                |   🗓️   |      |
| LogProbs                      |   ✅   |      |
| Prompt logProbs               |   🚧   |      |
| Best of                       |   ⛔   | Deprecated in vLLM [vllm#13361](https://github.com/vllm-project/vllm/issues/13361)    |
| Beam search                   |   ✅   |      |
| Tensor Parallel               |   ✅   |      |
| Pipeline Parallel             |   🗓️   |      |
| Expert Parallel               |   🗓️   |      |
| Data Parallel                 |   🗓️   |      |
| Prefill Decode Disaggregation |   🗓️   |      |
| Quantization                  |   ⚠️   |      |
| Sleep Mode                    |   🗓️   |      |

- ✅ Fully operational.
- ⚠️ Experimental support.
- 🚧 Under active development.
- 🗓️ Planned.
- ⛔ Not planned or deprecated.

# Configuration

For a complete list of configuration options, see [Environment Variables](env_vars.md).

## Backend Selection

The torch.compile backend can be configured with the `VLLM_SPYRE_DYNAMO_BACKEND` environment variable.

All models can be tested on CPU by setting this to `eager`.
To run inference on IBM Spyre Accelerators, the backend should be set as:

| Model type | vLLM backend | `VLLM_SPYRE_DYNAMO_BACKEND` configuration | Notes |
| --- | --- | --- | --- |
| Decoder | v0 | sendnn | V0 support for decoder models is deprecated |
| Decoder | v1 | sendnn | |
| Embedding | v0 | sendnn | V0 support for embedding models is deprecated|
| Embedding | v1 | sendnn | |

## Batching Modes

When running decoder models, vLLM Spyre supports a static batching mode and a continuous batching mode.

### Static Batching

With static batching, graphs are pre-compiled for the configured batch shapes and each batch must finish processing before a new batch can be scheduled. This adds extra constraints on the sizes of inputs and outputs for each request, and requests that do not fit the precompiled graphs will be rejected.

Static batching mode is enabled by default, and can be explicitly enabled by setting `VLLM_SPYRE_USE_CB=0`.

!!! caution
    There are no up-front checks that the compiled graphs will fit into the available memory on the Spyre cards. If the graphs are too large for the available memory, vllm will crash during model warmup.

The batch shapes are configured with the `VLLM_SPYRE_WARMUP_*` environment variables. For example, to warm up two graph shapes for one single large request and four smaller requests you could use:

```shell
export VLLM_SPYRE_WARMUP_BATCH_SIZES=1,4
export VLLM_SPYRE_WARMUP_PROMPT_LENS=4096,1024
export VLLM_SPYRE_WARMUP_NEW_TOKENS=1024,256
```

### Continuous Batching

!!! attention
    Continuous batching can be enabled with `VLLM_SPYRE_USE_CB=1`.

Continuous batching works much more like other accelerator implementations on vLLM. Requests can be continually appended to a running batch, and requests that finish generating can be evicted from the batch to make room for more requests. Neither chunked prefill nor prefix caching are currently supported though, so when a request is added to the running batch it must first be paused for a full prefill of the incoming prompt.

Unlike static batching, no warmup shapes need to be provided for continuous batching. While the user does not have to specify the prompt lengths explicitly (see `VLLM_SPYRE_WARMUP_PROMPT_LENS` for static batching), the vLLM argument `max-num-seqs` is used to set the maximum batch size (analogous to `VLLM_SPYRE_WARMUP_BATCH_SIZES` for static batching). The number of generated output tokens is implicitly limited by `max-model-len - padded_prompt_length` (see `VLLM_SPYRE_WARMUP_NEW_TOKENS` for static batching), where `padded_prompt_length` is the prompt length rounded up to the next multiple of the block size (64).

!!! attention
    Currently the maximal context length for which continuous batching is supported on IBM Spyre Accelerators is 32K (32,768). Therefore the length of the submitted prompts plus the number of requested output tokens should be less than 32K. We strongly recommend not setting the `max_tokens` too high, such that prompt lengths plus output tokens are well below 32K. Otherwise there is a risk of performance degradation due to scheduling constraints.

## Chunked Prefill

Chunked prefill is a technique that improves Inter-Token Latency (ITL) in continuous batching mode when large prompts need to be prefetched. Without it, these large prefills can negatively impact the performance of ongoing decodes. In essence, chunked prefill divides incoming prompts into smaller segments and processes them incrementally, allowing the system to balance prefill work with active decoding tasks.

For configuration and tuning guidance, see the [vLLM official documentation on chunked prefill](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill).

In the vLLM v1 engine, this feature is enabled by default. In vLLM-Spyre, however, users must explicitly enable it by setting the environment variable `VLLM_SPYRE_USE_CHUNKED_PREFILL=1`.

!!! note
    Chunked prefill requires continuous batching to be enabled by setting `VLLM_SPYRE_USE_CB=1`.

As in vLLM, the `max_num_batched_tokens` parameter controls how chunks are formed. However, because current versions of vLLM-Spyre cannot prefill and decode within the same engine step and only prefill a single prompt at a time, `max_num_batched_tokens` specifies the chunk size, whereas in upstream vLLM it represents a shared token budget for both prefills and decodes.

This parameter should be tuned according to your infrastructure, it is recommended to set it from `1024` to `4096` tokens and it **must** be multiple of the block size (currently fixed to `64`). For convenience, when using the model `ibm-granite/granite-3.3-8b-instruct` with `tp=4`, vLLM-Spyre automatically sets `max_num_batched_tokens` to `4096`, a value known to produce good hardware utilization in this setup.

## Caching Compiled Graphs

`torch_sendnn` supports caching compiled model graphs, which can vastly speed up warmup time when loading models in a distributed setting.

To enable this, set `TORCH_SENDNN_CACHE_ENABLE=1` and configure `TORCH_SENDNN_CACHE_DIR` to a directory to hold the cache files. By default, this feature is disabled.

### Require Precompiled Decoders

Model compilation can be resource intensive and disruptive in production environments. To mitigate this, artifacts stored in `TORCH_SENDNN_CACHE_DIR` can be persisted to a shared volume during pre-deployment. Requiring the server to load from the cache avoids unexpected recompilation on the inference server.

To enforce the use of precompiled models, set:

```sh
VLLM_SPYRE_REQUIRE_PRECOMPILED_DECODERS=1
```

and create an empty catalog file:

```sh
echo '{}' > ${TORCH_SENDNN_CACHE_DIR}/pre_compiled_cache_catalog.json
```

This configuration ensures that if a precompiled model is not found, an error will be raised. The catalog file is mandatory and serves as metadata for precompiled models in the cache. It enables the server to surface useful information and warnings in logs tagged with `[PRECOMPILED_WARN]`.

Catalog checks inspect metadata of the launch configuration that affect the cached artifacts including:

- vLLM configurations (tensor parallelism, batch size, static vs. continuous batching)
- Library versions used during precompilation
- Model name

If a matching entry is not found in the catalog, the server will still attempt to load from the cache. This allows precompiled models without catalog metadata to be used. However, if no precompiled model exists in the cache, the system will raise:

```text
RuntimeError: Compilation disabled
```

Scripts to generate and update pre_compiled_cache_catalog.json will be provided in future releases.

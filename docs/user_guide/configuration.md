# Configuration

For a complete list of configuration options, see [Environment Variables](env_vars.md).

## Backend Selection

The torch.compile backend can be configured with the `VLLM_SPYRE_DYNAMO_BACKEND` environment variable.

All models can be tested on CPU by setting this to `eager`.
To run inference on IBM Spyre Accelerators, the backend should be set as:

| Model type | vLLM backend | `VLLM_SPYRE_DYNAMO_BACKEND` configuration | Notes |
| --- | --- | --- | --- |
| Decoder | v0 | sendnn_decoder | V0 support for decoder models is deprecated |
| Decoder | v1 | sendnn_decoder | |
| Embedding | v0 | sendnn | |
| Embedding | v1 | N/A | Embedding models are not yet supported on V1 |

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
    Continuous batching is not currently supported on IBM Spyre Accelerators. A CPU-only implementation is available by setting `VLLM_SPYRE_DYNAMO_BACKEND=eager`. Continuous batching can be enabled with `VLLM_SPYRE_USE_CB=1`.

Continuous batching works much more like other accelerator implementations on vLLM. Requests can be continually appended to a running batch, and requests that finish generating can be evicted from the batch to make room for more requests. Neither chunked prefill nor prefix caching are currently supported though, so when a request is added to the running batch it must first be paused for a full prefill of the incoming prompt.

## Caching Compiled Graphs

`torch_sendnn` supports caching compiled model graphs, which can vastly speed up warmup time when loading models in a distributed setting.

To enable this, set `TORCH_SENDNN_CACHE_ENABLE=1` and configure `TORCH_SENDNN_CACHE_DIR` to a directory to hold the cache files. By default, this feature is disabled.

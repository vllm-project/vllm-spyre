# Detailed Performance Measurement

`sendnn-bench` is a drop-in replacement for [`vllm bench serve`](https://docs.vllm.ai/en/stable/benchmarking/cli/#online-benchmark) that collects **additional Spyre-specific per-request metrics** on top of the usual TTFT/TPOT/ITL/E2EL values described in [Benchmarking and Performance](./performance.md). It reuses the upstream implementation — every `vllm bench serve` flag keeps working — and augments it with scheduler-level information such as queue wait time, per-chunk prefill latencies, per-step decode latencies, prefix cache hit rate, left padding and request pausing.

## Usage

1. Start the server with Spyre metrics collection enabled:

```bash
SENDNN_INFERENCE_BENCH_METRICS_ENABLED=1 vllm serve \
    --model {model} \
    --max-model-len {max-model-len} \
    --max-num-seqs {max-num-seqs}
```

!!! warning

    `SENDNN_INFERENCE_BENCH_METRICS_ENABLED=1` must be set **on the server**. The client does not read it. Without it the server returns no Spyre metrics and the client logs a warning.

1. Run the benchmark client with `sendnn-bench serve` instead of `vllm bench serve`:

```bash
sendnn-bench serve \
    --model {model} \
    --endpoint /v1/completions \
    --dataset-name {custom/sharegpt/random...} \
    --dataset-path {path to dataset} \
    --num-prompts {num-prompts} \
    --max-concurrency {num-concurrent-users} \
    --save-result \
    --result-dir {path/to/results} \
    --result-filename result.json
```

!!! note

    `sendnn-bench` uses its own `spyre-chat` backend by default, which parses the extra metrics out of the streamed response. Do not override it with `--backend`.

    Any flag that makes a result JSON be written (`--save-result`, `--plot-timeline`, `--detailed-timeline`) requires **both** `--result-dir` and `--result-filename`, so that the file to inject the Spyre metrics into is unambiguous. `--append-result` is not supported.

The Spyre metrics are printed in a `SenDNN Metrics` section appended to the regular benchmark result table:

```text
============ Serving Benchmark Result ============
Successful requests:                     XX
...
----------------End-to-end Latency----------------
Mean E2EL (ms):                          XX
...
================= SenDNN Metrics =================
Total prefill chunks processed:          XX
Requests blocked by missing KV blocks:   XX
---------------- Queue Wait Time -----------------
Mean Queue Wait Time (ms):               XX
Median Queue Wait Time (ms):             XX
P99 Queue Wait Time (ms):                XX
P100 Queue Wait Time (ms):               XX
------------- Chunked Prefill Count --------------
...
------------ Chunked Prefill Latency -------------
...
--------------- Prefill Phase Time ---------------
...
------------- Time Spent Prefilling --------------
...
------------ Prefill Phase Idle Time -------------
...
-------------- Decode Step Latency ---------------
...
---------------- Prefix Cache Hit ----------------
...
-------------- Left Padding Blocks ---------------
...
----------------- Pause Latency ------------------
...
---------------- Number of Pauses ----------------
...
--------------- Total Time Paused ----------------
...
==================================================
```

## Additional Flags

Beyond the upstream `vllm bench serve` flags, `sendnn-bench` adds:

- **`--describe-metrics`**: writes a `sendnn_bench_metrics_description.txt` file into `--result-dir` (or the current directory), documenting what every printed metric measures and its sample granularity (one sample per request, per prefill chunk, per decode step, …). Recommended whenever you share results with someone else.

- **`--detailed-timeline`**: writes a `{result-filename}_detailed_timeline.html` Gantt chart next to the result JSON, viewable in any modern web browser. Unlike the upstream `--plot-timeline` — which shows a single TTFT bar per request — it breaks each request down into its queue wait, its individual chunked prefill steps, and its decode steps, which makes it easy to see where the time actually went.

- **`--decode-thresholds LOW,HIGH`**: two decode latency thresholds in milliseconds used to color the decode steps of the detailed timeline (green below `LOW`, orange between, red above `HIGH`). Only meaningful together with `--detailed-timeline`.

Combined with `--save-result`, per-request Spyre values are also injected into the result JSON as `spyre_*` arrays (`spyre_queued_time_s`, `spyre_chunk_prefill_latencies_s`, `spyre_decode_latencies_s`, `spyre_prefix_cache_hit_pct`, …), aligned with vLLM's own per-request arrays (`ttfts`, `itls`, `start_times`).

!!! info

    Plot generation requires the plotting libraries: `uv pip install vllm[bench]`

## Adding a New Metric

The set of collected metrics is meant to grow. The [`add-bench-metric`](https://github.com/torch-spyre/sendnn-inference/blob/main/.claude/skills/add-bench-metric/SKILL.md) Claude Code skill walks through every layer that a new per-request metric has to touch — scheduler-side timing, transport to the client, aggregation, printing, result JSON injection and tests. Give it a precise description of the metric and where its value should be computed:

```text
/add-bench-metric add a metric for the prefix cache hit percent, based on the number of
chunks saved from a cache hit over the expected number of prefill chunks, for each request.
```

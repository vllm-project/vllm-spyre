---
name: add-bench-metric
description: Add a new custom per-request benchmark metric to sendnn-bench serve. Covers every layer of the pipeline: scheduler-side timing → SpyreBenchState → _free_request → kv_transfer_params → SSE injection → client parsing → aggregation and printing → result JSON injection. Use when the user says "add a new bench metric", "track X per request", "surface Y in bench serve output", or similar.
argument-hint: <metric description e.g. "decode latency per request">
---

Add a new custom per-request benchmark metric: **$ARGUMENTS**.

If `$ARGUMENTS` is empty, ask the user what they want to measure before doing anything else.

---

## Background — the pipeline

Every custom bench metric travels through five layers (all scheduler-process-side):

```text
schedule()  →  SpyreBenchState (accumulate raw values)
  →  _free_request()  →  kv_transfer_params["__spyre__"]  (ZMQ to API server)
  →  patch_serving.py SSE injection  (into final usage SSE chunk)
  →  async_request_spyre_chat()  →  output.custom_metrics_dict
  →  _print_spyre_section()  +  _inject_spyre_metrics_into_result_file()
  →  _METRIC_DESCRIPTIONS  (--describe-metrics description file)
```

**Key architecture facts:**

- All timing is measured in the scheduler.
- `SpyreBenchState` is the single source of truth for per-request bench state. It is `None` when `SENDNN_INFERENCE_BENCH_METRICS_ENABLED` is off; every access must be guarded by `if self._bench is not None:`.
- Timing works by bracket: at the **end** of `schedule()`, a step-start timestamp (`prefill_step_start` or `decode_step_start`) is written into `_bench`. At the **start** of the next `update_from_output()`, the duration is computed from that timestamp and `time.time()`.
- `stats_logger.py` contains `SpyreRequestMetrics` / `SpyreMetricsRegistry` / `_SCHEDULER` which are **not part of the active metric pipeline** — do not add new metric fields there.
- `dataclasses.asdict()` is **not** used to serialize `SpyreBenchState` — the fields are serialised manually in `_free_request()` into a plain `dict` assigned to `kv_transfer_params["__spyre__"]`.

---

## Step 0 — Understand the new metric

Before writing any code, answer these questions (ask the user if unclear):

1. **Where is the raw value available?** Scheduler state (e.g., a timestamp set during `schedule()`)? `FinishedRequestStats` from vllm (e.g., `r.queued_time`)?
2. **What is the cardinality?** One scalar per request? One value per prefill chunk? One value per decode step? A list per request?
3. **What unit?** Seconds (convert to ms in the output)? Count? Bytes?
4. **Aggregation?** Mean/Median/PXX like the existing metrics (use `_section()`), or a run-level scalar (use a plain print line)?

---

## Step 1 — Add a field to `SpyreBenchState`

**File**: `sendnn_inference/v1/core/scheduler.py`

Add your field to the dataclass at the top of the file. **Every field carries a one- or two-line comment** saying what it holds — match that convention; a bare field is a review comment waiting to happen. State the unit and, for per-step lists, what one entry corresponds to (one prefill step / one decode step / one pause interval). Read the neighbouring fields in the real file rather than copying the abridged sketch below.

```python
@dataclass
class SpyreBenchState:
    # ... existing fields, each with its own comment ...

    # Duration of each decode step the request took part in.
    decode_latencies: dict[str, list[float]] = field(default_factory=dict)

    # NEW — keyed by request_id, choose the container type for your cardinality.
    # One entry per <prefill step | decode step | pause interval | request>; <unit>.
    my_new_metric: dict[str, <type>] = field(default_factory=dict)
```

---

## Step 2 — Populate the field in `update_from_output()` or `schedule()`

**File**: `sendnn_inference/v1/core/scheduler.py`

**If the value is a timing measurement** (most common): set a start timestamp at the end of `schedule()` (alongside the existing `prefill_step_start`/`decode_step_start` pattern), then compute and accumulate the duration in `update_from_output()` (alongside the existing chunk/decode duration blocks).

Example — adding a "time from first prefill to decode start" metric:

```python
# In update_from_output(), under `if self._bench is not None:`
if self._bench.prefill_step_start is not None:
    duration = now - self._bench.prefill_step_start
    for req_id in all_prefill_reqs:
        self._bench.my_new_metric.setdefault(req_id, []).append(duration)
```

**If the value comes from scheduler state** (e.g. request attributes): accumulate directly in `update_from_output()` under `if self._bench is not None:`:

```python
for req in self.ongoing_prefills:
    v = <computed from req or scheduler_output>
    if v is not None:
        self._bench.my_new_metric[req.request_id] = v
```

Always guard with `if self._bench is not None:`. Never read `self._bench` without this guard.

---

## Step 3 — Expose in `get_and_clear_chunk_stats()`

**File**: `sendnn_inference/v1/core/scheduler.py`

`get_and_clear_chunk_stats()` retrieves and removes per-request bench data in one atomic operation. Add your new field:

```python
def get_and_clear_chunk_stats(self, req_id: str) -> dict | None:
    if self._bench is None:
        return None
    lats = self._bench.chunk_latencies.pop(req_id, None)
    # ... existing pops ...
    my_val = self._bench.my_new_metric.pop(req_id, None)   # NEW
    if lats is None and dec_lats is None:
        return None
    return {
        "num_chunked_prefills": len(lats) if lats else 0,
        "chunk_prefill_latencies_s": lats or [],
        # ... existing fields ...
        "my_new_metric": my_val or <default>,              # NEW
    }
```

---

## Step 4 — Pack into `kv_transfer_params["__spyre__"]` in `_free_request()`

**File**: `sendnn_inference/v1/core/scheduler.py`

`_free_request()` builds the `spyre_data` dict that travels over ZMQ to the API server. Add your new field here:

```python
spyre_data = {
    "queued_time_s": queued_time_s,
    "num_chunked_prefills": chunk_stats["num_chunked_prefills"] if chunk_stats else 0,
    "chunk_prefill_latencies_s": chunk_stats["chunk_prefill_latencies_s"] if chunk_stats else [],
    # ... existing fields ...
    "my_new_metric": chunk_stats["my_new_metric"] if chunk_stats else <default>,   # NEW
}
```

This dict is what `patch_serving.py` picks up and injects into the SSE stream. No change to `patch_serving.py` itself is needed — it passes the whole dict through as `spyre_metrics`.

---

## Step 5 — No changes needed in `patch_serving.py` or `spyre_request_func.py`

`patch_serving.py` injects the entire `__spyre__` dict into the SSE chunk as-is. `async_request_spyre_chat` in `spyre_request_func.py` reads the `spyre_metrics` key from the final SSE usage chunk into `output.custom_metrics_dict` — your new key is automatically included. Verify by printing `output.custom_metrics_dict` in a test run if needed.

---

## Step 6 — Aggregate and print

**File**: `sendnn_inference/benchmarks/spyre_bench_serve.py`

In `_print_spyre_section()`, choose the output style based on cardinality:

**Option A — per-request distribution (Mean/Median/PXX)**: use `_section()`.

```python
# Scalar per request:
my_values = [m["my_new_metric"] for m in metrics_list if "my_new_metric" in m]

# List per request (flatten across all requests):
my_values = [v for m in metrics_list for v in m.get("my_new_metric", [])]

_section("My New Metric", my_values, "My New Metric (unit)")
```

**Option B — run-level scalar summary**: print a plain line *above* the `_section()` calls, mirroring the `total_prefill_chunks` pattern.

```python
total = sum(m.get("my_new_metric", 0) for m in metrics_list)
print("{:<40} {:<10}".format("My run-level total:", total))
```

---

## Step 7 — Add a description entry

**File**: `sendnn_inference/benchmarks/spyre_bench_serve.py`

Every printed metric has a short explanation in the module-level `_METRIC_DESCRIPTIONS` list, written to `sendnn_bench_metrics_description.txt` by `_write_metric_descriptions()` when the user passes `--describe-metrics`. Add an entry for your new metric:

```python
_METRIC_DESCRIPTIONS: list[tuple[str, str]] = [
    # ... existing entries ...
    (
        "My New Metric",   # must match the _section() header (or the run-level print label)
        "What it measures, in one or two sentences. State the sample granularity explicitly: "
        "one sample per request / per prefill chunk / per decode step / per pause interval.",
    ),
]
```

Rules for a good entry:

- **The first element must match the printed header exactly** — the `_section()` header string from Step 6, or the label of the run-level print line. This is what lets a reader map a description back to the table above it.
- **Keep the list in printed order** so the descriptions read in the same sequence as the table (run-level scalar lines first, then the `_section()` metrics).
- **Always state the sample granularity.** It changes how the mean should be read: a per-chunk metric is dominated by long-prompt requests, and a per-step metric batched across requests records the same value for every participant in that step.
- **Say what the measurement brackets**, not just its name — e.g. that a step latency is measured from the end of `schedule()` to `update_from_output()` and therefore covers the whole scheduler → executor → model round trip, not just the forward pass.
- **Mention a non-obvious exclusion or convention in one sentence** if there is one (e.g. pause metrics exclude an interval still open when the request finishes; prefix-cache hit is measured in chunks rather than tokens).
- Avoid characters that wrap badly: `textwrap.wrap()` will break on a lone ` - `, leaving a dangling hyphen at end of line. Write `(1 - a/b)` rather than `1 - a / b`.

---

## Step 8 — Inject into the result JSON

**File**: `sendnn_inference/benchmarks/spyre_bench_serve.py`

In `_inject_spyre_metrics_into_result_file()`, add your new key alongside the existing ones:

```python
# Per-request list (parallel to vllm's ttfts, itls, …)
result["spyre_my_new_metric"] = [m.get("my_new_metric", <default>) for m in metrics_list]

# Or a run-level scalar derived from an already-written list:
result["spyre_my_total"] = sum(result["spyre_my_new_metric"])
```

Use a `spyre_` prefix so the key is clearly SenDNN-owned in the vllm result JSON.

---

## Step 9 — Add tests

**File**: `tests/benchmarks/test_bench_metrics.py`

1. **Update `FAKE_METRICS`**: Add your new key with obviously synthetic values to both dict entries in the list.

2. **Update `test_inject_adds_spyre_keys`**: Assert the `spyre_`-prefixed key is present in the result JSON.

3. **Update `test_inject_values_correct`**: Assert the exact values are injected correctly for both requests.

4. **Update `test_scheduler_bench_metrics_accumulated`**: This is an integration test that runs a real engine and captures the bench state just before `_free_request` clears it. The capture (`_capturing_free`) and the post-run empty-dict check are both dynamic — they iterate over `dataclasses.fields(bench)` and need no changes. What you **do** need to add is a value assertion for your new field in the `for req_id in ("0", "1"):` block, following the pattern of the existing ones. For example, for a new list-per-step field:

    ```python
    # In the for req_id in ("0", "1"): block:
    assert len(info["my_new_metric"]) >= 1, (
        f"req {req_id}: expected ≥1 my_new_metric entry, got {info['my_new_metric']}"
    )
    for val in info["my_new_metric"]:
        assert isinstance(val, float) and val > 0, f"req {req_id}: non-positive my_new_metric {val}"
    ```

    For a scalar field (`arrival_ts`-style), check `is not None`:

    ```python
    assert info["my_scalar_field"] is not None, f"req {req_id}: my_scalar_field not set"
    ```

5. **Update `test_get_and_clear_returns_correct_dict`**: The test is driven by two dicts defined just above it — update both:

   - **`_BENCH_FIXTURE`** — add an entry for your new `SpyreBenchState` field. This dict must cover **every** field of `SpyreBenchState` (dict and scalar alike); `test_bench_fixture_covers_all_per_req_fields` compares `_BENCH_FIXTURE.keys()` against `dataclasses.fields(bench)` and fails if they diverge. For dict fields the value is used as the per-request payload (`bench.<field>["r0"] = value`); for scalar fields it is set directly (`setattr(bench, field, value)`).
   - **`_EXPECTED_RESULT`** — add an entry `"<result key>": <expected value>`. `test_get_and_clear_result_keys` asserts `result.keys() == _EXPECTED_RESULT.keys()`, so it will fail if the returned dict has any extra or missing keys.

    Example:

    ```python
    # In FAKE_METRICS entry 1:
    "my_new_metric": [0.001, 0.002],

    # In test_inject_adds_spyre_keys:
    assert "spyre_my_new_metric" in data

    # In test_inject_values_correct:
    assert data["spyre_my_new_metric"] == [[0.001, 0.002], [0.003]]

    # In the sentinel dicts (dict field example):
    _BENCH_FIXTURE: dict[str, Any] = {
        ...,
        "my_new_metric": [0.001, 0.002],  # NEW — dict field, keyed by req_id at test time
    }
    _EXPECTED_RESULT: dict[str, Any] = {
        ...,
        "my_new_metric_s": pytest.approx([0.001, 0.002]),  # NEW — key in returned dict
    }

    # Scalar field example (e.g. a single float per request stored directly):
    _BENCH_FIXTURE: dict[str, Any] = {
        ...,
        "my_scalar_field": 42.0,  # NEW — set directly via setattr
    }
    ```

6. **Update `test_print_spyre_section_output`** (if you added a new `_section()` call): Add `out.assert_contains("<SectionHeader>")` for the section separator, and add the label string (third argument to `_section()`) to the `for label in (...)` loop so mean/median/percentile lines are asserted. `assert_all_lines_covered()` is called at the end of the test and will fail if any output line was not covered by an assertion — the error message lists the exact uncovered lines.

7. **Cover the description entry.** `_METRIC_DESCRIPTIONS` must stay in sync with what `_print_spyre_section()` actually prints, and nothing enforces that automatically yet. Add (or extend) a test asserting that every printed section header has a matching description entry — walking the `_section()` header strings and checking each appears as a `_METRIC_DESCRIPTIONS[i][0]`. Without this, a new metric silently ships with no description.

---

## Checklist

Before declaring done:

- [ ] Field added to `SpyreBenchState` with the correct container type (scalar, list, dict-of-list…) and a short comment stating unit and what one entry corresponds to
- [ ] Populated in `update_from_output()` (or `schedule()`) under `if self._bench is not None:` guard
- [ ] Retrieved via `.pop()` in `get_and_clear_chunk_stats()` with a safe default if absent
- [ ] Key added to the `spyre_data` dict in `_free_request()` with a safe fallback when `chunk_stats is None`
- [ ] `_print_spyre_section()` updated with a new `_section()` call or run-level print line
- [ ] `_METRIC_DESCRIPTIONS` entry added, header matching the printed section exactly, in printed order, stating sample granularity
- [ ] `_inject_spyre_metrics_into_result_file()` updated with a new `result["spyre_..."]` key
- [ ] Tests updated: `FAKE_METRICS`, `test_inject_adds_spyre_keys`, `test_inject_values_correct`, `_BENCH_FIXTURE`, `_EXPECTED_RESULT` (drives `test_get_and_clear_returns_correct_dict`), `test_scheduler_bench_metrics_accumulated` adapted accordingly, `test_print_spyre_section_output` updated if a new `_section()` was added, and a check that the new metric has a `_METRIC_DESCRIPTIONS` entry
- [ ] No changes to `patch_serving.py`, `spyre_request_func.py`, or any model runner file
- [ ] No changes to `SpyreRequestMetrics` in `stats_logger.py` (not part of the active pipeline)
- [ ] All new tracking is gated by `self._bench is not None` (enforced by `SpyreBenchState` being `None` when env var is off)

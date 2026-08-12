# SPDX-License-Identifier: Apache-2.0
"""sendnn-bench serve — vllm bench serve extended with Spyre per-request metrics.

Usage:
    sendnn-bench serve --host localhost --port 8000 --model <model> \\
        --dataset-name random --num-prompts 20 --request-rate 2

Env var:
    SENDNN_INFERENCE_BENCH_METRICS_ENABLED=1  (must also be set on the server)
"""

import argparse
import asyncio
import json
import logging
import os
import time
from typing import Any

import numpy as np

from vllm.benchmarks.lib.endpoint_request_func import (
    ASYNC_REQUEST_FUNCS,
    RequestFuncInput,
)
from vllm.benchmarks.serve import add_cli_args, main_async
from vllm.utils.argparse_utils import FlexibleArgumentParser

from sendnn_inference.benchmarks.spyre_request_func import async_request_spyre_chat

logger = logging.getLogger(__name__)

_BACKEND_NAME = "spyre-chat"

# Shared accumulators — populated by the wrapper below during the benchmark run.
_spyre_metrics_collected: list[dict[str, Any]] = []
_request_outputs_collected: list[dict[str, Any]] = []

# Per-request join key. _spyre_metrics_collected fills in completion order while
# vllm's result arrays (ttfts/itls/start_times) are in submission order, so they
# are realigned by start_time at injection time rather than zipped positionally.
_SPYRE_START_TIME_KEY = "__spyre_start_time__"


def _make_collecting_func():
    """Return a wrapper around async_request_spyre_chat that accumulates
    custom_metrics_dict into _spyre_metrics_collected and per-request vLLM
    timing into _request_outputs_collected."""

    async def _wrapper(
        request_func_input: RequestFuncInput,
        session,
        pbar=None,
    ):
        output = await async_request_spyre_chat(request_func_input, session, pbar)
        if output.success:
            if output.custom_metrics_dict:
                # Copy so the join key doesn't leak into the splat below.
                _spyre_metrics_collected.append(
                    {**output.custom_metrics_dict, _SPYRE_START_TIME_KEY: output.start_time}
                )
                _request_outputs_collected.append(
                    {
                        "start_time": output.start_time,
                        "ttft": output.ttft,
                        "itl": output.itl,
                        "latency": output.latency,
                        "prompt_len": request_func_input.prompt_len,
                        "output_tokens": output.output_tokens,
                        **output.custom_metrics_dict,
                    }
                )
            else:
                logger.warning(
                    "Spyre metrics absent from response — is "
                    "SENDNN_INFERENCE_BENCH_METRICS_ENABLED set on the server?"
                )
        return output

    return _wrapper


def _register_backend() -> None:
    from vllm.benchmarks.lib.endpoint_request_func import OPENAI_COMPATIBLE_BACKENDS

    ASYNC_REQUEST_FUNCS[_BACKEND_NAME] = _make_collecting_func()
    # Register as an OpenAI-compatible backend so that vllm's main_async
    # enables ignore_eos for random datasets and allows sampling parameters.
    if _BACKEND_NAME not in OPENAI_COMPATIBLE_BACKENDS:
        OPENAI_COMPATIBLE_BACKENDS.append(_BACKEND_NAME)


def _build_parser() -> FlexibleArgumentParser:
    """Build an arg parser based on vllm's but with spyre-chat as default backend."""
    parser = FlexibleArgumentParser(
        description="Spyre-extended vllm bench serve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Register our backend before add_cli_args so it appears in --backend choices.
    _register_backend()
    add_cli_args(parser)

    # Override the default so --backend doesn't need to be specified explicitly.
    for action in parser._actions:
        if action.dest == "backend":
            action.default = _BACKEND_NAME
            break

    parser.add_argument(
        "--detailed-timeline",
        action="store_true",
        default=False,
        help=(
            "Write a detailed per-request Gantt-chart timeline HTML alongside the "
            "JSON result file (same name with a _detailed.html suffix). "
            "Requires --save-result and SENDNN_INFERENCE_BENCH_METRICS_ENABLED on the server."
        ),
    )
    parser.add_argument(
        "--describe-metrics",
        action="store_true",
        default=False,
        help=(
            "Write a 'sendnn_bench_metrics_description.txt' file (in --result-dir, or the "
            "current directory) describing every metric section: what it measures and its "
            "sample granularity."
        ),
    )
    parser.add_argument(
        "--decode-thresholds",
        type=str,
        metavar="LOW,HIGH",
        default=None,
        help=(
            "Two decode latency thresholds in milliseconds (comma-separated) for "
            "coloring in the detailed timeline. "
            "Decode steps below LOW are green, between LOW and HIGH are orange, "
            "above HIGH are red. When omitted, all decode steps are green."
        ),
    )

    return parser


# Short explanation of every metric printed by _print_spyre_section, written to
# _METRIC_DESCRIPTION_FILENAME when --describe-metrics is passed. Ordered to match
# the printed output.
_METRIC_DESCRIPTIONS: list[tuple[str, str]] = [
    (
        "Total prefill chunks processed",
        "Total number of chunked prefills executed by the server across all requests.",
    ),
    (
        "Requests blocked by missing KV blocks",
        "Number of requests that were held back at least once because not enough free "
        "KV-cache blocks were available. A non-zero count signals KV-cache pressure, "
        "not an error.",
    ),
    (
        "Queue Wait Time",
        "Time from when the API server received the request until the start of its first "
        "prefill step. Unlike vLLM's own queue time this includes the API-server -> "
        "engine-core hop, so it adds up with prefill and decode latencies to reconstruct "
        "TTFT. One sample per request.",
    ),
    (
        "Chunked Prefill Count",
        "Number of prefill chunks actually executed for a request. Lower than "
        "ceil(prompt_len / chunk_size) when a prefix-cache hit lets whole chunks be "
        "skipped. One sample per request.",
    ),
    (
        "Chunked Prefill Latency",
        "Wall-clock duration of a single prefill step. One sample per chunk.",
    ),
    (
        "Prefill Phase Time",
        "Elapsed time from the start of a request's first prefill step to the end of its "
        "last one, which is when its first token is produced. Starts after the request "
        "leaves the waiting queue, so 'Queue Wait Time' does not contribute. One sample "
        "per request.",
    ),
    (
        "Time Spent Prefilling",
        "Time a request spent actually executing prefill steps, i.e. the sum of its "
        "Chunked Prefill Latency samples. One sample per request.",
    ),
    (
        "Prefill Phase Idle Time",
        "Time a request spent inside its prefill phase without prefilling, computed as "
        "(Prefill Phase Time - Time Spent Prefilling). Covers pausing and any other "
        "reason the request made no prefill progress, but excludes the initial queue wait. "
        "One sample per request.",
    ),
    (
        "Decode Step Latency",
        "Wall-clock duration of a single decode step. One sample per decode step per request.",
    ),
    (
        "Prefix Cache Hit",
        "Fraction of a request's prefill chunks that were skipped thanks to the prefix "
        "cache, computed as (1 - executed_chunks/expected_chunks). Measured in chunks "
        "rather than tokens because Spyre only ever skips whole chunks. "
        "One sample per request.",
    ),
    (
        "Left Padding Blocks",
        "Per decode step, the number of KV-cache blocks of left padding a request carries "
        "because it is shorter than the batch's longest sequence "
        "(ceil(tkv/block_size) - ceil(computed_tokens/block_size)). Zero for the longest "
        "request; high values mean the batch mixes very different sequence lengths and "
        "wastes compute on padding.",
    ),
    (
        "Pause Latency",
        "Duration of a single pause interval - the time a decoding request spent evicted "
        "from the running batch because the batch TKV limit could not accommodate it. "
        "One sample per pause interval. A request still paused when it finishes has that "
        "final open interval excluded from all three pause metrics.",
    ),
    (
        "Number of Pauses",
        "How many times a request was paused and later resumed. One sample per request "
        "(including 0). A request still paused when it finishes has that final open "
        "interval excluded from all three pause metrics.",
    ),
    (
        "Total Time Paused",
        "Total time a request spent paused over its lifetime (the sum of its pause "
        "intervals). One sample per request, including 0. Read alongside Pause Latency to "
        "distinguish many short pauses from one long one. A request still paused when it "
        "finishes has that final open interval excluded from all three pause metrics.",
    ),
]


_METRIC_DESCRIPTION_FILENAME = "sendnn_bench_metrics_description.txt"


def _write_metric_descriptions(result_dir: str | None = None) -> None:
    """Write a short explanation of every metric section to a text file.

    Each description is a single unwrapped line so it reflows in any editor."""
    file_path = os.path.join(result_dir or ".", _METRIC_DESCRIPTION_FILENAME)

    lines = ["SenDNN benchmark metric descriptions", ""]
    for header, description in _METRIC_DESCRIPTIONS:
        lines.append(f"{header}:")
        lines.append(description)
        lines.append("")

    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        logger.info("SenDNN metric descriptions written to %s", file_path)
    except OSError as exc:
        logger.warning("Failed to write SenDNN metric descriptions to %s: %s", file_path, exc)


def _print_spyre_section(
    metrics_list: list[dict[str, Any]],
    selected_percentiles: list[float],
) -> None:
    """Print Spyre-specific metrics in vllm bench serve format."""
    if not metrics_list:
        return

    queue_times_ms = [m["queued_time_s"] * 1000 for m in metrics_list if "queued_time_s" in m]
    num_chunks_list = [
        m["num_chunked_prefills"] for m in metrics_list if "num_chunked_prefills" in m
    ]
    chunk_lats_ms = [
        lat * 1000 for m in metrics_list for lat in m.get("chunk_prefill_latencies_s", [])
    ]
    decode_lats_ms = [lat * 1000 for m in metrics_list for lat in m.get("decode_latencies_s", [])]
    total_prefill_chunks = sum(num_chunks_list)

    total_missing_blocks = sum(1 for m in metrics_list if m.get("was_missing_blocks", False))

    # Scalar summary lines (mirrors vllm's plain-count header section)
    print("{:<40} {:<10}".format("Total prefill chunks processed:", total_prefill_chunks))
    print("{:<40} {:<10}".format("Requests blocked by missing KV blocks:", total_missing_blocks))

    def _section(header: str, values: list[float], label: str) -> None:
        if not values:
            values = [0.0]
        arr = np.array(values)
        print("{s:{c}^{n}}".format(s=f" {header} ", n=50, c="-"))
        print("{:<40} {:<10.2f}".format(f"Mean {label}:", float(np.mean(arr))))
        print("{:<40} {:<10.2f}".format(f"Median {label}:", float(np.median(arr))))
        for p in selected_percentiles:
            print(
                "{:<40} {:<10.2f}".format(
                    f"P{int(p) if int(p) == p else p} {label}:",
                    float(np.percentile(arr, p)),
                )
            )

    cache_hit_pcts = [
        m["prefix_cache_hit_pct"] * 100 for m in metrics_list if "prefix_cache_hit_pct" in m
    ]

    prefill_elapsed_ms = [
        m["prefill_elapsed_s"] * 1000 for m in metrics_list if "prefill_elapsed_s" in m
    ]
    prefill_busy_ms = [m["prefill_busy_s"] * 1000 for m in metrics_list if "prefill_busy_s" in m]
    prefill_idle_ms = [m["prefill_idle_s"] * 1000 for m in metrics_list if "prefill_idle_s" in m]

    left_padding_blocks = [v for m in metrics_list for v in m.get("left_padding_blocks", [])]
    pause_lats_ms = [lat * 1000 for m in metrics_list for lat in m.get("pause_latencies_s", [])]
    pause_counts = [float(len(m.get("pause_latencies_s", []))) for m in metrics_list]
    total_pause_ms = [float(sum(m.get("pause_latencies_s", []))) * 1000 for m in metrics_list]

    _section("Queue Wait Time", queue_times_ms, "Queue Wait Time (ms)")
    _section("Chunked Prefill Count", num_chunks_list, "Num Chunked Prefills")
    _section("Chunked Prefill Latency", chunk_lats_ms, "Chunk Prefill Latency (ms)")
    _section("Prefill Phase Time", prefill_elapsed_ms, "Prefill Phase Time (ms)")
    _section("Time Spent Prefilling", prefill_busy_ms, "Time Spent Prefilling (ms)")
    _section("Prefill Phase Idle Time", prefill_idle_ms, "Prefill Phase Idle Time (ms)")
    _section("Decode Step Latency", decode_lats_ms, "Decode Step Latency (ms)")
    _section("Prefix Cache Hit", cache_hit_pcts, "Prefix Cache Hit (%)")
    _section("Left Padding Blocks", left_padding_blocks, "Left Padding Blocks")
    _section("Pause Latency", pause_lats_ms, "Pause Latency (ms)")
    _section("Number of Pauses", pause_counts, "Num Pauses")
    _section("Total Time Paused", total_pause_ms, "Total Time Paused (ms)")

    print("=" * 50)


def _inject_spyre_metrics_into_result_file(
    args: Any,
    metrics_list: list[dict[str, Any]],
    run_started_at: float,
) -> None:
    """If vllm wrote a result JSON (--save-result / --append-result / --result-filename),
    find it and inject per-request Spyre metric lists alongside vllm's own per-request
    fields (ttfts, itls, …)."""
    if not metrics_list:
        return
    if not (
        getattr(args, "save_result", False)
        or getattr(args, "append_result", False)
        or getattr(args, "result_filename", None)
    ):
        return

    # Locate the file vllm just wrote by finding the newest .json in the result dir
    # that was modified after we started the run.
    result_dir = getattr(args, "result_dir", None) or "."
    explicit_name = getattr(args, "result_filename", None)

    if explicit_name:
        candidate = (
            explicit_name
            if os.path.isabs(explicit_name)
            else os.path.join(result_dir, explicit_name)
        )
        candidates = [candidate] if os.path.isfile(candidate) else []
    else:
        try:
            candidates = [
                os.path.join(result_dir, f)
                for f in os.listdir(result_dir)
                if f.endswith(".json")
                and os.path.getmtime(os.path.join(result_dir, f)) >= run_started_at
            ]
        except OSError:
            candidates = []

    if not candidates:
        logger.warning("Could not locate vllm result JSON to inject Spyre metrics into.")
        return

    file_path = max(candidates, key=os.path.getmtime)

    try:
        with open(file_path, encoding="utf-8") as fh:
            result = json.load(fh)
    except Exception as exc:
        logger.warning("Failed to read vllm result JSON %s: %s", file_path, exc)
        return

    # Reorder metrics to match vllm's submission-ordered rows by start_time,
    # emitting one entry per row (missing rows get sentinels) so every spyre_*
    # array stays aligned with ttfts/itls/start_times.
    ordered: list[dict[str, Any] | None]
    start_times = result.get("start_times")
    if isinstance(start_times, list):
        by_start = {
            round(float(m[_SPYRE_START_TIME_KEY]), 9): m
            for m in metrics_list
            if _SPYRE_START_TIME_KEY in m
        }
        ordered = [by_start.get(round(float(st), 9)) for st in start_times]
        matched = sum(1 for m in ordered if m is not None)
        if matched != len(metrics_list):
            logger.warning(
                "Spyre metrics: matched %d/%d collected metric sets to result rows "
                "by start_time; unmatched rows use sentinels.",
                matched,
                len(metrics_list),
            )
    else:
        logger.warning(
            "Spyre metrics: result JSON has no 'start_times'; falling back to "
            "collection order, which may not match vllm's per-request arrays."
        )
        ordered = list(metrics_list)

    def _col(key: str, default: Any) -> list[Any]:
        return [default if m is None else m.get(key, default) for m in ordered]

    result["spyre_queued_time_s"] = _col("queued_time_s", None)
    result["spyre_num_chunked_prefills"] = _col("num_chunked_prefills", None)
    result["spyre_chunk_prefill_latencies_s"] = _col("chunk_prefill_latencies_s", [])
    result["spyre_chunk_prefill_start_times_s"] = _col("chunk_prefill_start_times_s", [])
    result["spyre_total_prefill_chunks"] = sum(
        n for n in result["spyre_num_chunked_prefills"] if n is not None
    )
    result["spyre_decode_latencies_s"] = _col("decode_latencies_s", [])
    result["spyre_decode_start_times_s"] = _col("decode_start_times_s", [])
    result["spyre_tkvs"] = _col("tkvs", [])
    result["spyre_prefill_elapsed_s"] = _col("prefill_elapsed_s", 0.0)
    result["spyre_prefill_busy_s"] = _col("prefill_busy_s", 0.0)
    result["spyre_prefill_idle_s"] = _col("prefill_idle_s", 0.0)
    result["spyre_prefix_cache_hit_pct"] = _col("prefix_cache_hit_pct", 0.0)
    result["spyre_left_padding_blocks"] = _col("left_padding_blocks", [])
    result["spyre_pause_latencies_s"] = _col("pause_latencies_s", [])
    result["spyre_pause_start_times_s"] = _col("pause_start_times_s", [])
    result["spyre_was_missing_blocks"] = _col("was_missing_blocks", False)
    result["spyre_num_requests_missing_blocks"] = sum(
        1 for v in result["spyre_was_missing_blocks"] if v
    )

    try:
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write("{\n")
            items = list(result.items())
            for i, (k, v) in enumerate(items):
                comma = "," if i < len(items) - 1 else ""
                fh.write(f"  {json.dumps(k)}: {json.dumps(v)}{comma}\n")
            fh.write("}\n")
        logger.info("Spyre metrics injected into %s", file_path)
    except Exception as exc:
        logger.warning("Failed to write Spyre metrics into result JSON %s: %s", file_path, exc)


def _run_vllm_and_capture_trailing(args: Any) -> tuple[str, str]:
    """Run vllm's main_async, letting stdout/stderr pass through live until the
    closing '=' * 50 line of the metrics table.  Everything written after that
    marker is captured and returned as (stdout_trailing, stderr_trailing)."""
    import io
    import sys

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    done = {"v": False}

    class _StdoutSplitter:
        def write(self, s):
            if not done["v"]:
                if s.strip() == "=" * 50:
                    done["v"] = True
                else:
                    orig_stdout.write(s)
            else:
                stdout_buf.write(s)

        def flush(self):
            orig_stdout.flush()

    class _StderrSplitter:
        def write(self, s):
            if not done["v"]:
                orig_stderr.write(s)
            else:
                stderr_buf.write(s)

        def flush(self):
            if not done["v"]:
                orig_stderr.flush()

    sys.stdout = _StdoutSplitter()
    sys.stderr = _StderrSplitter()
    try:
        asyncio.run(main_async(args))
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr

    return stdout_buf.getvalue(), stderr_buf.getvalue()


def main() -> None:
    import sys

    # Allow `sendnn-bench serve <args>` as an alias (the word "serve" is ignored).
    argv = sys.argv[1:]
    if argv and argv[0] == "serve":
        argv = argv[1:]

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Force our custom backend so Spyre metrics are always collected.
    args.backend = _BACKEND_NAME

    selected_percentiles = [float(p) for p in args.metric_percentiles.split(",")]

    _spyre_metrics_collected.clear()
    _request_outputs_collected.clear()

    run_started_at = time.time()
    stdout_trailing, stderr_trailing = _run_vllm_and_capture_trailing(args)

    print("{s:{c}^{n}}".format(s=" SenDNN Metrics ", n=50, c="="))
    _print_spyre_section(_spyre_metrics_collected, selected_percentiles)

    if getattr(args, "describe_metrics", False):
        _write_metric_descriptions(getattr(args, "result_dir", None))

    _inject_spyre_metrics_into_result_file(args, _spyre_metrics_collected, run_started_at)

    if getattr(args, "detailed_timeline", False):
        from pathlib import Path

        from sendnn_inference.benchmarks.spyre_plot import generate_detailed_timeline_plot

        # Derive the HTML path from the JSON result file: same name, _detailed.html suffix.
        result_dir = getattr(args, "result_dir", None) or "."
        explicit_name = getattr(args, "result_filename", None)
        if explicit_name:
            json_candidate = (
                explicit_name
                if os.path.isabs(explicit_name)
                else os.path.join(result_dir, explicit_name)
            )
            candidates = [json_candidate] if os.path.isfile(json_candidate) else []
        else:
            try:
                candidates = [
                    os.path.join(result_dir, f)
                    for f in os.listdir(result_dir)
                    if f.endswith(".json")
                    and os.path.getmtime(os.path.join(result_dir, f)) >= run_started_at
                ]
            except OSError:
                candidates = []

        if candidates:
            json_path = Path(str(max(candidates, key=os.path.getmtime)))
            html_path = json_path.with_name(json_path.stem + "_detailed_timeline.html")
            decode_thresholds_str = getattr(args, "decode_thresholds", None)
            # Parse comma-separated milliseconds and convert to seconds
            decode_thresholds = None
            if decode_thresholds_str:
                try:
                    thresholds_ms = [float(x.strip()) for x in decode_thresholds_str.split(",")]
                    if len(thresholds_ms) != 2:
                        raise ValueError("Expected exactly 2 comma-separated values")
                    decode_thresholds = [ms / 1000.0 for ms in thresholds_ms]
                except (ValueError, AttributeError) as e:
                    logger.warning(
                        "Invalid --decode-thresholds format: %s (expected LOW,HIGH in ms)",
                        e,
                    )
                    decode_thresholds = None
            generate_detailed_timeline_plot(
                _request_outputs_collected, html_path, decode_thresholds=decode_thresholds
            )
        else:
            logger.warning(
                "--detailed-timeline requires --save-result so the JSON path is known; "
                "no result file found, skipping timeline."
            )

    trailing = stdout_trailing + stderr_trailing
    if trailing.strip():
        print(trailing, end="")


if __name__ == "__main__":
    main()

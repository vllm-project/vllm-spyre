"""
Tests checking for vLLM upstream compatibility requirements.

As we remove support for old vLLM versions, we want to keep track of the
compatibility code that can be cleaned up.
"""

import inspect
import re

import pytest

pytestmark = pytest.mark.compat


# ---------------------------------------------------------------------------
# sendnn-bench serve custom metrics — sendnn_inference/v1/metrics/patch_serving.py
#
# patch_serving() wraps two upstream stream generators, passing the parameters
# below positionally. A rename/reorder/insert within that prefix, or a switch to
# keyword-only, silently misbinds result_generator and breaks every streaming
# request. Changes after the prefix are absorbed by *args/**kwargs.
#
# On failure: update the wrapper in patch_serving.py, then the prefix here.
# ---------------------------------------------------------------------------

# Must match the parameters named in patch_serving._patch_chat._patched_generator
CHAT_STREAM_GENERATOR_PREFIX = ("self", "request", "result_generator", "request_id")

# Must match the parameters named in patch_serving._patch_completions._patched_generator
COMPLETION_STREAM_GENERATOR_PREFIX = (
    "self",
    "request",
    "engine_inputs",
    "result_generator",
    "request_id",
)


def _assert_positional_prefix(func, expected_prefix: tuple[str, ...]) -> None:
    params = list(inspect.signature(func).parameters.values())
    actual_prefix = tuple(p.name for p in params[: len(expected_prefix)])

    assert actual_prefix == expected_prefix, (
        f"{func.__qualname__} leading parameters changed upstream: "
        f"expected {expected_prefix}, got {actual_prefix}. "
        f"patch_serving() binds these positionally and must be updated."
    )

    # Positional binding also requires that none of them became keyword-only.
    for param in params[: len(expected_prefix)]:
        assert param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
            f"{func.__qualname__} parameter '{param.name}' is now {param.kind.name}; "
            f"patch_serving() passes it positionally and must be updated."
        )


def test_sendnn_bench__serving_module_paths_unchanged():
    """patch_serving() imports these modules and skips patching (with only a
    warning) if they move, so bench metrics would silently go missing."""
    import importlib

    for module_path, cls_name, method_name in [
        (
            "vllm.entrypoints.openai.chat_completion.serving",
            "OpenAIServingChat",
            "chat_completion_stream_generator",
        ),
        (
            "vllm.entrypoints.openai.completion.serving",
            "OpenAIServingCompletion",
            "completion_stream_generator",
        ),
    ]:
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:  # pragma: no cover - only on upstream move
            pytest.fail(
                f"{module_path} is no longer importable ({e}); patch_serving() "
                f"skips patching and bench metrics will be missing from SSE output."
            )

        cls = getattr(module, cls_name, None)
        assert cls is not None, (
            f"{cls_name} no longer exists in {module_path}; patch_serving() must be updated."
        )
        assert hasattr(cls, method_name), (
            f"{cls_name}.{method_name} no longer exists; patch_serving() patches this "
            f"attribute and must be updated."
        )


def test_sendnn_bench__chat_stream_generator_signature_unchanged():
    """patch_serving._patch_chat wraps this method and binds its first
    parameters positionally."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat

    _assert_positional_prefix(
        OpenAIServingChat.chat_completion_stream_generator, CHAT_STREAM_GENERATOR_PREFIX
    )


def test_sendnn_bench__completion_stream_generator_signature_unchanged():
    """patch_serving._patch_completions wraps this method and binds its first
    parameters positionally."""
    from vllm.entrypoints.openai.completion.serving import OpenAIServingCompletion

    _assert_positional_prefix(
        OpenAIServingCompletion.completion_stream_generator, COMPLETION_STREAM_GENERATOR_PREFIX
    )


# ---------------------------------------------------------------------------
# sendnn_inference/benchmarks/spyre_bench_serve.py
#
# _run_vllm_and_capture_trailing() calls main_async() and splits its stdout on a
# closing line of specific form, so it relies on that line's exact shape.
#
# Both break modes are silent: never matching captures nothing, matching too early
# swallows upstream's metrics table.
# ---------------------------------------------------------------------------


def test_sendnn_bench__main_async_unchanged():
    """spyre_bench_serve imports main_async and calls it as main_async(args)."""
    import asyncio

    from vllm.benchmarks.serve import main_async

    assert asyncio.iscoroutinefunction(main_async), (
        "vllm.benchmarks.serve.main_async is no longer a coroutine function; "
        "_run_vllm_and_capture_trailing calls it via asyncio.run()."
    )

    params = list(inspect.signature(main_async).parameters.values())
    positional = [
        p
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    assert len(positional) == 1, (
        f"vllm.benchmarks.serve.main_async now takes {len(positional)} positional "
        f"parameters ({[p.name for p in positional]}); _run_vllm_and_capture_trailing "
        f"calls main_async(args) with exactly one."
    )


def test_sendnn_bench__metrics_table_end_marker_unchanged():
    """_run_vllm_and_capture_trailing splits stdout on the line closing upstream's
    metrics table. It must still be printed, exactly once, as a standalone print()."""
    import vllm.benchmarks.serve as upstream_serve

    from sendnn_inference.benchmarks.spyre_bench_serve import _VLLM_METRICS_TABLE_END_MARKER

    source = inspect.getsource(upstream_serve)

    # Matches `print("=" * 50)` / `print('=' * 50)` with flexible inner spacing.
    width = len(_VLLM_METRICS_TABLE_END_MARKER)
    char = _VLLM_METRICS_TABLE_END_MARKER[0]
    pattern = re.compile(rf"""print\(\s*['"]{re.escape(char)}['"]\s*\*\s*{width}\s*\)""")
    matches = pattern.findall(source)

    assert len(matches) == 1, (
        f"Expected exactly one `print({char!r} * {width})` in vllm.benchmarks.serve "
        f"(found {len(matches)})."
    )


def test_sendnn_bench__table_headers_do_not_collide_with_end_marker():
    """Upstream's centered section headers must keep a non-empty title, else one
    strips down to the end marker and the splitter swallows the metrics table."""
    import vllm.benchmarks.serve as upstream_serve

    from sendnn_inference.benchmarks.spyre_bench_serve import _VLLM_METRICS_TABLE_END_MARKER

    source = inspect.getsource(upstream_serve)
    marker_char = _VLLM_METRICS_TABLE_END_MARKER[0]

    # Upstream centers section titles in a run of '=' or '-', e.g.
    #   print("{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=50, c="="))
    for match in re.finditer(r"""\.format\(\s*s=\s*(['"])(.*?)\1""", source):
        title = match.group(2)
        assert title.strip(marker_char).strip(), (
            f"Upstream renders a centered header with title {title!r}, which strips to "
            f"the end-of-table marker and would flip _run_vllm_and_capture_trailing's "
            f"splitter early, swallowing upstream's metrics table."
        )


# ---------------------------------------------------------------------------
# sendnn-bench serve custom metrics — cross-process metric hop
#
# The scheduler-side override packs its per-request metrics into the dict that
# Scheduler._free_request returns, and relies on that dict riding
# kv_transfer_params all the way to the API server:
#
#   SpyreScheduler._free_request  (mutates super()'s returned kv_xfer_params)
#       -> EngineCoreOutput.kv_transfer_params  (ZMQ hop to API server process)
#           -> RequestOutput.kv_transfer_params  (read in patch_serving, gated on
#              RequestOutput.finished)
#
# If upstream stops returning the dict from _free_request, or drops
# kv_transfer_params from either struct, or renames RequestOutput.finished, the
# metrics silently vanish with a green suite. These tests pin each seam.
#
# On failure: the metric plumbing in scheduler.py / patch_serving.py must be
# reworked around the new upstream shape.
# ---------------------------------------------------------------------------


def test_sendnn_bench__free_request_returns_kv_xfer_dict():
    """SpyreScheduler._free_request calls super()._free_request(request,
    delay_free_blocks), mutates the returned dict with a '__spyre__' key, and
    returns it. Upstream must keep the (request, delay_free_blocks) signature and
    a dict|None return."""
    from vllm.v1.core.sched.scheduler import Scheduler

    sig = inspect.signature(Scheduler._free_request)
    params = list(sig.parameters.values())
    names = [p.name for p in params]

    assert names[:3] == ["self", "request", "delay_free_blocks"], (
        f"Scheduler._free_request parameters changed upstream: expected "
        f"(self, request, delay_free_blocks, ...), got {names}. "
        f"SpyreScheduler._free_request delegates positionally and must be updated."
    )

    delay_free_blocks = sig.parameters["delay_free_blocks"]
    assert delay_free_blocks.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD, (
        f"Scheduler._free_request 'delay_free_blocks' is now {delay_free_blocks.kind.name}; "
        f"SpyreScheduler._free_request passes it positionally and must be updated."
    )

    # The override packs metrics into this return value and hands it back; a None
    # or non-dict-bearing return would drop the metrics on the floor.
    ret = sig.return_annotation
    assert ret is not inspect.Signature.empty and "dict" in str(ret), (
        f"Scheduler._free_request return annotation is {ret!r}, no longer a dict|None; "
        f"SpyreScheduler._free_request stashes metrics under a '__spyre__' key in this "
        f"dict and must be updated."
    )


def test_sendnn_bench__kv_transfer_params_survives_to_request_output():
    """The '__spyre__' metrics ride kv_transfer_params from EngineCoreOutput to
    RequestOutput, and patch_serving reads them gated on RequestOutput.finished."""
    from vllm.outputs import RequestOutput
    from vllm.v1.engine import EngineCoreOutput

    assert "kv_transfer_params" in EngineCoreOutput.__struct_fields__, (
        "EngineCoreOutput no longer carries kv_transfer_params; the scheduler-side "
        "'__spyre__' metrics cannot cross to the API server process. patch_serving "
        "and scheduler._free_request must be updated."
    )

    ro_params = inspect.signature(RequestOutput.__init__).parameters
    assert "kv_transfer_params" in ro_params, (
        "RequestOutput.__init__ no longer accepts kv_transfer_params; patch_serving "
        "reads res.kv_transfer_params.get('__spyre__') and must be updated."
    )
    assert "finished" in ro_params, (
        "RequestOutput.__init__ no longer accepts 'finished'; patch_serving gates the "
        "'__spyre__' read on res.finished and must be updated."
    )


@pytest.mark.utils
def test_get_computed_blocks_returns_3_tuple():
    """KVCacheManager.get_computed_blocks returns a 3-tuple as of vLLM v0.27.1
    (previously a 2-tuple). When vLLM:lowest is bumped past v0.27.1, the
    unpack-to-3 shims in sendnn_inference/v1/core/scheduler.py and the mock
    return_value fixtures can be simplified and this test deleted.
    """
    from vllm.v1.core.kv_cache_manager import KVCacheManager

    sig = inspect.signature(KVCacheManager.get_computed_blocks)
    annotation = sig.return_annotation
    assert annotation != inspect.Parameter.empty
    args = getattr(annotation, "__args__", None)
    assert args is not None and len(args) >= 3, (
        f"get_computed_blocks returns {len(args) if args else '?'} values, expected >=3. "
        "Remove 3-value unpack shims in scheduler.py when vLLM:lowest > v0.27.1."
    )


@pytest.mark.utils
def test_get_num_blocks_to_allocate_has_num_local_computed_tokens():
    """KVCacheCoordinator.get_num_blocks_to_allocate gained
    num_local_computed_tokens in vLLM v0.27.1. When vLLM:lowest is bumped
    past v0.27.1, the explicit kwarg in scheduler.py is no longer a shim
    and this test can be deleted.
    """
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from sendnn_inference.compat_utils import has_argument

    assert has_argument(
        KVCacheCoordinator.get_num_blocks_to_allocate, "num_local_computed_tokens"
    ), (
        "get_num_blocks_to_allocate no longer accepts num_local_computed_tokens. "
        "Remove the kwarg from the call site in scheduler.py when vLLM:lowest > v0.27.1."
    )


@pytest.mark.utils
def test_finish_requests_returns_list_of_request():
    """Scheduler.finish_requests returns list[Request] as of vLLM v0.27.1
    (previously list[tuple[str, int]]). When vLLM:lowest is bumped past
    v0.27.1, the return type annotation on SpyreScheduler.finish_requests is
    no longer a shim and this test can be deleted.
    """
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

    sig = inspect.signature(Scheduler.finish_requests)
    annotation = sig.return_annotation
    assert annotation != inspect.Parameter.empty
    args = getattr(annotation, "__args__", None)
    assert args is not None and args[0] is Request, (
        f"finish_requests return type is {annotation!r}, expected list[Request]. "
        "Update the return annotation on SpyreScheduler.finish_requests when vLLM:lowest > v0.27.1."
    )

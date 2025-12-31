import hashlib
import os
import random

import pytest
import torch
from llm_cache import clear_llm_caches, get_cached_api_server, print_llm_cache_info
from llm_cache_util import SortKey, sort_tests_for_llm_caching
from spyre_util import get_spyre_backend_list, get_spyre_model_list, skip_unsupported_tp_size
from vllm.connections import global_http_connection
from vllm.distributed import cleanup_dist_env_and_memory

from vllm_spyre import envs

# Running with "fork" can lead to hangs/crashes
# Specifically, our use of transformers to compare results causes an OMP thread
# pool to be created, which is then lost when the next test launches vLLM and
# forks a worker.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def pytest_generate_tests(metafunc):
    """This hook is called during the collection phase,
    specifically when Pytest encounters a test function that
    needs parametrization. It receives a metafunc object,
    which provides information about the test function and
    allows for dynamic parametrization."""

    # default parameterizations
    default_warmup_shape = [[(64, 20, 4)]]
    default_max_num_seqs = [4]
    default_max_model_len = [512]
    default_max_num_batched_tokens = [128]

    existing_markers = [
        marker.name if marker.name != "parametrize" else marker.args[0]
        for marker in metafunc.definition.own_markers
    ]

    marker = metafunc.config.option.markexpr  # From CLI
    # TODO: make this condition better
    # it can accidentally be triggered if we say
    # -m "not full_model"
    if "full_model" in marker:
        # When -m full_model is called, all tests tagged with
        # full_model mark will be injected with these custom values
        if metafunc.definition.get_closest_marker("full_model"):
            _add_param(
                "model", get_spyre_model_list(full_size_models=True), metafunc, existing_markers
            )
            _add_param(
                "backend",
                ["sendnn"],
                metafunc,
                existing_markers,
            )
            _add_param(
                "warmup_shapes",
                [[(1024, 20, 4)]],
                metafunc,
                existing_markers,
            )
    else:
        # Default parameters
        _add_param("model", get_spyre_model_list(), metafunc, existing_markers)
        _add_param(
            "backend",
            get_spyre_backend_list(),
            metafunc,
            existing_markers,
        )
        _add_param(
            "warmup_shapes",
            default_warmup_shape,
            metafunc,
            existing_markers,
        )

    # apply to all
    _add_param(
        "max_model_len",
        default_max_model_len,
        metafunc,
        existing_markers,
    )

    _add_param(
        "max_num_seqs",
        default_max_num_seqs,
        metafunc,
        existing_markers,
    )

    _add_param(
        "max_num_batched_tokens",
        default_max_num_batched_tokens,
        metafunc,
        existing_markers,
    )

    # TODO: add both these using _add_param too
    # Will need to do some fancy stuff to add custom
    # markers
    if (
        "mode" in metafunc.fixturenames
        and "cb" not in existing_markers
        and "chunked_prefill" not in existing_markers
        and "cp" not in existing_markers
        and "pc" not in existing_markers
        and "mode" not in existing_markers
    ):
        metafunc.parametrize(
            "mode",
            [
                pytest.param("sb", marks=pytest.mark.sb, id="sb"),
                pytest.param("cb", marks=pytest.mark.cb, id="cb"),
                pytest.param("cp", marks=pytest.mark.chunked_prefill, id="cp"),
                pytest.param("pc", marks=pytest.mark.prefix_caching, id="pc"),
            ],
        )

    if "tp_size" in metafunc.fixturenames and "tp_size" not in existing_markers:
        metafunc.parametrize(
            "tp_size",
            [
                pytest.param(1),
                pytest.param(2, marks=pytest.mark.multi),
                pytest.param(4, marks=pytest.mark.multi),
                pytest.param(8, marks=pytest.mark.multi),
            ],
            ids=lambda val: f"TP({val})",
        )


def _add_param(param_name: str, param_value, metafunc, existing_markers) -> None:
    """helper function to parametrize stuff.
    We make sure to not parametrize something
    if it exists explicitly on the test"""
    if param_name in metafunc.fixturenames and param_name not in existing_markers:
        metafunc.parametrize(
            param_name,
            param_value,
            ids=lambda val: f"{param_name}({val})",
        )


def pytest_collection_modifyitems(config, items):
    """Modify tests at collection time"""
    _mark_all_e2e(items)

    _skip_unsupported_compiler_tests(config, items)

    sort_tests_for_llm_caching(items)

    with open(".test_sort.txt", "w") as f:
        for item in items:
            f.write("\n")
            f.write(str(item.listnames()[-2]) + " " + item.name + "\n")
            f.write(str(SortKey.from_item(item)) + "\n")


def _mark_all_e2e(items):
    """Mark all tests within the e2e package with the e2e marker"""
    for item in items:
        if "e2e" in str(item.nodeid):
            item.add_marker(pytest.mark.e2e)


def _skip_unsupported_compiler_tests(config, items):
    """Skip all tests that need compiler changes to run.
    This can be relaxed once the compiler changes are in place
    """

    markexpr = config.option.markexpr
    if "compiler_support_32k" in markexpr:
        return  # let pytest handle the collection logic

    skip_marker = pytest.mark.skip(reason="Needs compiler changes")
    for item in items:
        if "spyre" in item.keywords and "compiler_support_32k" in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture()
def use_llm_cache():
    """Fixture for test sorting to denote that this should use a cached LLM
    instance"""


@pytest.fixture(autouse=True)
def init_test_http_connection():
    # pytest_asyncio may use a different event loop per test
    # so we need to make sure the async client is created anew
    global_http_connection.reuse_client = False


@pytest.fixture()
def should_do_global_cleanup_after_test(request) -> bool:
    """Allow subdirectories to skip global cleanup by overriding this fixture.
    This can provide a ~10x speedup for non-GPU unit tests since they don't need
    to initialize torch.
    """

    return not request.node.get_closest_marker("skip_global_cleanup")


@pytest.fixture(autouse=True)
def cleanup_fixture(should_do_global_cleanup_after_test: bool):
    yield
    if should_do_global_cleanup_after_test:
        cleanup_dist_env_and_memory()


@pytest.fixture(autouse=True)
def dynamo_reset():
    yield
    torch._dynamo.reset()


# See https://github.com/okken/pytest-runtime-xfail/blob/master/pytest_runtime_xfail.py
# This allows us to conditionally set expected failures at test runtime
@pytest.fixture()
def runtime_xfail(request):
    """
    Call runtime_xfail() to mark running test as xfail.
    """

    def _xfail(reason=""):
        request.node.add_marker(pytest.mark.xfail(reason=reason))

    return _xfail


@pytest.fixture(scope="function")
def remote_openai_server(request):
    """Fixture to set up a test server."""
    params = request.node.callspec.params

    try:
        model = params["model"]
        backend = params["backend"]
    except KeyError as e:
        raise pytest.UsageError("Error setting up remote_openai_server params") from e

    # Default to None if not present
    quantization = params.get("quantization", None)

    # Add extra server args if present in test
    server_args = ["--quantization", quantization] if quantization else []

    if "tp_size" in params:
        tp_size = params["tp_size"]
        if int(tp_size) > 1:
            # Don't set tp size explicitly if it's 1
            skip_unsupported_tp_size(int(tp_size), backend)
            server_args.extend(["--tensor-parallel-size", str(tp_size)])

    if "mode" in params and params["mode"] in ["cb", "cp", "pc"]:
        max_model_len = params["max_model_len"]
        max_num_seqs = params["max_num_seqs"]
        env_dict = {"VLLM_SPYRE_USE_CB": "1", "VLLM_SPYRE_DYNAMO_BACKEND": backend}
        server_args.extend(
            ["--max_num_seqs", str(max_num_seqs), "--max-model-len", str(max_model_len)]
        )
        # Chunked prefill extra
        if params["mode"] in ["cp", "pc"]:
            env_dict.update({"VLLM_SPYRE_USE_CHUNKED_PREFILL": "1"})
            server_args.extend(
                [
                    "--max_num_batched_tokens",
                    str(128),
                ]
            )
        if params["mode"] == "pc":
            server_args.extend(
                [
                    "--enable-prefix-caching",
                ]
            )

    else:
        warmup_shapes = params["warmup_shapes"]
        warmup_prompt_length = [t[0] for t in warmup_shapes]
        warmup_new_tokens = [t[1] for t in warmup_shapes]
        warmup_batch_size = [t[2] for t in warmup_shapes]
        env_dict = {
            "VLLM_SPYRE_WARMUP_PROMPT_LENS": ",".join(map(str, warmup_prompt_length)),
            "VLLM_SPYRE_WARMUP_NEW_TOKENS": ",".join(map(str, warmup_new_tokens)),
            "VLLM_SPYRE_WARMUP_BATCH_SIZES": ",".join(map(str, warmup_batch_size)),
            "VLLM_SPYRE_DYNAMO_BACKEND": backend,
        }

    try:
        server = get_cached_api_server(model, server_args=server_args, server_env=env_dict)
        yield server
    except Exception as e:
        pytest.fail(f"Failed to setup server: {e}")


@pytest.fixture(scope="session", autouse=True)
def teardown_fixture():
    # Session scoped fixture will run once for the entire suite
    yield

    # Clear out any cached LLMs so no subprocesses get orphaned
    clear_llm_caches()
    print_llm_cache_info()


@pytest.fixture
def set_random_seed(request):
    func_hash = hashlib.sha256(request.node.originalname.encode("utf-8"))
    seed = int(func_hash.hexdigest(), 16)
    random.seed(seed)
    yield


@pytest.fixture()
def temporary_enable_log_propagate():
    """Context manager to temporarily enable log propagation."""
    import logging

    logger = logging.getLogger("vllm_spyre")
    logger.propagate = True
    yield
    logger.propagate = False


@pytest.fixture()
def caplog_vllm_spyre(temporary_enable_log_propagate, caplog):
    # To capture vllm-spyre log, we should enable propagate=True temporarily
    # because caplog depends on logs propagated to the root logger.
    yield caplog


@pytest.fixture(scope="function", autouse=True)
def clear_env_cache():
    envs.clear_env_cache()

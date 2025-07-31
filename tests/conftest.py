import hashlib
import os
import random

import pytest
import torch
from spyre_util import RemoteOpenAIServer, skip_unsupported_tp_size
from vllm.connections import global_http_connection
from vllm.distributed import cleanup_dist_env_and_memory

# Running with "fork" can lead to hangs/crashes
# Specifically, our use of transformers to compare results causes an OMP thread
# pool to be created, which is then lost when the next test launches vLLM and
# forks a worker.
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def pytest_collection_modifyitems(config, items):
    """ Mark all tests in e2e directory"""
    for item in items:
        if "e2e" in str(item.nodeid):
            item.add_marker(pytest.mark.e2e)


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

    def _xfail(reason=''):
        request.node.add_marker(pytest.mark.xfail(reason=reason))

    return _xfail


@pytest.fixture(scope="function")
def remote_openai_server(request):
    """ Fixture to set up a test server."""

    params = request.node.callspec.params

    try:
        model = params['model']
        backend = params['backend']
    except KeyError as e:
        raise pytest.UsageError(
            "Error setting up remote_openai_server params") from e

    if 'cb' in params:
        max_model_len = params["max_model_len"]
        max_num_seqs = params["max_num_seqs"]
        env_dict = {
            "VLLM_SPYRE_USE_CB": "1",
            "VLLM_SPYRE_DYNAMO_BACKEND": backend
        }
        server_args = [
            "--max_num_seqs",
            str(max_num_seqs), "--max-model-len",
            str(max_model_len)
        ]

    else:
        warmup_shape = params['warmup_shape']
        warmup_prompt_length = [t[0] for t in warmup_shape]
        warmup_new_tokens = [t[1] for t in warmup_shape]
        warmup_batch_size = [t[2] for t in warmup_shape]
        env_dict = {
            "VLLM_SPYRE_WARMUP_PROMPT_LENS":
            ','.join(map(str, warmup_prompt_length)),
            "VLLM_SPYRE_WARMUP_NEW_TOKENS":
            ','.join(map(str, warmup_new_tokens)),
            "VLLM_SPYRE_WARMUP_BATCH_SIZES":
            ','.join(map(str, warmup_batch_size)),
            "VLLM_SPYRE_DYNAMO_BACKEND":
            backend,
        }

        # Default to None if not present
        quantization = params.get('quantization', None)

        # Add extra server args if present in test
        server_args = ["--quantization", quantization] if quantization else []

        if 'tp_size' in params:
            tp_size = params['tp_size']
            skip_unsupported_tp_size(int(tp_size), backend)
            server_args.extend(["--tensor-parallel-size", str(tp_size)])

    try:
        with RemoteOpenAIServer(model, server_args,
                                env_dict=env_dict) as server:
            yield server
    except Exception as e:
        pytest.fail(f"Failed to setup server: {e}")


@pytest.fixture
def set_random_seed(request):
    func_hash = hashlib.sha256(request.node.originalname.encode('utf-8'))
    seed = int(func_hash.hexdigest(), 16)
    random.seed(seed)
    yield

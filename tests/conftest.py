import pytest
import torch
from spyre_util import RemoteOpenAIServer
from vllm.connections import global_http_connection
from vllm.distributed import cleanup_dist_env_and_memory


@pytest.fixture(params=[True, False])
def run_with_both_engines(request, monkeypatch):
    # Automatically runs tests twice, once with V1 and once without
    use_v1 = request.param
    # Tests decorated with `@skip_v1` are only run without v1
    skip_v1 = request.node.get_closest_marker("skip_v1")

    if use_v1:
        if skip_v1:
            pytest.skip("Skipping test on vllm V1")
        monkeypatch.setenv('VLLM_USE_V1', '1')
    else:
        monkeypatch.setenv('VLLM_USE_V1', '0')

    yield


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


@pytest.fixture(scope="function")
def remote_openai_server(request):
    """ Fixture to set up a test server."""

    params = request.node.callspec.params

    # Extract parameters from the test function for the server
    model = params['model']
    warmup_shape = params['warmup_shape']
    backend = params['backend']
    vllm_version = params['vllm_version']

    warmup_prompt_length = [t[0] for t in warmup_shape]
    warmup_new_tokens = [t[1] for t in warmup_shape]
    warmup_batch_size = [t[2] for t in warmup_shape]
    v1_flag = "1" if vllm_version == "V1" else "0"
    env_dict = {
        "VLLM_SPYRE_WARMUP_PROMPT_LENS":
        ','.join(map(str, warmup_prompt_length)),
        "VLLM_SPYRE_WARMUP_NEW_TOKENS": ','.join(map(str, warmup_new_tokens)),
        "VLLM_SPYRE_WARMUP_BATCH_SIZES": ','.join(map(str, warmup_batch_size)),
        "VLLM_SPYRE_DYNAMO_BACKEND": backend,
        "VLLM_USE_V1": v1_flag
    }

    # Add extra server args if present in test
    server_args = []
    if 'tensor_parallel_size' in params:
        tp_size = params['tensor_parallel_size']
        server_args.extend(["--tensor-parallel-size", str(tp_size)])

    try:
        with RemoteOpenAIServer(model, server_args,
                                env_dict=env_dict) as server:
            yield server
    except Exception as e:
        pytest.fail(f"Failed to setup server: {e}")

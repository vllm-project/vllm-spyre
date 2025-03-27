import pytest
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils import FlexibleArgumentParser, get_open_port
from spyre_util import RemoteOpenAIServer

from tests.spyre_util import get_spyre_backend_list, get_spyre_model_list


@pytest.mark.parametrize("model", get_spyre_model_list())
# (prompt_length/new_tokens/batch_size)
@pytest.mark.parametrize("warmup_shape", [[
    (64, 20, 4),
]])
@pytest.mark.parametrize("backend", get_spyre_backend_list())
@pytest.mark.parametrize("tensor_parallel_size", [2])
@pytest.mark.parametrize("vllm_version", ["V0", "V1"])
def test_openai_serving(model, warmup_shape, backend, tensor_parallel_size, vllm_version):
    """Test online serving with tensor parallelism using the `vllm serve` CLI"""

    # TODO: util or fixture-ize
    warmup_prompt_length = [t[0] for t in warmup_shape]
    warmup_new_tokens = [t[1] for t in warmup_shape]
    warmup_batch_size = [t[2] for t in warmup_shape]
    v1_flag = "1" if vllm_version == "V1" else "0"
    env_dict = {
        "VLLM_SPYRE_WARMUP_PROMPT_LENS":
        ','.join(str(val) for val in warmup_prompt_length),
        "VLLM_SPYRE_WARMUP_NEW_TOKENS":
        ','.join(str(val) for val in warmup_new_tokens),
        "VLLM_SPYRE_WARMUP_BATCH_SIZES":
        ','.join(str(val) for val in warmup_batch_size),
        "VLLM_SPYRE_DYNAMO_BACKEND":
        backend,
        "VLLM_USE_V1":
        v1_flag
    }

    with RemoteOpenAIServer(model, [], env_dict=env_dict, tensor_parallel_size=tensor_parallel_size) as server:
        # Run a few simple requests to make sure the server works.
        # This is not checking correctness of replies
        client = server.get_client()
        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=5,
                                               temperature=0.0)
        assert len(completion.choices) == 1
        assert len(completion.choices[0].text) > 0

        completion = client.completions.create(model=model,
                                               prompt="Hello World!",
                                               max_tokens=5,
                                               temperature=1.0,
                                               n=2)
        assert len(completion.choices) == 2
        assert len(completion.choices[0].text) > 0
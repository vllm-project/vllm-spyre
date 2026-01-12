import pytest
from pydantic import ValidationError
from huggingface_hub.errors import LocalEntryNotFoundError
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm import EngineArgs


from vllm_spyre.platform import SpyrePlatform
from spyre_util import environ_checkpoint, REFERENCE_MODELS

try:
    # old
    from vllm.utils import FlexibleArgumentParser
except ImportError:
    # new
    from vllm.utils.argparse_utils import FlexibleArgumentParser

global_default = 192


# The default chunk size we set for all models when chunked prefill
# is enabled is 1024. For some models (granite-3.3-8b-instruct)
# we force a model-specific value that the user can't change except
# by using VLLM_DT_CHUNK_LEN. In the case of granite the default is
# also 1024, so to be able to verify that the special model logic
# is working, we change the default to 192 (3 blocks), which is very
# unlikely to be used for specific models in the future.
# In the parametrization below, micro-g3 represents a generic model
# which will use the global default.
@pytest.mark.parametrize(
    "model_name, chunk_size",
    [
        pytest.param("ibm-ai-platform/micro-g3.3-8b-instruct-1b", global_default),
        pytest.param(
            "ibm-granite/granite-3.3-8b-instruct",
            1024,
            marks=[pytest.mark.full_model, pytest.mark.multi],
        ),
    ],
)
def test_generic_model_chunk_size_default(
    model_name: str, chunk_size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VLLM_SPYRE_USE_CB", "1")

    # Change the default so that we can differentiate the global
    # default from model-specific defaults.
    monkeypatch.setattr(SpyrePlatform, "DEFAULT_CHUNK_SIZE", global_default)

    # Some configuration code paths are only activate with sendnn and tp=4.
    # We want to enable them because they are the ones we care about for production.
    # But we need to patch sendnn_configured to prevent exceptions
    monkeypatch.setenv("VLLM_SPYRE_DYNAMO_BACKEND", "sendnn")

    def sendnn_configured() -> bool:
        return False

    monkeypatch.setattr(SpyrePlatform, "sendnn_configured", sendnn_configured)
    model = REFERENCE_MODELS[model_name]
    common_args = [
        "--model",
        model.name,
        "--revision",
        model.revision,
        "--max-model-len",
        "64",
        "--max-num-seqs",
        "32",
        "-tp",
        "4",
        "--swap-space",  # to prevent a validation error in the 16GB memory test env.
        "1",
    ]

    with environ_checkpoint():
        # Test that the upstream default is None but is changed to 2048 by
        # the VllmConfig initialization (when using CLI), but is then
        # overridden in our platform.py
        try:
            engine_args = _build_engine_args(
                [
                    *common_args,
                ]
            )
        except LocalEntryNotFoundError:
            pytest.skip("Skipping test of model not found in local cache")

        assert engine_args.max_num_batched_tokens is None
        vllm_config = engine_args.create_engine_config()
        # TODO: this behavior of changing the engine_args was introduced in v0.12.0.
        # Uncomment below when this version becomes the lowest supported version.
        # assert engine_args.max_num_batched_tokens == 2048
        # we override this in platform.py
        assert (
            vllm_config.scheduler_config.max_num_batched_tokens
            == vllm_config.model_config.max_model_len * vllm_config.scheduler_config.max_num_seqs
        )

    with environ_checkpoint():
        # Test that we override the default and even the user provided setting
        engine_args = _build_engine_args([*common_args, "--max-num-batched-tokens", "128"])
        assert engine_args.max_num_batched_tokens == 128
        vllm_config = engine_args.create_engine_config()
        assert engine_args.max_num_batched_tokens == 128
        # we override this in platform.py
        assert (
            vllm_config.scheduler_config.max_num_batched_tokens
            == vllm_config.model_config.max_model_len * vllm_config.scheduler_config.max_num_seqs
        )

    # Enable CP
    monkeypatch.setenv("VLLM_SPYRE_USE_CHUNKED_PREFILL", "1")

    with environ_checkpoint():
        # Test that the default is 192 when CP is enabled
        engine_args = _build_engine_args(
            [
                *common_args,
            ]
        )
        assert engine_args.max_num_batched_tokens == 192
        vllm_config = engine_args.create_engine_config()
        assert engine_args.max_num_batched_tokens == 192
        assert vllm_config.scheduler_config.max_num_batched_tokens == chunk_size

    with environ_checkpoint():
        # Test that we can still change the default
        engine_args = _build_engine_args([*common_args, "--max-num-batched-tokens", "128"])
        assert engine_args.max_num_batched_tokens == 128
        vllm_config = engine_args.create_engine_config()
        assert engine_args.max_num_batched_tokens == 128
        if chunk_size == global_default:
            assert vllm_config.scheduler_config.max_num_batched_tokens == 128
        else:
            assert vllm_config.scheduler_config.max_num_batched_tokens == chunk_size

    with environ_checkpoint():
        # Test that an invalid value will trigger an error (42 is not a multiple of the block size)
        engine_args = _build_engine_args([*common_args, "--max-num-batched-tokens", "42"])
        assert engine_args.max_num_batched_tokens == 42
        if chunk_size == global_default:
            with pytest.raises(ValidationError):
                vllm_config = engine_args.create_engine_config()
        else:
            assert vllm_config.scheduler_config.max_num_batched_tokens == chunk_size

    monkeypatch.setenv("VLLM_DT_CHUNK_LEN", "512")

    with environ_checkpoint():
        # Verify that VLLM_DT_CHUNK_LEN overrides the default
        engine_args = _build_engine_args(
            [
                *common_args,
            ]
        )
        assert engine_args.max_num_batched_tokens == 192
        vllm_config = engine_args.create_engine_config()
        assert engine_args.max_num_batched_tokens == 192
        assert vllm_config.scheduler_config.max_num_batched_tokens == 512

    with environ_checkpoint():
        # Verify that VLLM_DT_CHUNK_LEN overrides the cli setting
        engine_args = _build_engine_args([*common_args, "--max-num-batched-tokens", "64"])
        assert engine_args.max_num_batched_tokens == 64
        vllm_config = engine_args.create_engine_config()
        assert engine_args.max_num_batched_tokens == 64
        assert vllm_config.scheduler_config.max_num_batched_tokens == 512


def test_prefix_caching_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    model = REFERENCE_MODELS["ibm-ai-platform/micro-g3.3-8b-instruct-1b"]
    common_args = [
        "--model",
        model.name,
        "--revision",
        model.revision,
        "--max-model-len",
        "1024",
    ]

    with (
        environ_checkpoint(),
        pytest.MonkeyPatch.context() as m,
    ):
        # Verify that prefix caching is on by default unless we change that in
        # pre_register_and_update
        def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
            pass

        m.setattr(SpyrePlatform, "pre_register_and_update", pre_register_and_update)

        engine_args = _build_engine_args(
            [
                *common_args,
            ]
        )
        assert engine_args.enable_prefix_caching is None
        vllm_config = engine_args.create_engine_config()
        assert engine_args.enable_prefix_caching
        assert vllm_config.cache_config.enable_prefix_caching

    with environ_checkpoint():
        # Verify that prefix caching is off by default with vllm-spyre
        engine_args = _build_engine_args(
            [
                *common_args,
            ]
        )
        assert not engine_args.enable_prefix_caching
        vllm_config = engine_args.create_engine_config()
        assert not engine_args.enable_prefix_caching
        assert not vllm_config.cache_config.enable_prefix_caching

    with environ_checkpoint():
        # Test that it can be enabled
        engine_args = _build_engine_args([*common_args, "--enable-prefix-caching"])
        assert engine_args.enable_prefix_caching
        vllm_config = engine_args.create_engine_config()
        assert engine_args.enable_prefix_caching
        assert vllm_config.cache_config.enable_prefix_caching


def _build_engine_args(cli_args: list[str]) -> EngineArgs:
    parser = FlexibleArgumentParser(description="vLLM's remote OpenAI server.")
    parser = make_arg_parser(parser)
    args = parser.parse_args(cli_args)

    engine_args = EngineArgs.from_cli_args(args)

    return engine_args

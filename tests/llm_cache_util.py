"""Contains utilities for LLM caching"""

from typing import NamedTuple

from spyre_util import DecodeWarmupShapes
from vllm import LLM


def force_engine_shutdown(llm: LLM):
    """
    🌶️🌶️🌶️
    This hack is here because of an issue in vllm 0.9.2+ where a circular
    reference occurs in vllm.executor.ray_utils if ray is not installed. This
    circular reference holds a copy of the vllm config which contains a
    reference to the LLM, which means it can never be garbage collected.
    Since vllm.LLM relies on garbage collection to shut down its engine, the
    engine never shuts down. When running tensor parallel workloads, if the
    engine is never shut down then the TP worker processes are never killed.
    When the TP worker processes are held open, all future attempts to create a
    new engine will fail with an EADDRINUSE error.
    🌶️🌶️🌶️
    """
    llm.llm_engine.engine_core.shutdown()


def sort_tests_for_llm_caching(items: list) -> None:
    """Sorts a list of pytest cases based on the LLM parameterizations.

    This allows us to group tests together that use the same model and config,
    which means they can reuse the underlying LLM. Then we can cache the LLM
    across tests to save time.

    This is important because spinning up a new vLLM engine from scratch takes
    a decent amount of time, even with the torch compilation cache active. LLM
    creation dominates the runtime of our test suites.

    This sorts the `items` list in-place.
    """
    items.sort(key=SortKey.from_item)


class SortKey(NamedTuple):
    """Sort key that groups by runtime configuration.

    The order of attributes is important here and controls the test
    grouping.
    """

    cache_type: str  # None (empty str), online, llm, engine
    backend: str = ""
    model: str = ""
    tp_size: int = 1
    use_cb: bool = False
    max_model_len: int = 0
    max_num_seqs: int = 0
    num_blocks: int = 0
    warmup_shapes: DecodeWarmupShapes | None = None

    @staticmethod
    def from_item(item) -> "SortKey":
        cache_type = SortKey._get_cache_type(item)
        if not cache_type:
            # Don't add any extra re-ordering logic for tests that won't utilize
            # the cache
            return SortKey(cache_type=cache_type)

        if not hasattr(item, "callspec"):
            # This isn't great- we probably want to cache but can't because the
            # test has no parameters at all
            return SortKey(cache_type="")

        use_cb = SortKey._uses_cb(item)
        if use_cb:
            sort_kwargs = {
                "max_model_len": SortKey._get_max_model_len(item),
                "max_num_seqs": SortKey._get_max_num_seqs(item),
            }
        else:
            sort_kwargs = {
                "warmup_shapes": SortKey._get_warmup_shapes(item),
            }

        return SortKey(
            cache_type=cache_type,
            model=SortKey._get_model(item),
            backend=SortKey._get_backend(item),
            tp_size=SortKey._get_tp_size(item),
            use_cb=SortKey._uses_cb(item),
            num_blocks=SortKey._get_num_blocks(item),
            **sort_kwargs,
        )

    @staticmethod
    def _get_cache_type(item) -> str:
        # If not an e2e test then assume no cache
        if "e2e" not in item.listnames():
            return ""

        if "remote_openai_server" in item.fixturenames:
            # (Not actually caching these yet, but can in future)
            return "online"

        if "use_llm_cache" in item.fixturenames:
            return "llm"

        if "test_spyre_cb_scheduler_steps.py" in item.listnames():
            # Not currently cached and needs updating to fixture name
            # CB step tests require a raw engine for scheduler access
            return "engine"

        # Else shouldn't be using any cache
        return ""

    @staticmethod
    def _uses_cb(item) -> bool:
        """True if the test uses continuous batching, false for static batching.
        Checks for the pytest.mark.cb mark."""
        markers = {mark.name for mark in item.own_markers}
        return "cb" in markers

    @staticmethod
    def _get_max_model_len(item) -> int:
        params = item.callspec.params
        if "max_model_len" in params:
            SortKey._assert_param(
                isinstance(params["max_model_len"], int),
                "max_model_len must be an int",
                item,
            )
            return params["max_model_len"]
        # Put `-1` to indicate that this couldn't be found
        return -1

    @staticmethod
    def _get_max_num_seqs(item) -> int:
        params = item.callspec.params
        if "max_num_seqs" in params:
            SortKey._assert_param(
                isinstance(params["max_num_seqs"], int),
                "max_num_seqs must be an int",
                item,
            )
            return params["max_num_seqs"]
        # Put `-1` to indicate that this couldn't be found
        return -1

    @staticmethod
    def _get_warmup_shapes(item) -> list[tuple[int, int, int]]:
        key = "warmup_shapes"
        params = item.callspec.params
        if key in params:
            shapes = params[key]
            SortKey._assert_param(isinstance(shapes, list),
                                  "Warmup shape must be a list of tuples",
                                  item)
            SortKey._assert_param(
                isinstance(shapes[0], tuple),
                "Warmup shape must be a list of tuples",
                item,
            )
            return params[key]
        # Use -1s to indicate that this couldn't be found
        return [
            (-1, -1, -1),
        ]

    @staticmethod
    def _get_tp_size(item) -> int:
        TP_KEYS = ["tp_size", "tensor_parallel_size", "tp"]
        params = item.callspec.params
        for key in TP_KEYS:
            if key in params:
                SortKey._assert_param(isinstance(params[key], int),
                                      "tp size must be an int", item)
                return params[key]
        # Assume no TP if not set
        return 1

    @staticmethod
    def _get_model(item) -> str:
        MODEL_KEYS = ["model", "model_name"]
        params = item.callspec.params
        for key in MODEL_KEYS:
            if key in params:
                SortKey._assert_param(isinstance(params[key], str),
                                      "model must be a string", item)
                return params[key]
        # No assumption about default model, we likely don't need an llm if this
        # isn't set
        return ""

    @staticmethod
    def _get_backend(item) -> str:
        if "backend" in item.callspec.params:
            backend = item.callspec.params["backend"]
            # if isinstance(backend, tuple) and len(backend) == 1:
            #     backend = backend[0]

            SortKey._assert_param(isinstance(backend, str),
                                  "backend must be a string.", item)
            return backend
        # If backend isn't given then this is likely a spyre-only test
        return "sendnn"

    @staticmethod
    def _get_num_blocks(item) -> int:
        if "available_blocks" in item.callspec.params:
            blocks = item.callspec.params["available_blocks"]
            SortKey._assert_param(isinstance(blocks, int),
                                  "available_blocks must be an int.", item)
            return blocks
        # Most tests don't use this param
        return 0

    @staticmethod
    def _assert_param(condition, message, item):
        assert condition, (message + f"\n\n\tTest: {item.listnames()}"
                           f"\n\n\tParams: {item.callspec.params}")

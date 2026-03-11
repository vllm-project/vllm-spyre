"""
pytest11 plugin for vllm-spyre-next.

Auto-activates when vllm-spyre-next is installed. Disable with:
    pytest -p no:spyre_next_test

Responsibilities:
  1. Filter upstream vLLM tests via declarative YAML config (upstream_tests.yaml)
  2. Apply global skip rules (e.g. CUDA device parametrizations)
  3. Provide the default_vllm_config fixture with Spyre-specific config
"""
from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any

import pytest
import yaml

from vllm_spyre_next.testing.models import (
    AllowEntry,
    BlockEntry,
    FileConfig,
    ParamOverride,
    ParamSkip,
    UpstreamTestConfig,
)

_YAML_FILENAME = "upstream_tests.yaml"
_YAML_PATH = Path(__file__).parent / _YAML_FILENAME


def _load_upstream_config() -> UpstreamTestConfig:
    """Load and parse upstream_tests.yaml from the same directory as this file."""
    if not _YAML_PATH.exists():
        return UpstreamTestConfig()

    with open(_YAML_PATH) as f:
        raw = yaml.safe_load(f)

    if not raw or "tests" not in raw or "files" not in raw["tests"]:
        return UpstreamTestConfig()

    return _parse_config(raw["tests"])


def _parse_config(raw_tests: dict) -> UpstreamTestConfig:
    """Parse the 'tests' section of the YAML into typed dataclasses."""
    files: list[FileConfig] = []

    for file_entry in raw_tests.get("files", []):
        allow_list: list[AllowEntry] = []
        for allow in file_entry.get("allow_list", []):
            params_section = allow.get("params", {})

            param_skips: list[ParamSkip] = []
            for param_name, values in params_section.get("skip", {}).items():
                param_skips.append(
                    ParamSkip(param_name=param_name, values=frozenset(values))
                )

            param_overrides: list[ParamOverride] = []
            for param_name, values in params_section.get("override", {}).items():
                param_overrides.append(
                    ParamOverride(param_name=param_name, values=tuple(values))
                )

            allow_list.append(
                AllowEntry(
                    test=allow["test"],
                    mode=allow.get("mode", "mandatory_pass"),
                    tags=tuple(allow.get("tags", [])),
                    param_skips=tuple(param_skips),
                    param_overrides=tuple(param_overrides),
                )
            )

        block_list = [BlockEntry(test=b["test"]) for b in file_entry.get("block_list", [])]

        files.append(
            FileConfig(
                rel_path=file_entry["rel_path"],
                allow_list=tuple(allow_list),
                block_list=tuple(block_list),
            )
        )

    return UpstreamTestConfig(files=tuple(files))


# Load once at import time
_UPSTREAM_CONFIG: UpstreamTestConfig = _load_upstream_config()


# ---------------------------------------------------------------------------
# Upstream YAML filtering
# ---------------------------------------------------------------------------


def _apply_upstream_yaml_filter(
    config: pytest.Config,
    items: list[pytest.Item],
    upstream_config: UpstreamTestConfig,
) -> None:
    """Apply YAML allow_list/block_list filtering to upstream tests.

    Only affects tests whose file path is under ``config._upstream_tests_base``.
    Tests not listed in the YAML config are skipped (opt-in model).
    ``block_list`` takes precedence over ``allow_list``.
    """
    upstream_tests_base = getattr(config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    upstream_tests_base = Path(upstream_tests_base).resolve()
    # rel_path in YAML is relative to the upstream repo root, which is the
    # parent of the tests/ directory.
    upstream_repo_root = upstream_tests_base.parent

    # Build lookup: resolved absolute path -> FileConfig
    file_configs: dict[Path, FileConfig] = {}
    for fc in upstream_config.files:
        abs_path = (upstream_repo_root / fc.rel_path).resolve()
        file_configs[abs_path] = fc

    for item in items:
        test_path = Path(item.fspath).resolve()

        # Only filter upstream tests.
        if not test_path.is_relative_to(upstream_tests_base):
            continue

        fc = _find_file_config(test_path, file_configs)
        if fc is None:
            item.add_marker(
                pytest.mark.skip(reason=f"upstream test file not in {_YAML_FILENAME}")
            )
            continue

        test_name = item.originalname or item.name

        # block_list takes precedence.
        if _matches_block_list(test_name, fc.block_list):
            item.add_marker(
                pytest.mark.skip(reason=f"blocked by {_YAML_FILENAME} block_list")
            )
            continue

        allow_entry = _find_allow_entry(test_name, fc.allow_list)
        if allow_entry is None:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"test not in {_YAML_FILENAME} allow_list for {fc.rel_path}"
                )
            )
            continue

        # Parameter-level skips within the allowed test.
        if _should_skip_params(item, allow_entry):
            item.add_marker(
                pytest.mark.skip(reason=f"parameter combination skipped by {_YAML_FILENAME}")
            )
            continue

        # Apply mode-based markers.
        if allow_entry.mode == "xfail":
            item.add_marker(pytest.mark.xfail(
                reason=f"expected failure per {_YAML_FILENAME}", strict=False,
            ))
        elif allow_entry.mode == "xfail_strict":
            item.add_marker(pytest.mark.xfail(
                reason=f"expected strict failure per {_YAML_FILENAME}", strict=True,
            ))


def _find_file_config(
    test_path: Path, file_configs: dict[Path, FileConfig],
) -> FileConfig | None:
    """Find the FileConfig for a test file path.

    Supports both exact file matches and directory-level configs.
    """
    # Exact file match first.
    if test_path in file_configs:
        return file_configs[test_path]

    # Directory prefix match (rel_path points to a directory).
    for config_path, fc in file_configs.items():
        if test_path.is_relative_to(config_path):
            return fc

    return None


def _matches_block_list(test_name: str, block_list: tuple[BlockEntry, ...]) -> bool:
    """Return True if test_name matches any block_list entry."""
    return any(fnmatch.fnmatch(test_name, entry.test) for entry in block_list)


def _find_allow_entry(
    test_name: str, allow_list: tuple[AllowEntry, ...],
) -> AllowEntry | None:
    """Return the first matching allow_list entry, or None."""
    for entry in allow_list:
        if fnmatch.fnmatch(test_name, entry.test):
            return entry
    return None


def _should_skip_params(item: pytest.Item, allow_entry: AllowEntry) -> bool:
    """Return True if the item's parameters match any param skip rule."""
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return False

    for ps in allow_entry.param_skips:
        if ps.param_name in callspec.params:
            if callspec.params[ps.param_name] in ps.values:
                return True
    return False


# ---------------------------------------------------------------------------
# Hook: pytest_generate_tests — override parametrize values from YAML
# ---------------------------------------------------------------------------


def _resolve_upstream_repo_root(config: pytest.Config) -> Path | None:
    """Return the upstream repo root, or None if upstream tests are not active."""
    base = getattr(config, "_upstream_tests_base", None)
    if base is None:
        return None
    return Path(base).resolve().parent


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Replace upstream parametrize values with Spyre-compatible ranges.

    Reads ``params.override`` from upstream_tests.yaml and rewrites
    ``@pytest.mark.parametrize`` markers in-place before test generation.
    """
    upstream_repo_root = _resolve_upstream_repo_root(metafunc.config)
    if upstream_repo_root is None:
        return

    test_path = Path(metafunc.definition.fspath).resolve()
    upstream_tests_base = (upstream_repo_root / "tests").resolve()
    if not test_path.is_relative_to(upstream_tests_base):
        return

    # Build file config lookup (same logic as collection filtering).
    file_configs: dict[Path, FileConfig] = {}
    for fc in _UPSTREAM_CONFIG.files:
        file_configs[(upstream_repo_root / fc.rel_path).resolve()] = fc

    fc = _find_file_config(test_path, file_configs)
    if fc is None:
        return

    test_name = metafunc.definition.originalname or metafunc.definition.name
    allow_entry = _find_allow_entry(test_name, fc.allow_list)
    if allow_entry is None or not allow_entry.param_overrides:
        return

    for po in allow_entry.param_overrides:
        if po.param_name not in metafunc.fixturenames:
            #TODO: Add logging here that we are skippiung the param name
            continue
        for i, marker in enumerate(metafunc.definition.own_markers):
            if marker.name == "parametrize" and marker.args[0] == po.param_name:
                new_marker = pytest.mark.parametrize(po.param_name, list(po.values))
                metafunc.definition.own_markers[i] = new_marker.mark
                break


# ---------------------------------------------------------------------------
# Hook: pytest_collection_modifyitems — YAML filter for upstream tests
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Apply upstream YAML filtering to collected tests (opt-in model)."""
    _apply_upstream_yaml_filter(config, items, _UPSTREAM_CONFIG)


# ---------------------------------------------------------------------------
# Fixture: default_vllm_config — Spyre-specific override
# ---------------------------------------------------------------------------

def _spyre_default_vllm_config(monkeypatch):
    """Spyre-specific VllmConfig. Plain generator — NOT @pytest.fixture decorated.

    Used as a replacement for fixturedef.func in the pytest_fixture_setup hook.
    Must NOT be decorated with @pytest.fixture (would trigger pytest 8 call guard).
    """
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.platforms import PlatformEnum, current_platform
    from vllm.plugins import load_general_plugins

    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)
    load_general_plugins()

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    with set_current_vllm_config(config):
        yield


@pytest.fixture()
def default_vllm_config(monkeypatch):
    """Fallback fixture when no upstream conftest defines default_vllm_config."""
    yield from _spyre_default_vllm_config(monkeypatch)


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    """Swap any default_vllm_config fixture (including upstream conftest) with Spyre config.

    Modifies fixturedef in-place and returns None so the default
    pytest_fixture_setup hook executes our function via call_fixture_func
    (which handles generator teardown correctly).
    """
    if fixturedef.argname == "default_vllm_config":
        fixturedef.func = _spyre_default_vllm_config  # plain function, no call guard
        fixturedef.argnames = ("monkeypatch",)  # so pytest resolves the dependency

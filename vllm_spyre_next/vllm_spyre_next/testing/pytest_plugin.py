"""
pytest11 plugin for vllm-spyre-next.

When running pytest from the vllm repo, this plugin:
  1. Registers Spyre custom ops (RMSNorm etc.) early via load_general_plugins
  2. Filters tests via declarative YAML config (upstream_tests.yaml)
  3. Provides the default_vllm_config fixture with Spyre-specific config
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path

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
    if not _YAML_PATH.exists():
        return UpstreamTestConfig()
    with open(_YAML_PATH) as f:
        raw = yaml.safe_load(f)
    if not raw or "tests" not in raw or "files" not in raw["tests"]:
        return UpstreamTestConfig()
    return _parse_config(raw["tests"])


def _parse_config(raw_tests: dict) -> UpstreamTestConfig:
    files: list[FileConfig] = []
    for file_entry in raw_tests.get("files", []):
        allow_list: list[AllowEntry] = []
        for allow in file_entry.get("allow_list", []):
            params_section = allow.get("params", {})
            param_skips = [
                ParamSkip(param_name=k, values=frozenset(v))
                for k, v in params_section.get("skip", {}).items()
            ]
            param_overrides = [
                ParamOverride(param_name=k, values=tuple(v))
                for k, v in params_section.get("override", {}).items()
            ]
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


_UPSTREAM_CONFIG: UpstreamTestConfig = _load_upstream_config()


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config):
    """Register Spyre plugins and detect vllm repo."""
    # Set env vars BEFORE any vllm imports
    os.environ["VLLM_PLUGINS"] = "spyre_next"
    
    # Load plugins early to register custom ops before test modules import RMSNorm
    from vllm.plugins import load_general_plugins
    load_general_plugins()
    
    # Detect vllm repo
    rootdir = Path(config.rootdir)
    tests_dir = rootdir / "tests"
    vllm_pkg = rootdir / "vllm"
    config._upstream_tests_base = tests_dir if (tests_dir.is_dir() and vllm_pkg.is_dir()) else None


def _find_file_config(test_path: Path, file_configs: dict[Path, FileConfig]) -> FileConfig | None:
    if test_path in file_configs:
        return file_configs[test_path]
    for config_path, fc in file_configs.items():
        if test_path.is_relative_to(config_path):
            return fc
    return None


def _matches_block_list(test_name: str, block_list: tuple[BlockEntry, ...]) -> bool:
    return any(fnmatch.fnmatch(test_name, e.test) for e in block_list)


def _find_allow_entry(test_name: str, allow_list: tuple[AllowEntry, ...]) -> AllowEntry | None:
    for entry in allow_list:
        if fnmatch.fnmatch(test_name, entry.test):
            return entry
    return None


def _should_skip_params(item: pytest.Item, allow_entry: AllowEntry) -> bool:
    callspec = getattr(item, "callspec", None)
    if not callspec:
        return False
    for ps in allow_entry.param_skips:
        if ps.param_name in callspec.params and callspec.params[ps.param_name] in ps.values:
            return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    upstream_tests_base = getattr(config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    upstream_tests_base = Path(upstream_tests_base).resolve()
    upstream_repo_root = upstream_tests_base.parent
    file_configs = {(upstream_repo_root / fc.rel_path).resolve(): fc for fc in _UPSTREAM_CONFIG.files}

    for item in items:
        test_path = Path(item.fspath).resolve()
        if not test_path.is_relative_to(upstream_tests_base):
            continue

        fc = _find_file_config(test_path, file_configs)
        if fc is None:
            item.add_marker(pytest.mark.skip(reason=f"not in {_YAML_FILENAME}"))
            continue

        test_name = item.originalname or item.name
        if _matches_block_list(test_name, fc.block_list):
            item.add_marker(pytest.mark.skip(reason=f"blocked by {_YAML_FILENAME}"))
            continue

        allow_entry = _find_allow_entry(test_name, fc.allow_list)
        if allow_entry is None:
            item.add_marker(pytest.mark.skip(reason=f"not in allow_list"))
            continue

        if _should_skip_params(item, allow_entry):
            item.add_marker(pytest.mark.skip(reason=f"param skipped"))
            continue

        if allow_entry.mode == "xfail":
            item.add_marker(pytest.mark.xfail(strict=False))
        elif allow_entry.mode == "xfail_strict":
            item.add_marker(pytest.mark.xfail(strict=True))


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    upstream_tests_base = getattr(metafunc.config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    upstream_repo_root = Path(upstream_tests_base).resolve().parent
    test_path = Path(metafunc.definition.fspath).resolve()
    file_configs = {(upstream_repo_root / fc.rel_path).resolve(): fc for fc in _UPSTREAM_CONFIG.files}
    
    fc = _find_file_config(test_path, file_configs)
    if not fc:
        return

    test_name = metafunc.definition.originalname or metafunc.definition.name
    allow_entry = _find_allow_entry(test_name, fc.allow_list)
    if not allow_entry or not allow_entry.param_overrides:
        return

    for po in allow_entry.param_overrides:
        if po.param_name not in metafunc.fixturenames:
            continue
        for i, marker in enumerate(metafunc.definition.own_markers):
            if marker.name == "parametrize" and marker.args[0] == po.param_name:
                metafunc.definition.own_markers[i] = pytest.mark.parametrize(
                    po.param_name, list(po.values)
                ).mark
                break


def _spyre_default_vllm_config(monkeypatch):
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.config.compilation import CompilationConfig
    from vllm.platforms import PlatformEnum, current_platform

    monkeypatch.setattr(type(current_platform), "_enum", PlatformEnum.OOT)

    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(custom_ops=["all"]),
    )
    with set_current_vllm_config(config):
        yield


@pytest.fixture()
def default_vllm_config(monkeypatch):
    yield from _spyre_default_vllm_config(monkeypatch)


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef, request):
    # Only replace fixture when running upstream vLLM tests, not local plugin tests
    upstream_tests_base = getattr(request.config, "_upstream_tests_base", None)
    if upstream_tests_base and fixturedef.argname == "default_vllm_config":
        fixturedef.func = _spyre_default_vllm_config
        fixturedef.argnames = ("monkeypatch",)

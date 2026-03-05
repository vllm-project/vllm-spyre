"""
pytest11 plugin for vllm-spyre-next.

Auto-activates when vllm-spyre-next is installed. Disable with:
    pytest -p no:spyre_next_test

Responsibilities:
  1. Skip upstream vLLM test parametrizations that Spyre doesn't support
  2. Provide the default_vllm_config fixture with Spyre-specific config
"""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass(frozen=True)
class SkipRule:
    """A rule for skipping unsupported test parametrizations.

    Attributes:
        test_pattern: fnmatch glob matched against the test nodeid.
                      Use "*" to match all tests.
        reason:       Human-readable skip reason shown in pytest output.
        param_name:   Parametrized argument name to check.
                      If None, the rule matches on test_pattern alone.
        param_values: Parameter values that trigger the skip.
                      If None (and param_name is set), any value triggers it.
        xfail:        If True, mark as xfail instead of skip.
    """
    test_pattern: str
    reason: str
    param_name: str | None = None
    param_values: frozenset[Any] | None = None
    xfail: bool = False


# ---------------------------------------------------------------------------
# Rule table — edit this as Spyre support evolves
# ---------------------------------------------------------------------------

SKIP_RULES: list[SkipRule] = [
    # RMSNorm: residual not yet implemented
    SkipRule(
        "*test_rms_norm*",
        param_name="add_residual",
        param_values=frozenset([True]),
        reason="SpyreRMSNorm: residual not yet implemented",
    ),
    # RMSNorm: strided (non-contiguous) inputs not supported
    SkipRule(
        "*test_rms_norm*",
        param_name="strided_input",
        param_values=frozenset([True]),
        reason="SpyreRMSNorm: strided input not supported",
    ),
    # Fused quant layernorm: CUDA-only kernel test
    SkipRule(
        "*test_fused_rms_norm_quant*",
        reason="CUDA-only fused quant kernel, not applicable to Spyre",
    ),
    # Global: skip CUDA device parametrizations
    SkipRule(
        "*",
        param_name="device",
        param_values=frozenset([f"cuda:{i}" for i in range(8)] + ["cuda"]),
        reason="Spyre platform: CUDA devices not available",
    ),
]


# ---------------------------------------------------------------------------
# Hook: pytest_collection_modifyitems — apply skip rules
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Walk collected tests and apply skip/xfail markers from SKIP_RULES."""
    for item in items:
        for rule in SKIP_RULES:
            if _rule_matches(item, rule):
                if rule.xfail:
                    marker = pytest.mark.xfail(reason=rule.reason, strict=False)
                else:
                    marker = pytest.mark.skip(reason=rule.reason)
                item.add_marker(marker)
                break  # first matching rule wins


def _rule_matches(item: pytest.Item, rule: SkipRule) -> bool:
    """Return True if the test item matches the given SkipRule."""
    if not fnmatch.fnmatch(item.nodeid, rule.test_pattern):
        return False

    # No param_name means the rule matches on test_pattern alone
    if rule.param_name is None:
        return True

    callspec = getattr(item, "callspec", None)
    if callspec is None or rule.param_name not in callspec.params:
        return False

    if rule.param_values is None:
        return True

    return callspec.params[rule.param_name] in rule.param_values


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

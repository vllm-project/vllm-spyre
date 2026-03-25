"""Data models for the vllm-spyre-next test infrastructure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Upstream YAML config model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSkip:
    """A parameter-level skip rule within an allow_list entry."""

    param_name: str
    values: frozenset[Any]


@dataclass(frozen=True)
class ParamAllow:
    """A parameter-level allow rule within an allow_list entry.

    Only test cases with parameter values in this allowlist will run.
    If specified, this takes precedence over param_skips for the same parameter.
    """

    param_name: str
    values: frozenset[Any]


@dataclass(frozen=True)
class ParamOverride:
    """Replace upstream parametrize values with Spyre-compatible ranges."""

    param_name: str
    values: tuple[Any, ...]


@dataclass(frozen=True)
class AllowEntry:
    """An allow_list entry for an upstream test function.

    Attributes:
        test:            fnmatch glob matched against the test function name.
        mode:            mandatory_pass | xfail | xfail_strict.
        tags:            Free-form labels for traceability (no runtime effect).
        param_skips:     Parameter combinations to skip within this test.
        param_allows:    Parameter combinations to allow (whitelist). If specified,
                         only these parameter values will run.
        param_overrides: Parameter values to replace upstream defaults with.
    """

    test: str
    mode: str = "mandatory_pass"
    tags: tuple[str, ...] = ()
    param_skips: tuple[ParamSkip, ...] = ()
    param_allows: tuple[ParamAllow, ...] = ()
    param_overrides: tuple[ParamOverride, ...] = ()


@dataclass(frozen=True)
class BlockEntry:
    """A block_list entry — test function to skip entirely."""

    test: str


@dataclass(frozen=True)
class FileConfig:
    """Filter configuration for a single upstream test file.

    Attributes:
        rel_path:   Path relative to upstream repo root
                    (e.g. "tests/kernels/core/test_layernorm.py").
        allow_list: Tests allowed to run from this file.
        block_list: Tests blocked from running (takes precedence over allow_list).
    """

    rel_path: str
    allow_list: tuple[AllowEntry, ...] = ()
    block_list: tuple[BlockEntry, ...] = ()


@dataclass(frozen=True)
class UpstreamTestConfig:
    """Top-level upstream test filter configuration loaded from YAML."""

    files: tuple[FileConfig, ...] = ()

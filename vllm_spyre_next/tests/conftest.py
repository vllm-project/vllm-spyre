"""
Tests for vllm-spyre-next including upstream vLLM tests

This conftest.py provides automatic integration of upstream vLLM tests into the
vllm-spyre-next test suite. It clones the vLLM repository at a specific commit
and dynamically injects those tests into the pytest collection.

For usage instructions and configuration options, see:
docs/vllm_spyre_next/contributing/README.md#testing
"""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


# Global logger for pytest terminal output
_terminal_reporter = None


def _log(msg: str):
    """Log message to pytest terminal reporter if available.
    This allows logs to be printed from this file even when pytest is capturing output.
    """
    if _terminal_reporter:
        _terminal_reporter.write_line(msg)
    else:
        # Fallback to stderr when terminal reporter not available
        print(msg, file=sys.stderr)


def _cache_root() -> Path:
    """
    Cache directory for cloned tests (sticky between runs)
    """
    # Respect XDG if present, fallback to ~/.cache
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "vllm-upstream-tests"


def _extract_vllm_commit_from_pyproject() -> str | None:
    """
    Extract the vLLM git commit SHA from pyproject.toml [tool.uv.sources] section.
    Returns None if not found or parseable.
    """
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return None

    try:
        content = pyproject_path.read_text()
        # Look for vllm source with git and rev
        # Pattern: vllm = { git = "...", rev = "commit_sha" }
        match = re.search(
            r'vllm\s*=\s*\{\s*git\s*=\s*"[^"]+"\s*,\s*rev\s*=\s*"([0-9a-f]{7,40})"\s*\}', content
        )
        if match:
            return match.group(1)
    except Exception:
        pass

    return None


def _resolve_vllm_commit() -> str:
    """
    Resolve the vLLM commit SHA to use for cloning upstream tests.
    Priority: VLLM_COMMIT env var > pyproject.toml > error
    """
    # Allow env var override for testing/CI
    env_commit = os.environ.get("VLLM_COMMIT", "").strip()
    if env_commit:
        if not re.match(r"^[0-9a-f]{7,40}$", env_commit):
            raise ValueError(f"Invalid VLLM_COMMIT format: {env_commit}")
        return env_commit

    # Extract from pyproject.toml
    sha = _extract_vllm_commit_from_pyproject()
    if sha:
        return sha

    # Fail with clear instructions
    raise RuntimeError(
        "Could not resolve vLLM commit. Either:\n"
        "  1. Set VLLM_COMMIT=<sha> environment variable, or\n"
        "  2. Ensure vllm is specified with 'rev' in pyproject.toml [tool.uv.sources]"
    )


def _run(cmd: list[str], cwd: Path | None = None, max_retries: int = 3) -> None:
    """Run command with optional retries for network operations."""
    for attempt in range(max_retries):
        try:
            subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
            return
        except subprocess.CalledProcessError:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # Exponential backoff: 1s, 2s, 4s
            else:
                raise


def _ensure_repo_at_commit(repo_dir: Path, url: str, commit: str, sparse_paths: list[str]) -> Path:
    """
    Ensure repo cloned at 'repo_dir/commit' with sparse checkout of 'sparse_paths'.
    Returns the path to the working tree at that commit.
    """
    # We create a separate worktree per commit to allow co-existence of different commits
    base_dir = repo_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    git_dir = base_dir / "repo.git"

    if not git_dir.exists():
        _run(["git", "init", "--bare", str(git_dir)])

    # Prepare a worktree dir per commit
    wt_dir = base_dir / f"worktree-{commit[:12]}"
    if wt_dir.exists():
        # Already prepared; assume valid
        _log(f"[vllm-upstream] Using cached worktree at {wt_dir}")
        return wt_dir

    # Create temp dir to set up the sparse worktree then move into place atomically
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as td:
        td_path = Path(td)
        _run(["git", "--git-dir", str(git_dir), "remote", "add", "origin", url])
        _log(f"[vllm-upstream] Fetching commit {commit[:12]} from {url}")
        _run(["git", "--git-dir", str(git_dir), "fetch", "--depth=1", "origin", commit])

        # Create a new worktree at temp
        _run(
            ["git", "--git-dir", str(git_dir), "worktree", "add", "--detach", str(td_path), commit]
        )

        # Enable sparse checkout at the worktree
        _run(["git", "sparse-checkout", "init", "--cone"], cwd=td_path)
        _run(["git", "sparse-checkout", "set", *sparse_paths], cwd=td_path)

        # Ensure we're exactly at the commit (detached HEAD)
        _run(["git", "checkout", "--detach", commit], cwd=td_path)

        # Atomically move into place
        td_path.rename(wt_dir)

    return wt_dir


def _prepare_upstream_tests_dir() -> Path:
    commit = _resolve_vllm_commit()
    cache_root = _cache_root()
    wt_dir = _ensure_repo_at_commit(
        repo_dir=cache_root,
        url=os.environ.get("VLLM_REPO_URL", "https://github.com/vllm-project/vllm"),
        commit=commit,
        sparse_paths=["tests"],
    )
    tests_dir = wt_dir / "tests"
    if not tests_dir.is_dir():
        raise RuntimeError(f"Upstream tests directory not found at {tests_dir}")
    return tests_dir


# -------------------------------
# Pytest hooks
# -------------------------------


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    """
    Clone vLLM and inject upstream tests into the test session.
    This runs early in pytest initialization.

    Configure via environment variables:
    - SKIP_UPSTREAM_TESTS: Set to "1" or "true" to skip cloning and running upstream tests
    - UPSTREAM_TESTS_PATHS: Comma-separated paths (default: "models/language/generation")
    """
    global _terminal_reporter
    _terminal_reporter = config.pluginmanager.get_plugin("terminalreporter")

    # Check if upstream tests should be skipped entirely
    skip_upstream = os.environ.get("SKIP_UPSTREAM_TESTS", "").lower() in ("1", "true", "yes")
    if skip_upstream:
        _log("[vllm-upstream] SKIP_UPSTREAM_TESTS is set, skipping upstream test collection")
        config._upstream_tests_base = None
        return

    try:
        # Comma separated list of upstream paths
        DEFAULT_UPSTREAM_TESTS_PATHS = "models/language/generation"

        # Ensure VLLM_PLUGINS is set to spyre_next for all tests
        os.environ["VLLM_PLUGINS"] = "spyre_next"

        # Get list of paths to include from upstream tests
        paths_env = os.environ.get("UPSTREAM_TESTS_PATHS", DEFAULT_UPSTREAM_TESTS_PATHS).strip()
        upstream_paths = [p.strip() for p in paths_env.split(",") if p.strip()]

        if not upstream_paths:
            _log("[vllm-upstream] No upstream test paths specified, skipping")
            config._upstream_tests_base = None
            return

        upstream_tests_base = _prepare_upstream_tests_dir()

        # Add each configured path to test collection
        for rel_path in upstream_paths:
            upstream_tests_dir = upstream_tests_base / rel_path
            if not upstream_tests_dir.exists():
                _log(f"[vllm-upstream] Warning: Path not found: {upstream_tests_dir}")
                continue

            _log(f"[vllm-upstream] Including tests from: {rel_path}")
            config.args.append(str(upstream_tests_dir))

        # Store upstream test base path for use in pytest_collection_modifyitems
        config._upstream_tests_base = upstream_tests_base

    except Exception as e:
        # Fail early with a readable message
        raise SystemExit(f"[vllm-upstream] Failed to prepare upstream tests: {e}") from e


def pytest_collection_modifyitems(config, items):
    """
    Mark all upstream tests with 'upstream' marker.
    Mark subset of tests matching regex patterns with 'upstream_passing' marker.

    Can configure passing patterns via UPSTREAM_PASSING_PATTERNS env var with
    comma-separated regex patterns
    Example: "test_basic.*,test_simple_generation"
    """
    upstream_tests_base = getattr(config, "_upstream_tests_base", None)
    if not upstream_tests_base:
        return

    # Get passing test patterns from environment and compile as regex
    DEFAULT_UPSTREAM_PASSING_PATTERN = "facebook"
    patterns_env = os.environ.get(
        "UPSTREAM_PASSING_PATTERNS", DEFAULT_UPSTREAM_PASSING_PATTERN
    ).strip()
    passing_patterns = []
    if patterns_env:
        for pattern_str in patterns_env.split(","):
            pattern_str = pattern_str.strip()
            if pattern_str:
                try:
                    passing_patterns.append(re.compile(pattern_str))
                except re.error as e:
                    _log(f"[vllm-upstream] Warning: Invalid regex pattern '{pattern_str}': {e}")

    upstream_marker = pytest.mark.upstream
    passing_marker = pytest.mark.upstream_passing

    marked_count = 0
    passing_count = 0

    for item in items:
        # Check if test is from upstream directory
        test_path = Path(item.fspath)
        if test_path.is_relative_to(upstream_tests_base):
            # Mark as upstream
            item.add_marker(upstream_marker)
            marked_count += 1

            # Check if test matches any passing pattern (regex)
            test_nodeid = item.nodeid
            if passing_patterns and any(
                pattern.search(test_nodeid) for pattern in passing_patterns
            ):
                item.add_marker(passing_marker)
                passing_count += 1

            # Update node ID to include vLLM path prefix
            rel_path = test_path.relative_to(upstream_tests_base)
            vllm_prefix = f"VLLM_UPSTREAM/tests/{rel_path}"
            # Replace the file path portion of nodeid with the prefixed version
            original_nodeid = item.nodeid
            # Extract the test name part (after ::)
            if "::" in original_nodeid:
                _, test_part = original_nodeid.split("::", 1)
                item._nodeid = f"{vllm_prefix}::{test_part}"
            else:
                item._nodeid = vllm_prefix

    if marked_count > 0:
        _log(f"[vllm-upstream] Marked {marked_count} tests as 'upstream'")
        if passing_count > 0:
            _log(f"[vllm-upstream] Marked {passing_count} tests as 'upstream_passing'")


# Made with Bob

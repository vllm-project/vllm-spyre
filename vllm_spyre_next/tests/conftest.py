import os
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

# -------------------------------
# Configuration knobs (env can override)
# -------------------------------
VLLM_REPO_URL = os.environ.get("VLLM_REPO_URL", "https://github.com/vllm-project/vllm")


# Cache directory for cloned tests (sticky between runs)
def _cache_root() -> Path:
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


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


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
        return wt_dir

    # Create temp dir to set up the sparse worktree then move into place atomically
    with tempfile.TemporaryDirectory(dir=str(base_dir)) as td:
        td_path = Path(td)
        _run(["git", "--git-dir", str(git_dir), "remote", "add", "origin", url])
        # Fetch only the required commit
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
        url=VLLM_REPO_URL,
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


@pytest.fixture(autouse=True, scope="session")
def ensure_spyre_next_plugin():
    """
    Ensure VLLM_PLUGINS is set to spyre-next for all tests.
    This must run before any vLLM imports to properly load the plugin.
    """
    os.environ["VLLM_PLUGINS"] = "spyre-next"


def pytest_configure(config):
    """
    Clone vLLM and inject upstream tests into the test session.
    This runs early in pytest initialization.

    Configure via UPSTREAM_TESTS_PATHS env var (comma-separated paths).
    Default: "models/language/generation"
    """
    # Comma separated list of upstream paths
    DEFAULT_UPSTREAM_TESTS_PATHS = "models/language/generation"
    try:
        # Get list of paths to include from upstream tests
        paths_env = os.environ.get("UPSTREAM_TESTS_PATHS", DEFAULT_UPSTREAM_TESTS_PATHS).strip()
        upstream_paths = [p.strip() for p in paths_env.split(",") if p.strip()]

        if not upstream_paths:
            print("[vllm-upstream] No upstream test paths specified, skipping")
            return

        upstream_tests_base = _prepare_upstream_tests_dir()

        # Add each configured path to test collection
        for rel_path in upstream_paths:
            upstream_tests_dir = upstream_tests_base / rel_path
            if not upstream_tests_dir.exists():
                print(f"[vllm-upstream] Warning: Path not found: {upstream_tests_dir}")
                continue

            print(f"[vllm-upstream] Including tests from: {rel_path}")
            config.args.append(str(upstream_tests_dir))

    except Exception as e:
        # Fail early with a readable message
        raise SystemExit(f"[vllm-upstream] Failed to prepare upstream tests: {e}") from e

    # Store upstream test base path for use in pytest_collection_modifyitems
    config._upstream_tests_base = upstream_tests_base if upstream_paths else None


def pytest_collection_modifyitems(config, items):
    """
    Mark all upstream tests with 'upstream' marker.
    Mark subset of tests matching regex patterns with 'upstream_passing' marker.
    Collect and register all marks from upstream tests to suppress warnings.

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
                    print(f"[vllm-upstream] Warning: Invalid regex pattern '{pattern_str}': {e}")

    upstream_marker = pytest.mark.upstream
    passing_marker = pytest.mark.upstream_passing

    marked_count = 0
    passing_count = 0
    upstream_marks = set()

    for item in items:
        # Check if test is from upstream directory
        test_path = Path(item.fspath)
        try:
            test_path.relative_to(upstream_tests_base)
            is_upstream = True
        except ValueError:
            is_upstream = False

        if is_upstream:
            # Collect all marks from upstream tests
            for mark in item.iter_markers():
                upstream_marks.add(mark.name)

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

    # Register all collected upstream marks to suppress warnings
    for mark_name in upstream_marks:
        if mark_name not in ("upstream", "upstream_passing"):
            config.addinivalue_line("markers", f"{mark_name}: mark from upstream vLLM tests")

    if marked_count > 0:
        print(f"[vllm-upstream] Marked {marked_count} tests as 'upstream'")
        if passing_count > 0:
            print(f"[vllm-upstream] Marked {passing_count} tests as 'upstream_passing'")
        if upstream_marks:
            print(f"[vllm-upstream] Registered {len(upstream_marks)} upstream markers")


# Made with Bob

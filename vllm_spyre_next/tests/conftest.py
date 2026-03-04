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
            r'vllm\s*=\s*\{\s*git\s*=\s*"[^"]+"\s*,\s*rev\s*=\s*"([0-9a-f]{7,40})"\s*\}',
            content
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
        _run(["git", "--git-dir", str(git_dir), "worktree", "add", "--detach", str(td_path), commit])

        # Enable sparse checkout at the worktree
        _run(["git", "sparse-checkout", "init", "--cone"], cwd=td_path)
        _run(["git", "sparse-checkout", "set", *sparse_paths], cwd=td_path)

        # Optional: ensure we're exactly at the commit (detached HEAD)
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
    try:
        # Get list of paths to include from upstream tests
        paths_env = os.environ.get("UPSTREAM_TESTS_PATHS", "models/language/generation").strip()
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

    # If a subset is provided via env (for CI convenience), also inject -k
    subset_env = os.environ.get("UPSTREAM_TESTS_SUBSET", "").strip()
    if subset_env:
        print(f"[vllm-upstream] Applying filter: -k {subset_env}")
        config.args.extend(["-k", subset_env])

# Made with Bob

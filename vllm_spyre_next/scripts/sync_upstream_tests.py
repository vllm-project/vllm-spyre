"""
Sync upstream-tests dependency group with vLLM test dependencies.

Run this whenever the vLLM version is updated to keep test dependencies in sync.

Usage:
    python scripts/sync_upstream_tests.py <vllm_path>
"""

import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def clear_upstream_tests_section(pyproject_path: Path) -> None:
    """
    Clear the upstream-tests dependency group, leaving an empty group
    so uv doesn't error on it being referenced in default-groups.
    """
    with open(pyproject_path) as f:
        lines = f.readlines()

    result, inside, depth = [], False, 0
    for line in lines:
        if not inside and re.match(r"^upstream-tests\s*=\s*\[", line):
            inside = True
            depth = line.count("[") - line.count("]")
            result.append("upstream-tests = []\n")
            if depth <= 0:
                inside = False
            continue
        if inside:
            depth += line.count("[") - line.count("]")
            if depth <= 0:
                inside = False
            continue
        result.append(line)

    with open(pyproject_path, "w") as f:
        f.writelines(result)


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <vllm_path>", file=sys.stderr)
        return 1

    vllm_path = Path(sys.argv[1]).resolve()
    test_in = vllm_path / "requirements" / "test.in"

    if not test_in.exists():
        print(f"Error: {test_in} not found", file=sys.stderr)
        print("Expected structure: <vllm_path>/requirements/test.in", file=sys.stderr)
        return 1

    pyproject_path = PROJECT_ROOT / "pyproject.toml"
    if not pyproject_path.exists():
        print(f"Error: {pyproject_path} not found", file=sys.stderr)
        return 1

    print("Clearing existing upstream-tests section...")
    clear_upstream_tests_section(pyproject_path)

    print(f"Adding dependencies from {test_in}...")
    result = subprocess.run(
        ["uv", "add", "--group", "upstream-tests", "--no-sync", "-r", test_in],
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        print(f"Error: uv command failed with exit code {result.returncode}", file=sys.stderr)
        return 1

    print("Done.")
    print("Review changes to pyproject.toml and uv.lock before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Contributing to vLLM Spyre Next

Thank you for your interest in contributing to the Spyre plugin for vLLM! There are several ways you can contribute:

- Identify and report any issues or bugs.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

## Issues

If you encounter a bug or have a feature request, please search [existing issues](https://github.com/vllm-project/vllm-spyre/issues?q=is%3Aissue) first to see if it has already been reported. If not, please create a new issue, by using our [issue templates](https://github.com/vllm-project/vllm-spyre/issues/new/choose):

- **🐛 Bug Report**: For reporting bugs and unexpected behavior
- **🚀 Feature Request**: For suggesting new features or improvements

You can also reach out for support in the `#sig-spyre` channel in the [vLLM Slack](https://inviter.co/vllm-slack) workspace.

## Getting Started

Check out the [Installation Guide](../getting_started/installation.md) for instructions on how to set up your development environment.

## Testing

The project includes both local tests (located in `vllm_spyre_next/tests/`) for vllm-spyre-next specific functionality, and upstream vLLM tests automatically cloned from the vLLM repository at the commit specified in `pyproject.toml`, for compatibility verification.

### Test Markers

The test suite uses pytest markers to categorize tests:

```python
--8<-- "vllm_spyre_next/pyproject.toml:test-markers-definition"
```

By default, `pytest` runs tests marked `spyre` or `upstream_passing`. Some useful overrides:

```bash
# Run only local tests
pytest -m spyre

# Run only passing upstream tests
pytest -m upstream_passing

# Run all upstream tests, including non-passing
pytest -m upstream

# Run upstream tests not yet marked as passing
pytest -m "upstream and not upstream_passing"
```

### Upstream Test Integration

Upstream tests are cloned from the vLLM repository at the commit pinned in `pyproject.toml`, fetching only the `tests/` directory. Cloned tests are cached in `~/.cache/vllm-upstream-tests` (or `$XDG_CACHE_HOME/vllm-upstream-tests`) with separate worktrees per commit, allowing multiple vLLM versions to be tested simultaneously. All upstream tests run with `VLLM_PLUGINS=spyre_next` set automatically. See `tests/conftest.py` for implementation details.

!!! tip
    To force a re-clone, remove `~/.cache/vllm-upstream-tests`.

    ```bash
    rm -rf ~/.cache/vllm-upstream-tests
    ```

### Configuration

**SKIP_UPSTREAM_TESTS**: Skip upstream tests entirely. Accepts `1`, `true`, or `yes`.

**VLLM_COMMIT**: Override the vLLM commit SHA from `pyproject.toml`.

**VLLM_REPO_URL**: Override the vLLM repository URL. Defaults to `https://github.com/vllm-project/vllm`.

**UPSTREAM_TESTS_PATHS**: Comma-separated paths to include, relative to `tests/`. Defaults to `models/language/generation`.

**UPSTREAM_PASSING_PATTERNS**: Comma-separated regex patterns used to identify tests marked `upstream_passing`. Defaults to `facebook` (matches Meta/Facebook model tests).

!!! tip
    Environment variables can be passed directly to the `pytest` command, e.g. `VLLM_COMMIT=abc123def456 pytest`.

## Pull Requests

### Linting

When submitting a PR, please make sure your code passes all linting checks. We use prek with a .pre-commit-config.yaml file to run checks on every commit.

The `format.sh` script will run prek from an isolated virtual environment using [uvx](https://docs.astral.sh/uv/guides/tools/). The only requirement is that you have `uv` installed.

```sh
bash format.sh
```

Alternatively, you can [install prek](https://github.com/j178/prek?tab=readme-ov-file#installation) and set up a git hook to run it on every commit with:

```sh
prek install
```

### DCO and Signed-off-by

When contributing, you must agree to the [DCO](https://github.com/vllm-project/vllm-spyre/blob/main/DCO).Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## Additional Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [torch-spyre Documentation](https://github.com/torch-spyre/torch-spyre)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [uv Documentation](https://docs.astral.sh/uv/)

## License

See <gh-file:LICENSE>.

# Installation

This guide covers the installation of `vllm-spyre-next` using `uv`, a fast Python package installer and resolver.

## Prerequisites

- Python >= 3.11
- `uv` package manager installed ([installation guide](https://github.com/astral-sh/uv))
- Access to systems where `sendnn` is available (required for torch-spyre compilation)

## Installation with uv sync

The `vllm-spyre-next` plugin uses `uv` for dependency management and installation. The project configuration in `vllm_spyre_next/pyproject.toml` includes several important settings that ensure proper installation:

### Basic Installation

From the `vllm_spyre_next` directory, run:

```bash
uv sync --frozen
```

This command will:

1. Install all project dependencies
2. Build vLLM from source with CPU backend support
3. Build torch-spyre from source
4. Install PyTorch 2.10.0 from the CPU-specific index

### Configuration Highlights

The `pyproject.toml` file includes several key configurations:

#### 1. Build Configuration

```toml
[tool.uv]
build-constraint-dependencies = ["torch==2.10.0"]
extra-build-variables = { vllm = { VLLM_TARGET_DEVICE = "cpu" } }
```

These settings ensure:

- All packages are built with the same PyTorch version (2.10.0)
- vLLM is built specifically for the CPU backend

#### 2. Source Repositories

The plugin pulls dependencies from specific Git repositories:

```toml
[tool.uv.sources]
vllm = { git = "https://github.com/vllm-project/vllm", rev = "..." }
torch-spyre = { git = "https://github.com/torch-spyre/torch-spyre", rev = "..." }
```

This ensures that both torch-spyre and vllm are compiled from source, instead of pulling pre-compiled wheels from pypi.

#### 3. PyTorch CPU Index

```toml
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

We ensure that the cpu flavor of pytorch is installed, as we're not building cuda support.

## Verification

After installation, verify the plugin is correctly installed:

```bash
python -c "import vllm_spyre_next; print(vllm_spyre_next.__version__)"
```

## Development Installation

For development work, install with the dev dependency group:

```bash
uv sync --group dev
```

This includes additional tools like pytest, pytest-asyncio, and other testing utilities.

## Troubleshooting

### Build Failures

If you encounter build failures:

1. **torch-spyre compilation**: Ensure `sendnn` is available on your system. See internal development documentation for how to set up a dev environment with `sendnn`.
2. **vLLM build**: Check that you have sufficient memory and CPU resources for compilation
3. **Dependency conflicts**: Review the `override-dependencies` section in `pyproject.toml`

## Next Steps

After installation, you can start using vllm-spyre-next with your vLLM applications. The plugin will automatically be loaded by vLLM when the appropriate platform is detected.

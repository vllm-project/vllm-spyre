# Installation

## With Docker

First, download `vllm-spyre`:

```shell
git clone https://github.com/vllm-project/vllm-spyre.git
cd vllm-spyre
```

Build image from source:

```shell
docker build . -f Dockerfile.spyre -t vllm-spyre
docker run -it --rm vllm-spyre bash
```

## In a local environment

We use the [uv](https://docs.astral.sh/uv/) package manager to manage the
installation of the plugin and its dependencies. `uv` provides advanced
dependency resolution which is required to properly install dependencies like
`vllm` without overwriting critical dependencies like `torch`.

```shell
# Install uv
pip install uv

# Install vllm-spyre
git clone https://github.com/vllm-project/vllm-spyre.git
cd vllm-spyre
VLLM_TARGET_DEVICE=empty uv pip install -e .
```

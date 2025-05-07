# Installation

We use the [uv](https://docs.astral.sh/uv/) package manager to manage the
installation of the plugin and its dependencies. `uv` provides advanced
dependency resolution which is required to properly install dependencies like
`vllm` without overwriting critical dependencies like `torch`.

```bash
# Install uv
pip install uv

# Install vllm-spyre
git clone https://github.com/vllm-project/vllm-spyre.git
cd vllm-spyre
VLLM_TARGET_DEVICE=empty uv pip install -e .
```

# Installation

We use the [uv](https://docs.astral.sh/uv/) package manager to manage the
installation of the plugin and its dependencies. `uv` provides advanced
dependency resolution which is required to properly install dependencies like
`vllm` without overwriting critical dependencies like `torch`.

1. Clone vllm-spyre

   ```sh
   git clone https://github.com/vllm-project/vllm-spyre.git
   cd vllm-spyre
   ```

1. Install uv
  
   ```sh
   pip install uv
   ```
  
1. Create a new env

   ```sh
   uv venv --python 3.12 --seed .venv
   ```

1. Activate it
  
   ```sh
   source .venv/bin/activate
   ```

1. Install `vllm-spyre` locally with dev (and optionally lint) dependencies
  
   ```sh
   uv sync --frozen --active --inexact
   ```
  
   or also with lint:
  
   ```sh
   uv sync --frozen --active --inexact --group lint
   ```

!!! tip
    The `dev` group (i.e. `--group dev`) is enabled by default.

1. (Optional) Install torch through pip
  
   If you don't have it installed already. Will be needed
   for running examples or tests.
  
   ```sh
   pip install torch==2.7.0
   ```

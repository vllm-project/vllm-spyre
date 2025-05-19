# Local Development

This document describes how to install and configure vllm-spyre for local
development on a CPU without a spyre card. It has been tested on `arm64` (M1 chip) and `x86_64` machines.

## Installation

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
   uv sync --frozen --active
   ```
  
   or also with lint:
  
   ```sh
   uv sync --frozen --group lint
   ```

   :::{tip}
   `--group dev` is enabled by default
   :::

1. (Optional) Install torch through pip
  
   This is needed to run examples or tests.
   Also version doesn't matter at the moment.
  
   ```sh
   pip install torch==2.7.0
   ```
  
   :::{note}
   There might be some version resolution errors.
   Ignore them for now.
   :::

# Local Development

This document describes how to install and configure vllm-spyre for local
development without a spyre card. It has been tested on `arm64` (M1 chip)
and `x86_64` machines.

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

   :::{note}
   `--group dev` is enabled by default
   :::

2. (Optional) Install torch through pip
  
   This is needed to run examples or tests.
   We can't use uv since pyproject.toml prevents it.
   Also version doesn't matter at the moment.
  
   ```sh
   pip install torch==2.7.0
   ```

   Note: There might be some version resolution errors.
   Ignore them for now.

### Run tests
  
1. (arm64 only) Install xgrammar
   (This is needed for testing v1 stuff for local testing on arm64 machines.
   It's installed for x86_64 automatically.
   Also version doesn't matter at the moment.)

   ```sh
   uv pip install xgrammar==0.1.19
   ```

2. (optional)  Download `JackFram/llama-160m` model for tests

   ```sh
   python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')"
   ```

   Note: Downloading the same model using HF API does not work locally.

   We assume the model lands here:

   ```sh
   .cache/huggingface/hub/models--JackFram--llama-160m
   ```

3. Source env variables needed for tests

   ```sh
   source _local_env.sh
   ```

4. (optional) Install dev dependencies (if spyre was installed without uv)
  
   ```sh
   uv pip install --group dev
   ```

5. Run the tests:
  
   ```sh
   python -m pytest -v -x tests -m "v1 and cpu and e2e"
   ```

### Run examples

Note: Make sure `model name` aligns with the model that you downloaded

```sh
HF_HUB_OFFLINE=1 python examples/offline_inference_spyre.py
```

Note: We use `HF_HUB_OFFLINE=1` otherwise vllm tries to download a
different version of the model using HF API which might not work locally.

## Continuous Batching(CB) custom installation

(Temporary section until FMS custom branch is merged to main)

Do this after following all steps for installation and testing above.

1. Install custom FMS branch for CB:

   ```sh
   uv pip install git+https://github.com/foundation-model-stack/foundation-model-stack.git@paged_attn_mock --force-reinstall
   ```

### Run only CB tests

```sh
HF_HUB_OFFLINE=1 python -m pytest -v -x tests/e2e -m cb
```

### Run CB example

Note: You will have to change the `model-name` in the example file before running it.

```sh
HF_HUB_OFFLINE=1 python examples/offline_inference_spyre_cb.py
```

## Debugging using debugpy

`launch.json` content:

```sh
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python Debugger: local",
      "type": "debugpy",
      "request": "attach",
      "connect": {
        "host": "localhost",
        "port": 5678
      },
      "justMyCode": false
    }
  ]
}

```

Run using

```sh
python -m debugpy --listen 5678  -m pytest ...
```

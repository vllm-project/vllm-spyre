# Local development

You can run the examples and tests locally on a linux or an M1 mac.

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
   uv venv --python 3.12 --seed
   ```

1. Activate it
  
   ```sh
   source .venv/bin/activate
   ```

1. Install vvlm-spyre locally with dev (and optionally lint) dependencies
  
   (`--group dev` is enabled by default)

  ```sh
   uv sync --frozen
   ```
  
   or also with lint:
  
   ```sh
   uv sync --frozen --group lint
   ```

1. Install torch through pip
  
   (We can't use uv since pyproject.toml prevents it.
   Also version doesn't matter at the moment.)
  
   ```sh
   pip install torch==2.7.0
   ```

   Note: There might be some version resolution errors.
   Ignore them for now.

### Run tests
  
1. Install xgrammar
   (This is needed for testing v1 stuff.
   Also version doesn't matter at the moment.)

   ```sh
   uv pip install xgrammar==0.1.19
   ```

1. Download test model (optional)

   ```sh
   python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')"
   ```

   Note: Downloading the same model using HF API does not work locally.

   Assuming the model lands here:

   ```sh
   .cache/huggingface/hub/models--JackFram--llama-160m
   ```

1. Source env variables needed for tests

   ```sh
   source _local_env.sh
   ```

1. (optional) Install dev dependencies (if spyre was installed in editable mode)
  
   ```sh
   uv pip install --group dev
   ```

1. Run the tests:
  
   ```sh
   python -m pytest -v -x tests -m "v1 and cpu and e2e"
   ```

### Run examples

Note: You will have to change the `model-name` in the example files before
running it.

```sh
python examples/offline_inference_spyre.py
```

## Continuous Batching(CB) custom installation

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

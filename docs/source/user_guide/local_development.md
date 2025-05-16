# Running tests/debug locally on an M1

1. Create a new env

   ```sh
   uv venv --python 3.12 --seed
   ```
  
   ```sh
   source .venv/bin/activate
   ```

1. Install dev (and optionally lint dependencies)
  
    ```sh
   uv sync --frozen --group dev
   ```
  
or with lint:
  
```sh
   uv sync --frozen --group dev --group lint
   ```

1. Sourcing variables

   ```sh
   source _local_env.sh
   ```

1. Install torch through pip
  
   (We can't use uv since pyproject.toml prevents it.
   Also version doesn't matter at the moment.)
  
   ```sh
   pip install torch==2.7.0
   ```

1. Install xgrammar
   (This is needed for testing v1 stuff.
   Also version doesn't matter at the moment.)

   ```sh
   uv pip install xgrammar==0.1.19
   ```

1. Download model

   ```sh
   python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')"
   ```

   Assuming model lands here:

   ```sh
   .cache/huggingface/hub/models--JackFram--llama-160m
   ```

1. Run tests
  
   ```sh
   python -m pytest -v -x tests -m "v1 and cpu and e2e"
    ```

## Continuous Batching(CB) custom installation

1. Install custom FMS branch for CB:

   `uv pip install git+https://github.com/foundation-model-stack/foundation-model-stack.git@paged_attn_mock --force-reinstall`

1. Run only CB tests:

   ```sh
   python -m pytest -v -x tests/e2e -m cb
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

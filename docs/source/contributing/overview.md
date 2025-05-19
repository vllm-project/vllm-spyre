# Contributing

Thank you for your interest in contributing to Spyre support on vLLM!

When submitting a PR, please make sure your code passes all linting checks:

## Linting

You can install the linting requirements using either `uv` or `pip`.

Using `uv`:

```bash
uv sync --frozen --group lint --active --inexact
```

Using `pip`:

```bash
uv pip compile --group lint > requirements-lint.txt
pip install -r requirements-lint.txt
```

After installing the requirements, run the formatting script:

```bash
bash format.sh
```

Then, make sure to commit any changes made by the formatter:

```bash
git add .
git commit -s -m "Apply linting and formatting"
```

Now, you’re good to go! 🚀

## DCO and Signed-off-by

When contributing changes to this project, you must agree to the [DCO](https://github.com/vllm-project/vllm/blob/main/DCO).
Commits must include a `Signed-off-by:` header which certifies agreement with
the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## Testing

### Running tests locally on CPU (No Spyre card)
  
1. (arm64 only) Install xgrammar
  
   :::{tip}
   It's installed for x86_64 automatically.
   :::

   ```sh
   uv pip install xgrammar==0.1.19
   ```

1. (optional) Download `JackFram/llama-160m` model for tests

   ```sh
   python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')"
   ```

   :::{caution}
   Downloading the same model using HF API does not work locally on `arm64`.
   :::

   :::{tip}
   :class: dropdown
   We assume the model lands here:

   ```sh
   .cache/huggingface/hub/models--JackFram--llama-160m
   ```
  
    :::

1. Source env variables needed for tests

   ```sh
   source _local_env.sh
   ```

1. (optional) Install dev dependencies (if vllm-spyre was installed without uv)
  
   ```sh
   uv pip install --group dev
   ```

1. Run the tests:
  
   ```sh
   python -m pytest -v -x tests -m "v1 and cpu and e2e"
   ```

### Continuous Batching (CB)

:::{attention}
Temporary section until FMS custom branch is merged to main
:::

Continuous batching requires a custom installation at the moment until the FMS custom branch is merged to main.

To try it out, after following all steps for setting up testing as mentioned above,

1. Install custom FMS branch for CB:

   ```sh
   uv pip install git+https://github.com/foundation-model-stack/foundation-model-stack.git@paged_attn_mock --force-reinstall
   ```

#### Run only CB tests

```sh
python -m pytest -v -x tests/e2e -m cb
```

## Debugging

We can debug using `debugpy` in VS code.

 This is the content of the `launch.json` file which we need for debugging in VS code:

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

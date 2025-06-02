# Contributing

Thank you for your interest in contributing to Spyre support on vLLM!

There are several ways you can contribute:

- Identify and report any issues or bugs.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

## Issues

If you encounter a bug or have a feature request, please search [existing issues](https://github.com/vllm-project/vllm-spyre/issues?q=is%3Aissue) first to see if it has already been reported. If not, please [create a new issue](https://github.com/vllm-project/vllm-spyre/issues/new/choose), providing as much relevant information as possible.

You can also reach out for support in the `#sig-spyre` channel in the [vLLM Slack](https://inviter.co/vllm-slack) workspace.

## Testing

### Testing Locally on CPU (No Spyre card)
  
1. Install `xgrammar` (only for `arm64` systems):
  
   :::{tip}
   `xgrammar` is automatically installed on `x86_64` systems.
   :::

   ```sh
   uv pip install xgrammar==0.1.19
   ```

1. (Optional) Download the `JackFram/llama-160m` model:

   ```sh
   python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')"
   ```

   :::{caution}
   The Hugging Face API download does **not** work on `arm64`.
   :::

   By default, the model is saved to:

   ```sh
   .cache/huggingface/hub/models--JackFram--llama-160m
   ```

1. Source environment variables:

   ```sh
   source _local_envs_for_test.sh
   ```

1. (Optional) Install development dependencies:
  
   ```sh
   uv pip install --group dev
   ```

1. Run the tests:
  
   ```sh
   python -m pytest -v -x tests -m "v1 and cpu and e2e"
   ```

   Here are a list of `pytest` markers you can use to filter tests:

   :::{literalinclude} ../../../pyproject.toml
   :start-after: begin-test-markers-definition
   :end-before: end-test-markers-definition
   :language: python
   :::

### Testing Continuous Batching

:::{attention}
Continuous batching currently requires the custom installation described below until the FMS custom branch is merged to main.
:::

After completing the setup steps above:

1. Install custom FMS branch to enable support for continuous batching:

   ```sh
   uv pip install git+https://github.com/foundation-model-stack/foundation-model-stack.git@paged_attn_mock --force-reinstall
   ```

2. Run the continuous batching tests:

   ```sh
   python -m pytest -v -x tests/e2e -m cb
   ```

## Pull Requests

### Linting

When submitting a PR, please make sure your code passes all linting checks. You can install the linting requirements using either `uv` or `pip`.

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

### DCO and Signed-off-by

When contributing, you must agree to the [DCO](https://github.com/vllm-project/vllm-spyre/blob/main/DCO).Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## License

See <gh-file:LICENSE>.

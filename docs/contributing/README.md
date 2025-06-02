# Contributing to vLLM Spyre

Thank you for your interest in contributing to the Spyre plugin for vLLM! There are several ways you can contribute:

- Identify and report any issues or bugs.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

## Issues

If you encounter a bug or have a feature request, please search [existing issues](https://github.com/vllm-project/vllm-spyre/issues?q=is%3Aissue) first to see if it has already been reported. If not, please [create a new issue](https://github.com/vllm-project/vllm-spyre/issues/new/choose), providing as much relevant information as possible.

You can also reach out for support in the `#sig-spyre` channel in the [vLLM Slack](https://inviter.co/vllm-slack) workspace.

## Developing

### Building the docs with MkDocs

#### Install MkDocs and Plugins

Install MkDocs along with the [plugins](https://github.com/vllm-project/vllm-spyre/blob/main/mkdocs.yaml) used in the vLLM Spyre documentation.

```bash
pip install -r docs/requirements-docs.txt
```

!!! note
    Ensure that your Python version is compatible with the plugins (e.g., `mkdocs-awesome-nav` requires Python 3.10+)

#### Start the Development Server

MkDocs comes with a built-in dev-server that lets you preview your documentation as you work on it.

Make sure you're in the same directory as the `mkdocs.yml` configuration file in the `vllm-spyre` repository, and then start the server by running the `mkdocs serve` command:

```bash
mkdocs serve
```

Example output:

```console
INFO    -  Documentation built in 106.83 seconds
INFO    -  [22:02:02] Watching paths for changes: 'docs', 'mkdocs.yaml'
INFO    -  [22:02:02] Serving on http://127.0.0.1:8000/
```

#### View in Your Browser

Open up [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser to see a live preview:.

#### Learn More

For additional features and advanced configurations, refer to the official [MkDocs Documentation](https://www.mkdocs.org/).

## Testing

### Testing Locally on CPU (No Spyre card)

!!! tip
    `xgrammar` is automatically installed on `x86_64` systems.

Install `xgrammar` (only for `arm64` systems):

```sh
uv pip install xgrammar==0.1.19
``` 

Optionally, download the `JackFram/llama-160m` model:

```sh
python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')"
```

!!! caution
    The Hugging Face API download does **not** work on `arm64`.

By default, the model is saved to `.cache/huggingface/hub/models--JackFram--llama-160m`.

Then, source the environment variables:

```sh
source _local_envs_for_test.sh
```

Optionally, install development dependencies:

```sh
uv pip install --group dev
```

Now, you can run the tests:
  
```sh
python -m pytest -v -x tests -m "v1 and cpu and e2e"
```

Here is a list of `pytest` markers you can use to filter them:

```python
--8<-- "pyproject.toml:test-markers-definition"
```

### Testing Continuous Batching

!!! attention
    Continuous batching currently requires the custom installation described below until the FMS custom branch is merged to main.

After completing the setup steps above, install custom FMS branch to enable support for continuous batching:

```sh
uv pip install git+https://github.com/foundation-model-stack/foundation-model-stack.git@paged_attn_mock --force-reinstall
```

Then, run the continuous batching tests:

```sh
python -m pytest -v -x tests/e2e -m cb
```

## Pull Requests

### Linting

When submitting a PR, please make sure your code passes all linting checks. You can install the linting requirements using either `uv` or `pip`.

Using `uv`:

```sh
uv sync --frozen --group lint --active --inexact
```

Using `pip`:

```sh
uv pip compile --group lint > requirements-lint.txt
pip install -r requirements-lint.txt
```

After installing the requirements, run the formatting script:

```sh
bash format.sh
```

Then, make sure to commit any changes made by the formatter:

```sh
git add .
git commit -s -m "Apply linting and formatting"
```

### DCO and Signed-off-by

When contributing, you must agree to the [DCO](https://github.com/vllm-project/vllm-spyre/blob/main/DCO).Commits must include a `Signed-off-by:` header which certifies agreement with the terms of the DCO.

Using `-s` with `git commit` will automatically add this header.

## License

See <gh-file:LICENSE>.

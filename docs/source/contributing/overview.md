# Contributing

Thank you for your interest in contributing to Spyre support on vLLM!

When submitting a PR, please make sure your code passes all linting checks:

## Linting

You can install the linting requirements using either `uv` or `pip`.

Using `uv`:

```bash
uv sync --frozen --group lint
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

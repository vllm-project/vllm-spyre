# Contributing to vLLM-Spyre

Thank you for contributing to vLLM support on Spyre!

Make sure your code passes all linting checks — otherwise, your pull request won't be merged.

## 1. Install Linting Requirements

You can install the linting requirements using either `uv` or `pip`.

### Using `uv`

```bash
uv sync --frozen --group lint
```

### Using `pip`

```bash
uv pip compile --group lint > requirements-lint.txt
pip install -r requirements-lint.txt
```

## 2. Run the Formatter

After installing the requirements, run the formatting script:

```bash
bash format.sh
```

## 3. Commit the Changes

Make sure to commit any changes made by the formatter:

```bash
git add .
git commit -m "Apply linting and formatting"
```

Now, you’re good to go! 🚀

# Installation

We use the [uv](https://docs.astral.sh/uv/) package manager to manage the
installation of the plugin and its dependencies. `uv` provides advanced
dependency resolution which is required to properly install dependencies like
`vllm` without overwriting critical dependencies like `torch`.

First, clone the `vllm-spyre` repo:

```sh
git clone https://github.com/vllm-project/vllm-spyre.git
cd vllm-spyre
```

Then, install `uv`:
  
```sh
pip install uv
```

Now, create and activate a new [venv](https://docs.astral.sh/uv/pip/environments/):
  
```sh
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
```

To install `vllm-spyre` locally with development dependencies, use the following command:

```sh
uv sync --frozen --active --inexact
```

To include optional linting dependencies, include `--group lint`:

```sh
uv sync --frozen --active --inexact --group lint
```

!!! tip
    The `dev` group (i.e. `--group dev`) is enabled by default.

Finally, the `torch` is needed to run examples and tests. If it is not already installed, install it using `pip`:

```sh
pip install torch==2.7.0
```

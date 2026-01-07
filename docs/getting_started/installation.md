# Installation

We use the [uv](https://docs.astral.sh/uv/) package manager to manage the
installation of the plugin and its dependencies. `uv` provides advanced
dependency resolution which is required to properly install dependencies like
`vllm` without overwriting critical dependencies like `torch`.

## Install `uv`

You can [install `uv`](https://docs.astral.sh/uv/guides/install-python/) using `pip`:

```sh
pip install uv
```

## Create a Python Virtual Environment

Now create and activate a new Python (3.12) [virtual environment](https://docs.astral.sh/uv/pip/environments/):

```sh
uv venv --python 3.12 --seed .venv --system-site-packages
source .venv/bin/activate
```

??? question "Why do I want the `--system-site-packages`?"
    Because the full `torch_sendnn` stack is only available pre-installed in a
    base environment, we need to add the `--system-site-packages` to the new
    virtual environment in order to fully support the Spyre hardware.

    **Note**, pulling in the system site packages is not required for CPU-only
    installations.

## Install vLLM with the vLLM-Spyre Plugin

You can either install a released version of the vLLM-Spyre plugin directly from
[PyPI](https://pypi.org/project/vllm-spyre/) or you can install from source by
cloning the [vLLM-Spyre](https://github.com/vllm-project/vllm-spyre) repo from
GitHub.

=== "Release (PyPI)"

    ```sh
    echo "torch; sys_platform == 'never'
    torchaudio; sys_platform == 'never'
    torchvision; sys_platform == 'never'
    triton; sys_platform == 'never'" > overrides.txt
    
    uv pip install vllm-spyre --overrides overrides.txt
    ```

    ??? question "Why do I need the `--overrides`?"
        To avoid dependency resolution errors, we need to install `torch`
        separately and tell `uv` to ignore any of it's dependencies while
        installing the `vllm-spyre` plugin.

=== "Source (GitHub)"

    First, clone the `vllm-spyre` repo:
    
    ```sh
    git clone https://github.com/vllm-project/vllm-spyre.git
    cd vllm-spyre
    ```
    
    To install `vllm-spyre` locally with development dependencies, use the following command:
    
    ```sh
    uv sync --frozen --active --inexact
    ```
    
    !!! tip
        The `dev` group (i.e. `--group dev`) is enabled by default.

## Install PyTorch

Finally, `torch` is needed to run examples and tests. If it is not already installed,
install it using `pip`.

The Spyre runtime stack supports specific `torch` versions. Use the compatible version for each `torch_sendnn` release:

| torch_sendnn | torch |
| -- | -- |
| 1.0.0 | 2.7.1 |

=== "Linux"

    ```sh
    pip install torch=="2.7.1+cpu" --index-url "https://download.pytorch.org/whl/cpu"
    ```

=== "Windows/macOS"

    ```sh
    pip install torch=="2.7.1"
    ```

!!! note
    On Linux the `+cpu` package should be installed, since we don't need any of
    the `cuda` dependencies which are included by default for Linux installs.
    This requires `--index-url https://download.pytorch.org/whl/cpu` on Linux.
    On Windows and macOS the CPU package is the default one.

## Troubleshooting

As the installation process is evolving over time, you may have arrived here after
following outdated installation steps. If you encountered any of the errors below,
it may be easiest to start over with a new Python virtual environment (`.venv`)
as outlined above.

### Installation using `pip` (instead of `uv`)

If you happen to have followed the pre-`uv` installation instructions, you might
encounter an error like this:

```sh
LookupError: setuptools-scm was unable to detect version for /home/senuser/multi-aiu-dev/_dev/sentient-ci-cd/_dev/sen_latest/vllm-spyre.
      
    Make sure you're either building from a fully intact git repository or PyPI tarballs. Most other sources (such as GitHub's tarballs, a git checkout without the .git folder) don't contain the necessary metadata and will not work.
      
    For example, if you're using pip, instead of https://github.com/user/proj/archive/master.zip use git+https://github.com/user/proj.git#egg=proj
```

Make sure the follow the latest installation steps outlined above.

### Failed to activate the Virtual Environment

If you encounter any of the following errors, it's likely you forgot to activate
the (correct) Python Virtual Environment:

```sh
  File "/home/senuser/.local/lib/python3.12/site-packages/vllm/config.py", line 2260, in __post_init__
    self.device = torch.device(self.device_type)
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Device string must not be empty
```

### No module named `torch`

You may have installed PyTorch into the system-wide Python environment, not into
the virtual environment used for vLLM-Spyre:

```sh
  File "/home/senuser/multi-aiu-dev/_dev/sentient-ci-cd/_dev/sen_latest/vllm-spyre/.venv/lib64/python3.12/site-packages/vllm/env_override.py", line 4, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

Make sure to activate the same virtual environment for installing `torch` that
was used to install `vllm-spyre`. If you already have a system-wide `torch`
installation and want to reuse that for your `vllm-spyre` environment, you can
create a new virtual environment and add the `--system-site-packages` flag to
pull in the `torch` dependencies from the base Python environment:

```sh
rm -rf .venv
uv venv --python 3.12 --seed .venv --system-site-packages
source .venv/bin/activate
```

If you forget to override the `torch` dependencies when installing a released
version from PyPI, you will likely see a dependency resolution error like this:

```sh
$ uv pip install vllm-spyre

Using Python 3.12.11 environment at: .venv3
Resolved 155 packages in 45ms
  × Failed to build `xformers==0.0.28.post1`
  ├─▶ The build backend returned an error
  ╰─▶ Call to `setuptools.build_meta:__legacy__.build_wheel` failed (exit status: 1)

      [stderr]
      Traceback (most recent call last):
        File "<string>", line 14, in <module>
        File "~.cache/uv/builds-v0/.tmpo0aEXS/lib/python3.12/site-packages/setuptools/build_meta.py", line 331, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=[])
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        File "~.cache/uv/builds-v0/.tmpo0aEXS/lib/python3.12/site-packages/setuptools/build_meta.py", line 301, in _get_build_requires
          self.run_setup()
        File "~.cache/uv/builds-v0/.tmpo0aEXS/lib/python3.12/site-packages/setuptools/build_meta.py", line 512, in run_setup
          super().run_setup(setup_script=setup_script)
        File "~.cache/uv/builds-v0/.tmpo0aEXS/lib/python3.12/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 24, in <module>
      ModuleNotFoundError: No module named 'torch'

      hint: This error likely indicates that `xformers@0.0.28.post1` depends on `torch`, but doesn't declare it as a build dependency. If `xformers` is a first-party package, consider adding
      `torch` to its `build-system.requires`. Otherwise, `uv pip install torch` into the environment and re-run with `--no-build-isolation`.
  help: `xformers` (v0.0.28.post1) was included because `vllm-spyre` (v0.1.0) depends on `vllm` (v0.2.5) which depends on `xformers`
```

<!-- markdownlint-disable MD051 link-fragments -->

To avoid this error, make sure to include the dependency `--overrides` as described
in the installation from a [Release (PyPI)](#release-pypi) section.

<!-- markdownlint-enable MD051 -->

### No solution found when resolving dependencies

If you forget to override the `torch` dependencies when installing from PyPI you
will likely see a dependency resolution error like this:

```sh
$ uv pip install vllm-spyre==0.4.1
  ...
  × No solution found when resolving dependencies:
  ╰─▶ Because fms-model-optimizer==0.2.0 depends on torch>=2.1,<2.5 and only the following versions of fms-model-optimizer are available:
          fms-model-optimizer<=0.2.0
          fms-model-optimizer==0.3.0
      we can conclude that fms-model-optimizer<0.3.0 depends on torch>=2.1,<2.5.
      And because fms-model-optimizer==0.3.0 depends on torch>=2.2.0,<2.6 and all of:
          vllm>=0.9.0,<=0.9.0.1
          vllm>=0.9.2
      depend on torch==2.7.0, we can conclude that all versions of fms-model-optimizer and all of:
          vllm>=0.9.0,<=0.9.0.1
          vllm>=0.9.2
       are incompatible.
      And because only the following versions of vllm are available:
          vllm<=0.9.0
          vllm==0.9.0.1
          vllm==0.9.1
          vllm==0.9.2
      and vllm-spyre==0.4.1 depends on fms-model-optimizer, we can conclude that all of:
          vllm>=0.9.0,<0.9.1
          vllm>0.9.1
       and vllm-spyre==0.4.1 are incompatible.
      And because vllm-spyre==0.4.1 depends on one of:
          vllm>=0.9.0,<0.9.1
          vllm>0.9.1
      and you require vllm-spyre==0.4.1, we can conclude that your requirements are unsatisfiable.
```

<!-- markdownlint-disable MD051 link-fragments -->

To avoid this error, make sure to include the dependency `--overrides` as described
in the installation from a [Release (PyPI)](#release-pypi) section.

<!-- markdownlint-enable MD051 -->

# Contributing to vLLM Spyre

Thank you for your interest in contributing to the Spyre plugin for vLLM! There are several ways you can contribute:

- Identify and report any issues or bugs.
- Suggest or implement new features.
- Improve documentation or contribute a how-to guide.

## Issues

If you encounter a bug or have a feature request, please search [existing issues](https://github.com/vllm-project/vllm-spyre/issues?q=is%3Aissue) first to see if it has already been reported. If not, please [create a new issue](https://github.com/vllm-project/vllm-spyre/issues/new/choose), providing as much relevant information as possible.

You can also reach out for support in the `#sig-spyre` channel in the [vLLM Slack](https://inviter.co/vllm-slack) workspace.

## Docs

### Building the docs with MkDocs

#### Install MkDocs and Plugins

Install MkDocs along with the [plugins](https://github.com/vllm-project/vllm-spyre/blob/main/mkdocs.yaml) used in the vLLM Spyre documentation.

```bash
uv pip install -r docs/requirements-docs.txt
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

Optionally, download the `ibm-ai-platform/micro-g3.3-8b-instruct-1b` model:

```sh
python -c "from transformers import pipeline; pipeline('text-generation', model='ibm-ai-platform/micro-g3.3-8b-instruct-1b')"
```

!!! caution
    The Hugging Face API download does **not** work on `arm64`.

By default, the model is saved to `.cache/huggingface/hub/models--ibm-ai-platform--micro-g3.3-8b-instruct-1b`.

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
python -m pytest -v -x tests -m "cpu and e2e"
```

Here is a list of `pytest` markers you can use to filter them:

```python
--8<-- "pyproject.toml:test-markers-definition"
```

### Testing Continuous Batching

Run the continuous batching tests:

```sh
python -m pytest -v -x tests/e2e -m cb
```

## Debugging

!!! tip
    You can `oc edit` a pod and change the image without having the pod schedule to a different node. This can be useful for testing whether software or hardware is the issue.

- The script `/opt/sentient/bin/aiu-query-devices` in the pod can be used to see the connectivity between the `AIUs` on the machine. You can also infer this from environment variables with names like `AIU_TIER_\d_SET_\d_RANK_\d`.
  
- `SPYRE_DEVICES` can be used to select which devices will be selected for each `RANK`. This is similar to how `CUDA_VISIBLE_DEVICES` works for GPU.
  
    !!! example
        `0,2,4,6` will assign rank `0` to AIU index `0`, rank `1` to AIU index `2`, rank `2` to AIU index `4`, and rank `3` to AIU index `6`.
  
    - An alternative is to use `AIU_WORLD_RANK_\d=0000:aa:00.0` to explicitly map ranks to `PCI` addresses (make sure there are no duplicates used at runtime).
  
- A bash script that uses `/opt/sentient/senlib/bin/senlib_unit_test` to check each `AIU` allocated to the pod to see if they work for a basic test:
  
    ```shell
    --8<-- "tools/check_aiu.sh"
    ```

### Logging levels

Various log levels that can be configured:

- `DTLOG_LEVEL` - `TRACE, DEBUG, INFO, WARNING, ERROR`
- `TORCH_SENDNN_LOG` - `WARNING, CRITICAL`
- `VLLM_LOGGING_LEVEL` - `DEBUG, INFO, WARNING, ERROR`

!!! tip
    `DTLOG_LEVEL=INFO` (piped to file) can help you see what device addresses are actually in use. Look for the string `Opened: SEN:VFIO`.

!!! tip
    In order to stop massive log spew, this configuration is ideal:
    ```
    export DTLOG_LEVEL=ERROR
    export TORCH_SENDNN_LOG=CRITICAL
    ```

### Topology Aware Allocation

This section is specific to the AIU operator and scheduling workloads onto specific cards.

(TODO: link to docs once they exist)

- This mode supports users to request a special set of AIU cards based on `PCI` topology. By using this mode, we can guarantee to pick up AIU cards of a particular class in the node:
  
    - `Tier0` provides a set of cards in the same `PCI` switch.
    - `Tier1` provides a set of cards from at most one-hop away `PCI` switch.
    - `Tier2` provides a set of cards from at most two-hops away `PCI` switch.

- Running a Multi AIU Job using `ibm.com/aiu_pf_tier0,tier1,tier2`:
  
    - This resource type is used for picking up a topology aware card set, which is required to run tensor parallel (`TP`) workloads more effectively. By using `tierX` class resource, `TP` users can automatically get a best performing card set for the workload.

- The maximum number of allocatable resources in each tier depends on the platform & cluster, but we can get up to:
  
    - `Tier0` - `4` cards
    - `Tier1` - `8` cards
    - `Tier2` - `16` cards

- Devices in `tier0` can do `peer-to-peer (P2P) RDMA`, devices on different trees use `Host DMA` sharing files through `/dev/shm`.

    !!! warning
         If you request cards greater than the cards supported by the switch, the pod will never be scheduled. In the above example, if you specify `ibm.com/aiu_pf_tier0: 5` in your yaml, the pod will never be scheduled because the maximum set of cards in `tier0` was specified as `4`.

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

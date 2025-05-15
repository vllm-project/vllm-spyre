# Mac

## Running tests/debug locally on an M1

1. Create a new env
    ```
    uv venv --python 3.12 --seed

    source .venv/bin/activate
    ```
1. Install dev (and optionally lint dependencies) 
    ```
    uv sync --frozen --group dev
    or
    uv sync --frozen --group dev --group lint
    ```
1. Sourcing variables

    ```
    source _local_env.sh
    ```
    
1. Install torch through pip 
    <!-- (can't use uv since pyproject.toml prevents it) -->
    <!-- version doesn't matter atm -->
    ```
    pip install torch==2.7.0
    ```

1. Install xgrammar
    <!-- needed for v1 stuff -->
    <!-- version doesn't matter atm -->
    ```
    uv pip install xgrammar==0.1.19
    ```

1. Install model
    
    ```
    python -c "from transformers import pipeline; pipeline('text-generation', model='JackFram/llama-160m')"
    ```
    Assuming model lands here:
    ```
    .cache/huggingface/hub/models--JackFram--llama-160m
    ```

2. Run tests
    ```
    python -m pytest -v -x tests -m "v1 and cpu and e2e"
    ```

**Continuous Batching(CB) custom installation**

1. Install custom FMS branch for CB:
   
   ```uv pip install git+https://github.com/foundation-model-stack/foundation-model-stack.git@paged_attn_mock --force-reinstall```

1. Run only CB tests: 

    ```
    python -m pytest -v -x tests/e2e -m cb
    ```



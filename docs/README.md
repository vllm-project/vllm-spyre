# vLLM Spyre Plugin docs

Live docs: [vllm-spyre.readthedocs.io](https://vllm-spyre.readthedocs.io)

## Build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## Open the docs with your browser

```bash
python -m http.server -d _build/html/
```

Launch your browser and open [localhost:8000](http://localhost:8000/).

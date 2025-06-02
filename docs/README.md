# vLLM Spyre Plugin docs

Live doc: [vllm-spyre.readthedocs.io](https://vllm-spyre.readthedocs.io)

## How to build the docs

```bash
# Install dependencies.
pip install -r requirements-docs.txt

# Build the docs.
make clean
make html
```

## How to view the docs in a web browser

```bash
python -m http.server -d _build/html/
```

Launch your browser and open [localhost:8000](http://localhost:8000/).

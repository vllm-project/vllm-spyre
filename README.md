# Spyre plugin for vLLM

The vLLM Spyre plugin (`vllm-spyre`) is a dedicated backend extension that enables seamless integration of IBM Spyre Accelerator with vLLM. It follows the architecture describes in [vLLM's Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system.html), making it easy to integrate IBM's advanced AI acceleration into existing vLLM workflows.

## Installation

### With Docker

First, download vllm-spyre

```
git clone https://github.com/vllm-project/vllm-spyre.git
cd vllm-spyre
```

Build image from source

```
docker build . -f Dockerfile.spyre -t vllm-spyre
docker run -it --rm vllm-spyre bash
```

### In a local environment

```
# Install vllm
pip install vllm==0.8.0

# Install vllm-spyre
cd ..
git clone https://github.com/vllm-project/vllm-spyre.git
cd vllm-spyre
pip install -v -e .
```

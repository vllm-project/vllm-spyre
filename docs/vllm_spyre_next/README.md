# vLLM-Spyre-Next

## Overview

The `vllm-spyre-next` plugin represents the next iteration of vLLM-Spyre, built on the `torch-spyre` stack. This plugin integrates with `torch-spyre` to leverage the PyTorch Inductor backend for model compilation, providing enhanced performance and optimization capabilities for large language model inference.

## Key Features

- **Torch Inductor Backend**: Utilizes PyTorch's native Inductor compiler for optimized model execution
- **torch-spyre Integration**: Built on the torch-spyre framework for advanced compilation and optimization
- **vLLM Platform Plugin**: Seamlessly integrates with vLLM's plugin architecture via the `vllm.platform_plugins` entry point
- **CPU-Optimized**: Configured for efficient CPU-based inference with vLLM 0.15.1+cpu

## Architecture

The plugin registers itself as a vLLM platform plugin through the entry point:

```python
[project.entry-points."vllm.platform_plugins"]
spyre_next = "vllm_spyre_next:register"
```

This allows vLLM to automatically discover and load the plugin, enabling torch-spyre-based compilation and execution.

## Getting Started

To get started with vllm-spyre-next, see the [Installation Guide](getting_started/installation.md).

## Documentation

- [Installation](getting_started/installation.md) - Setup and installation instructions

## Requirements

- Python >= 3.11
- torch-spyre (built from source)
- vLLM 0.15.1+cpu
- PyTorch 2.10.0 (CPU version)

## License

Apache 2.0

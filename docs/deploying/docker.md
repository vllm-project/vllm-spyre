# Using Docker

<!--
TODO: Add section on RHOAI officially supported images, once they exist
!-->

## Spyre base images

Base images containing the driver stack for IBM Spyre accelerators are available from the [ibm-aiu](https://quay.io/repository/ibm-aiu/base?tab=tags) organization on Quay. This includes the `torch_sendnn` package, which is required for using torch with Spyre cards.

!!! attention
    These images contain an install of the `torch` package. The specific version installed is guaranteed to be compatible with `torch_sendnn`. Overwriting this install with a different version of `torch` may cause issues.

## Using community built images

Community maintained images are also [available on Quay](https://quay.io/repository/ibm-aiu/vllm-spyre?tab=tags), the latest x86 build is `quay.io/ibm-aiu/vllm-spyre:latest.amd64`.

!!! caution
    These images are provided as a reference and come with no support guarantees.

## Building vLLM Spyre's Docker Image from Source

You can build and run vLLM Spyre from source via the provided <gh-file:docker/Dockerfile.amd64>. To build vLLM Spyre:

```shell
DOCKER_BUILDKIT=1 docker build . --target release --tag vllm/vllm-spyre --file docker/Dockerfile.amd64
```

!!! note
    This Dockerfile currently only supports the x86 platform

## Running vLLM Spyre in a Docker Container

To run your vLLM Spyre image on a host with Spyre cards installed:

```shell
$ docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v /dev/vfio:/dev/vfio \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    vllm/vllm-spyre <model> <args...>
```

To run your vLLM Spyre image on a host without Spyre cards installed:

```shell
$ docker run \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HUGGING_FACE_HUB_TOKEN=<secret>" \
    --env "VLLM_SPYRE_DYNAMO_BACKEND=eager" \
    vllm/vllm-spyre <model> <args...>
```

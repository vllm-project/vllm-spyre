# Using Docker

<!--
TODO: Add section on RHOAI officially supported images, once they exist
!-->

## Building vLLM Spyre's Docker Image from Source

You can build and run vLLM Spyre from source via the provided gh-file:docker/Dockerfile. To build vLLM Spyre:

```shell
DOCKER_BUILDKIT=1 docker build . --target release --tag vllm/vllm-spyre --file docker/Dockerfile
```

:::{note}
This Dockerfile currently only supports the x86 platform
:::

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

:::{tip}
To run your vLLM Spyre image in CPU mode for debugging, or on a machine without Spyre cards installed, add the flag `--env VLLM_SPYRE_DYNAMO_BACKEND=eager`
:::

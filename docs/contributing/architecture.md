# Plugin Architecture

The Spyre plugin extends or replaces three main components in vLLM:

1. Scheduler
2. Model worker and model runner
3. Modeling code

To better understand how these modifications take place, it's helpful to
first consider the state of the native vllm for GPU architecture.

![vLLM architecture](images/vllm_v1.svg)

As we can see, the API server, the engine core and the workers live in
different processes. All three refer to the platform API for backend
specific concerns.

In vLLM-spyre, we provide an implementation for the platform API that is
loaded at the vllm startup time and bootstraps all the other components.

![vLLM Spyre architecture](images/vllm_v1_spyre.svg)

As we can see in the diagram, the plugin modifies mainly the engine core
process and the worker processes. There are some request validation hooks
in the platform API that the API server invokes to make sure that the request
can be handled by the backend, but that is all.

In the engine core we customize the scheduler for the constrains imposed
by static batching and continuous batching.

The changes are broader in the worker process. Here most of the main
classes have Spyre-specific implementations. From the vLLM code we reuse
mainly the sampling code with all the logits processing, as well as the pooling
code for non-generative use cases.

We have model runners for 3 cases: static batching, continuous batching and
pooling. The pooling model runner is very similar to the static batching one,
but the main differences are that it does pooling instead of sampling and it
uses the transformers modeling code instead of the `foundation-model-stack`
code.

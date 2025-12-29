# Multimodal Models in vLLM Spyre

In order to understand how to get multimodal models running through vLLM Spyre, it is important to understand the differences between how models are implemented in vLLM & vLLM Spyre. To illustrate this, we use `llava_next` as an example, because `granite vision` is the only multimodal model currently supported.

## For vLLM
In vLLM, models are implemented as their own model class. The class implementation generally inherits from `SupportsMultiModal`, and importantly, it registers multimodal processing information.

```python
@MULTIMODAL_REGISTRY.register_processor(
    LlavaNextMultiModalProcessor,
    info=LlavaNextProcessingInfo,
    dummy_inputs=LlavaDummyInputsBuilder,
)
```

If you are coming from a background of working with non-multimodal models, the more important pieces to be aware of are how the preprocesing differs, and how things differ in prefill. More specifically:

- While text only models typically use a tokenizer, multimodal models generally have interleaved inputs. The manner in which this is accomplished is by using a *model specific* token that indicates that the corresponding positions should be replaced with multimodal features. Logically this essentially means something like the following:

    - Given the text: `<image> Describe this image` & an example image
    - We preprocess the text *and* the example image
    - Then, we run the preprocessed image through the corresponding part of the model for encoding that modality, e.g., vision encoder + projector
    - Finally, we create merged multimodal embeddings, where the indices for the special `<image>` token are replaced with the extracted visual features, and the non multimodal tokens have embeddings extracted from the LLM

This has a few implications that may be nonobvious. Namely:

1. A picture is worth a lot of tokens; The multimodal features corresponding to each special token are not a single embedding, and tend to vary based on a few factors, e.g., aspect ratio / image size. Bigger images tend to take up more context.

2. Because of the above ^, an expansion step is generally needed to offset the input IDs. For example, if the `<image>` token represents an image that will take up `num_features` in the context, we can replace the `<image>` with `<image>`*`num_features`; this is done in the vLLM model specific preprocessing class & related utils, and lets us to directly mask the extracted multimodal features into the embeddings.

3. Multimodal is most relevant at prefill time, because at decode time, we simply have embeddings from the space of the LLM, and we do not need to encode the multimodal data again. As such, the original data can essentially be dropped after encoding during prefill.

4. Due to the nature of how multimodal embeddings are merged, the model needs to be able to accept embeddings as inputs, and not just token IDs.

5. As a result of ^, we must be careful to handle warmup correctly with respect to `torch.compile`, *especially* when it comes to AIU. More details on this below.

For more extensive documentation in how to implement multimodal in vLLM, see the docs [here](https://docs.vllm.ai/en/latest/contributing/model/multimodal/#prompt-updates) - the above is mostly meant as context for how think of these models with respect to vLLM Spyre.

### Extending to vLLM Spyre
In vLLM Spyre, models are implemented with a generic wrapper around FMS; the implementation is *not* model specific. This adds several points of awkwardness in porting multimodal FMS wrappers into vLLM Spyre. In general, the best way to get the model working is as follows:

1. Make sure it runs correctly with vLLM and the HuggingFace implementation *before* porting the FMS implementation into vLLM Spyre.*

2. Ensure the config is being correctly unwrapped and that the model instance is being recognized as `is_multimodal`. This will cause prefill/decode to use embeddings as inputs instead of token IDs.

3. Ensure that the results of prefill/decode are actually embeddings; the interfaces are not very solidifed yet, so if misconfigured, e.g., underneath it's prepared wrong or the incorrect `iteration` is passed to FMS, it is easy to do things like accidentally getting input IDs instead of embeddings in decode, which can cause cryptic compiler failures. In the future we will warn about this if things aren't looking quite right.

4. Ensure that the preprocessed kwargs are correctly handled and preprocessed (e.g., `pixel_values`). This is currently very model specific and needs to be made generic in the future.

5. When configuring the warmup, ensure that you provide multimodal inputs so that you warm up *all* parts of the model; e.g., in the case of granite vision, providing only input IDs/text embeddings will only warm up the LLM. This will cause crashes when running the compiled model with multimodal inputs, since the multimodal encoder / projector need to be warmed up as well.

** Aside from uniformity, the main reason it's desirable to get the model running in vLLM *before* vLLM Spyre is that even though the model implementation is different, the preprocessor that vLLM uses to intialize it when it is running through vLLM Spyre is based on the underlying config, and is the *same*. This means that to implement the model in FMS, we do not have to reimplement any of the preprocessing wrapping or prompt substitution/multimodal token expansion logic, which is very well patterned in vLLM. This is ideal for keeping changes for specific model architectures in our generic wrapper to a minimum.

### TEST 1 - Disable prefix caching

from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.config import AttentionConfig


def print_outputs(outputs, engine):
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        print("-" * 50)
    for m in engine.llm_engine.get_metrics():
        if "cache" in m.name:
            print(m.name, m.value)


def main():
    # Configuration
    # MODEL = "ibm-granite/granite-3.0-8b-base"  # Tiny model
    MODEL = "ibm-granite/granite-3.3-8b-instruct"  # Instruct model
    # MODEL = "ibm-granite/granite-4.0-tiny-preview"  # Granite 3
    # MODEL = "ibm-granite/granite-4.0-h-small"       # Granite 4
    # MODEL = "ibm-granite/granite-4.0-h-tiny"       # Granite 4
    # MODEL = "facebook/opt-125m" # Small model

    # Sampling parameter for the inference process
    sampling_params = SamplingParams(
        max_tokens=5,  # Maximum number of tokens to produce
    )

    # Prompts to use for inference
    prompts = [
        "What are IBMs main businesses?",
    ]

    engine = LLM(
        model=MODEL,  # Model to use for inference.
        gpu_memory_utilization=0.9,  # By increasing utilization, you can provide more KV cache space.
        enable_prefix_caching=True,  # Flag determining whether prefix caching is enabled or disabled.
        # enforce_eager=True,              # Flag determining whether eager mode or torch.compile should be used.
        # mamba_ssm_cache_dtype="float32", # Datatype of the mamba cache (if any).
        # dtype="float32",                 # Datatype of the model.
        # max_num_batched_tokens=8192,    # Maximum number of tokens for a prefill before being chunked
        # disable_cascade_attn=True,  ## compliates logic with mamba
        disable_log_stats=False,  ## stats
        attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
    )

    # Generate response for prompt 0
    outputs = engine.generate(prompts[0], sampling_params)
    print_outputs(outputs, engine)


if __name__ == "__main__":
    main()

import os
from vllm import LLM, SamplingParams
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.config import AttentionConfig, CompilationConfig

PROMPTS = [
    "What are IBMs main businesses?",
]

def print_outputs(outputs, engine):
    print("-" * 50)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")
        print("-" * 50)
    for m in engine.llm_engine.get_metrics():
        if "cache" in m.name:
            print(m.name, m.value)


def main(args):
    sampling_params = SamplingParams(max_tokens=5)

    engine = LLM(
        model=args.model,
        # gpu_memory_utilization=0.9,
        # enable_prefix_caching=True,
        # enforce_eager=args.enforce_eager,
        enforce_eager=True,
        # dtype="float16",
        # disable_log_stats=False,
        # attention_config=AttentionConfig(backend=AttentionBackendEnum.CUSTOM),
        # compilation_config=CompilationConfig(custom_ops=[args.custom_ops]),
    )

    outputs = engine.generate(
        PROMPTS,
        sampling_params
    )
    print_outputs(outputs, engine)


if __name__ == "__main__":
    """
    python examples/vllm_spyre_inference.py \
        --model ibm-granite/granite-3.3-8b-instruct \
        --enforce_eager \
        --custom_ops all
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default="ibm-granite/granite-3.3-8b-instruct",
        help="Model to run E2E inference with",
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Skip torch.compile, run in eager mode",
    )
    parser.add_argument(
        "--custom_ops",
        type=str,
        default="all",
        required=False,
        help="CustoOps to enable",
    )
    args = parser.parse_args()

    main(args)

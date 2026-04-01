"""E2E test script for selective custom op offloading.

Environment variables:
  VLLM_SPYRE_CUSTOM_OPS: "all", "none", "rms_norm", "silu_and_mul", or comma-separated
  VLLM_SPYRE_DYNAMO_BACKEND: "eager" for no torch.compile
"""

import os
import logging

# Configure logging before imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("vllm_spyre_next").setLevel(logging.DEBUG)

from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig


def main():
    MODEL = os.environ.get("TEST_MODEL", "ibm-granite/granite-3.3-8b-instruct")

    sampling_params = SamplingParams(max_tokens=5)
    prompts = ["What are IBMs main businesses?"]

    engine = LLM(
        model=MODEL,
        dtype="float16",
        # Enable custom ops - this triggers forward_oot dispatch
        compilation_config=CompilationConfig(custom_ops=["none"]),
        # Skip torch.compile to avoid torch_spyre decomposition issues
        enforce_eager=True,
        disable_log_stats=False,
    )

    print("\n" + "=" * 50)
    print("Running generation...")
    print("=" * 50 + "\n")

    outputs = engine.generate(prompts[0], sampling_params)
    for output in outputs:
        print(f"Generated: {output.outputs[0].text!r}")


if __name__ == "__main__":
    main()

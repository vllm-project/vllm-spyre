"""
This example shows how to run offline inference using continuous batching 
on CPU.
"""

import os
import platform
import time

from vllm import LLM, SamplingParams

# Continuous batching currently requires the installation of the branch 
# https://github.com/foundation-model-stack/foundation-model-stack/tree/paged_attn_mock

max_tokens1 = 65
max_tokens2 = 67
max_tokens3 = 7
max_num_seqs = 2  # max batch size

if platform.machine() == "arm64":
    print("Detected arm64 running environment. "
          "Setting HF_HUB_OFFLINE=1 otherwise vllm tries to download a "
          "different version of the model using HF API which might not work "
          "locally on arm64.")
    os.environ["HF_HUB_OFFLINE"] = "1"

os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = 'eager'
os.environ['VLLM_SPYRE_USE_CB'] = '1'
os.environ['VLLM_USE_V1'] = '1'

template = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user.\n\n### Instruction:\n{}\n\n### Response:")

instructions = [
    "Provide a list of instructions for preparing chicken soup for a family" + \
        " of four.",
    "Provide instructions for preparing chicken soup.",
    "Provide a list of instructions for preparing chicken soup for a family.",
]

prompts = [template.format(instr) for instr in instructions]

max_tokens_list = [max_tokens1, max_tokens2, max_tokens3]

sampling_params = [
    SamplingParams(max_tokens=mt, temperature=0.0, ignore_eos=True)
    for mt in max_tokens_list
]

# Create an LLM.
llm = LLM(model="/models/llama-194m",
          tokenizer="/models/llama-194m",
          max_model_len=2048,
          block_size=2048,
          max_num_seqs=max_num_seqs)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("=============== GENERATE")
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
print("Time elaspsed for %d tokens is %.2f sec" %
      (len(outputs[0].outputs[0].token_ids), time.time() - t0))
print("===============")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
print("===============")
for output in outputs:
    print(output.outputs[0])

import os
import platform
import time

from vllm import LLM, SamplingParams

# RUN with fms branch: https://github.com/foundation-model-stack/
# foundation-model-stack/tree/paged_attn_mock

max_tokens1 = 65
max_tokens2 = 67
max_tokens3 = 7
max_num_seqs = 2  # defines max batch size

if platform.machine() == "arm64":
    print("Detected arm64 running environment. "
          "Setting HF_HUB_OFFLINE=1 otherwise vllm tries to download a "
          "different version of the model using HF API which might not work "
          "locally on arm64.")
    os.environ["HF_HUB_OFFLINE"] = "1"

# defining here to be able to run/debug directly from VSC (not via terminal)
os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = 'eager'
os.environ['VLLM_SPYRE_USE_CB'] = '1'
os.environ['VLLM_USE_V1'] = '1'

# Sample prompts.
template = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user.\n\n### Instruction:\n{}\n\n### Response:")

prompt1 = template.format(
    "Provide a list of instructions for preparing chicken soup for a family "
    "of four.")

prompt2 = template.format("Provide instructions for preparing chicken soup.")

prompt3 = template.format(
    "Provide a list of instructions for preparing chicken soup for a family.")

prompts = [
    prompt1,
    prompt2,
    prompt3,
]

# Create a sampling params object.
sampling_params1 = SamplingParams(max_tokens=max_tokens1,
                                  temperature=0.0,
                                  ignore_eos=True)

sampling_params2 = SamplingParams(max_tokens=max_tokens2,
                                  temperature=0.0,
                                  ignore_eos=True)

sampling_params3 = SamplingParams(max_tokens=max_tokens3,
                                  temperature=0.0,
                                  ignore_eos=True)

sampling_params = [
    sampling_params1,
    sampling_params2,
    sampling_params3,
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

import os
import sys
import time

sys.path.insert(0, "/workspace/vllm")
from vllm import LLM, SamplingParams
import vllm
print('vllm version: ', vllm.__version__)
# RUN with fms branch: https://github.com/foundation-model-stack/
# foundation-model-stack/tree/paged_attn_mock

# set left padding
max_tokens1 = 3
max_tokens2 = 6
# max_tokens3 = 7

max_num_seqs = 2  # defines max batch size

# defining here to be able to run/debug directly from VSC (not via terminal)
os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = 'eager'
os.environ['VLLM_SPYRE_USE_CB'] = '1'
os.environ['VLLM_SPYRE_RM_PADDED_BLOCKS'] = '1'
os.environ['VLLM_USE_V1'] = '1'

os.environ['VLLM_DT_MAX_CONTEXT_LEN'] = '2048'
os.environ['VLLM_DT_MAX_BATCH_SIZE'] = str(max_num_seqs)
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(12355)
os.environ['VLLM_LOGGING_LEVEL'] = 'DEBUG'

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
    # prompt3,
]

# Create a sampling params object.
sampling_params1 = SamplingParams(max_tokens=max_tokens1,
                                  temperature=0.0,
                                  ignore_eos=True)

sampling_params2 = SamplingParams(max_tokens=max_tokens2,
                                  temperature=0.0,
                                  ignore_eos=True)

# sampling_params3 = SamplingParams(max_tokens=max_tokens3,
#                                   temperature=0.0,
#                                   ignore_eos=True)

sampling_params = [
    sampling_params1,
    sampling_params2,
    # sampling_params3,
]

# Create an LLM.
# llm = LLM(model="/workspace/models/granite-3.2-8b-instruct",
#           tokenizer="/workspace/models/granite-3.2-8b-instruct",
#           max_model_len=2048,
#           block_size=2048)

llm = LLM(model="/workspace/models/llama-194m",
          tokenizer="/workspace/models/llama-194m",
          max_model_len=1000,
          block_size=2048,
          max_num_seqs=max_num_seqs)

# llm = LLM(model="/workspace/models/granite-3b-base",
#           tokenizer="/workspace/models/granite-3b-base",
#           max_model_len=2048,
#           block_size=2048,
#           tensor_parallel_size = 2)

# llm = LLM(model="/workspace/models/llama-3.1-8b-instruct",
#           tokenizer="/workspace/models/llama-3.1-8b-instruct",
#           max_model_len=2048,
#           block_size=2048)

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

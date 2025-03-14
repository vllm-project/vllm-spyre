import os
import time

from vllm import LLM, SamplingParams

max_tokens = 15
early_stop = 5

os.environ["VLLM_SPYRE_WARMUP_PROMPT_LENS"] = '64'
os.environ["VLLM_SPYRE_WARMUP_NEW_TOKENS"] = str(max_tokens)
os.environ['VLLM_SPYRE_WARMUP_BATCH_SIZES'] = '4'

# defining here to be able to run/debug directly from VSC (not via terminal)
os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = 'eager'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'

# Sample prompts.
template = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user.\n\n### Instruction:\n{}\n\n### Response:")
prompt1 = template.format(
    "Provide a list of instructions for preparing chicken soup for a family "
    "of four.")
prompts = [prompt1, prompt1, prompt1, prompt1]

# Create a sampling params object: first sequence will terminate early and will
# replaced with continuous batching
sampling_params = []
for i in range(4):
    sampling_params.append(
        SamplingParams(max_tokens=early_stop if i == 0 else max_tokens,
                       temperature=0.0,
                       ignore_eos=True))
# Create an LLM
llm = LLM(model="models/llama-194m",
          tokenizer="models/llama-194m",
          max_model_len=2048,
          block_size=2048)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("=============== GENERATE")
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
print("Time elaspsed for %d tokens is %.2f sec" %
      (len(outputs[0].outputs[0].token_ids), time.time() - t0))
for output in outputs:
    print(output.outputs[0])

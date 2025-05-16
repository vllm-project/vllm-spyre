""" Test for online serving.

On the server side, run the following commands
    python3 -m vllm.entrypoints.openai.api_server \
        --model /models/llama-7b-chat/ \
        --max-model-len=2048 \
        --block-size=2048

Then the default batch size 1, max prompt length of 64 and maximum of 20
ecode tokens is used. Otherwise change the behavior with the environment 
variables `VLLM_SPYRE_WARMUP_BATCH_SIZES`, `VLLM_SPYRE_WARMUP_PROMPT_LENS`, 
and `VLLM_SPYRE_WARMUP_NEW_TOKENS`.
"""

import time

from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id

template = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request. Be polite in your response to the "
    "user.\n\n### Instruction:\n{}\n\n### Response:")
prompt1 = template.format(
    "Provide a list of instructions for preparing chicken soup for a family "
    "of four.")

prompt2 = template.format(
    "Please compare New York City and Zurich and provide a list of attractions "
    "for each city.")

prompt3 = template.format(
    "Provide detailed instructions for preparing asparagus soup for a family "
    "of four.")

prompts = [prompt1, prompt2, prompt3]

# make sure that the specified batch size is in VLLM_SPYRE_WARMUP_BATCH_SIZES
batch_size = 1
print('submitting prompts of batch size', batch_size)

# making sure not to submit more prompts than the batch size
for i in range(0, len(prompts), batch_size):
    prompt = prompts[i:i + batch_size]

    # Completion API
    stream = False
    max_tokens = 20

    print(f"Prompt: {prompt}")
    start_t = time.time()

    completion = client.completions.create(model=model,
                                           prompt=prompt,
                                           echo=False,
                                           n=1,
                                           stream=stream,
                                           temperature=0.0,
                                           max_tokens=max_tokens)

    end_t = time.time()
    print("Results:")
    if stream:
        for c in completion:
            print(c)
    else:
        print(completion)

    total_t = end_t - start_t
    print(f"Duration: {total_t}s")

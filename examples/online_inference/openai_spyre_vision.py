""" 
This example shows how to use Spyre with vLLM for running online inference,
using granite vision. Note that currently, multimodal is *not* supported for
static baching!

First, start the server with the following command:

VLLM_SPYRE_USE_CB=1 \
VLLM_SPYRE_DYNAMO_BACKEND=sendnn \
vllm serve 'ibm-granite/granite-vision-3.3-2b' \
    --max-model-len=16384 \
    --max-num-seqs=4

This sets up a server with max batch size 4. To actually exercise continuous 
batching make sure to submit multiple prompts at once by running this script 
with `--batch_size` > 1. Note that an image can take up around 5k tokens in
the maximal case, so be sure to consider this when setting the max-model-len
and running this script.
"""

import argparse
import time

from openai import OpenAI

parser = argparse.ArgumentParser(
    description="Script to submit an inference request to vllm server.")

parser.add_argument(
    "--max_tokens",
    type=int,
    default=8,
    help="Maximum tokens.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
)
parser.add_argument(
    "--num_prompts",
    type=int,
    default=4,
)
parser.add_argument(
    "--stream",
    action=argparse.BooleanOptionalAction,
    help="Whether to stream the response.",
)

args = parser.parse_args()

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

def get_vllm_prompts(num_prompts):
    """Get the vLLM prompts to be processed."""
    template = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n<image>\n{}\n<|assistant|>\n"  # noqa: E501

    img_urls = [
        "https://vllm-public-assets.s3.us-west-2.amazonaws.com/vision_model_images/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",  # noqa: E501
        "https://vllm-public-assets.s3.us-west-2.amazonaws.com/multimodal_asset/duck.jpg",  # noqa: E501
    ]

    instructions = [
        "describe this image.",
        "what is shown in this image?",
        "are there any animals in this image?",
    ]

    prompts = []
    for img_url in img_urls:
        for instr in instructions:
            prompts.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instr},
                        {
                            "type": "image_url",
                            "image_url": {"url": img_url},
                        },
                    ],
                }                
            )

    prompts = prompts * (num_prompts // len(prompts) + 1)
    return prompts[:num_prompts]

models = client.models.list()
model = models.data[0].id


prompts = get_vllm_prompts(args.num_prompts)
batch_size = args.batch_size
print('submitting prompts of batch size', batch_size)

# making sure not to submit more prompts than the batch size
for i in range(0, len(prompts), batch_size):
    prompt = prompts[i:i + batch_size]

    stream = args.stream

    print(f"Prompt: {prompt}")
    start_t = time.time()

    chat_completion = client.chat.completions.create(
        messages=prompt,
        model=model,
        max_completion_tokens=args.max_tokens,
        stream=stream,
    )

    end_t = time.time()
    print("Results:")
    if stream:
        for c in chat_completion:
            print(c.choices[0].delta.content, end="")
    else:
        print(chat_completion.choices[0].message.content)

    total_t = end_t - start_t
    print(f"\nDuration: {total_t}s")

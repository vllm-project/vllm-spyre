"""
This example shows how to run offline inference using continuous batching.
"""
from vllm.assets.image import ImageAsset
import argparse
import os
import platform
import time

from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default="ibm-granite/granite-vision-3.3-2b")
parser.add_argument("--max_model_len",
                    "--max-model-len",
                    type=int,
                    default=8192) # one image has a max context of ~5k
parser.add_argument("--max_num_seqs", "--max-num-seqs", type=int, default=2)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--num-prompts", "-n", type=int, default=10)
parser.add_argument(
    "--max-tokens",
    type=str,
    default="65",
    help="Comma separated list of max tokens to use for each prompt. "
    "This list is repeated until prompts are exhausted.")
parser.add_argument("--backend",
                    type=str,
                    default='sendnn',
                    choices=['eager', 'sendnn'])
args = parser.parse_args()

max_num_seqs = args.max_num_seqs  # defines the max batch size

if platform.machine() == "arm64":
    print("Detected arm64 running environment. "
          "Setting HF_HUB_OFFLINE=1 otherwise vllm tries to download a "
          "different version of the model using HF API which might not work "
          "locally on arm64.")
    os.environ["HF_HUB_OFFLINE"] = "1"

os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = args.backend
os.environ['VLLM_SPYRE_USE_CB'] = '1'

template = "<|system|>\nA chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\n<|user|>\n<image>\n{}\n<|assistant|>\n"

images = [
    ImageAsset('cherry_blossom').pil_image,
    ImageAsset('stop_sign').pil_image,
]   


instructions = [
    "describe this image.",
    "what is shown in this image?",
    "what kind of flowers are these?",
]


prompts = []
for img in images:
    width, height = img.size
    for instr in instructions:
        new_width = int(.1 * width)
        new_height = int(.1 * height)
        prompts.append(
            {
                "prompt": template.format(instr),
                "multi_modal_data": {
                    "image": img.resize((new_width, new_height)),
                }    
            }
        )


prompts = prompts * (args.num_prompts // len(prompts) + 1)
prompts = prompts[0:args.num_prompts]

# Set differring max_tokens so that the requests drop out of the batch at
# different times
max_tokens = [int(v) for v in args.max_tokens.split(",")]
max_tokens = max_tokens * (args.num_prompts // len(max_tokens) + 1)
max_tokens = max_tokens[0:args.num_prompts]

sampling_params = [
    SamplingParams(max_tokens=m, temperature=0.0, ignore_eos=True)
    for m in max_tokens
]

# Create an LLM.
llm = LLM(model=args.model,
          tokenizer=args.model,
          max_model_len=args.max_model_len,
          max_num_seqs=max_num_seqs,
          tensor_parallel_size=args.tp)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("=============== GENERATE")
t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
print("Time elaspsed for %d tokens is %.2f sec" %
      (len(outputs[0].outputs[0].token_ids), time.time() - t0))
print("===============")
for output in outputs:
    print(output.outputs[0])
print("===============")
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    # Prompt contains expanded image tokens, so just print
    # what's after the last one for readability
    print(f"\nPrompt:\n {prompt.split("<image>")[-1].split("<assistant>")[0].strip()}")
    print(f"\nGenerated text:\n {generated_text!r}\n")
    print("-----------------------------------")

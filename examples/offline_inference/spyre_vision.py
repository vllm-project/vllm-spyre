# Ugly wip quick demo script based on spyre_inference.py

from vllm.assets.image import ImageAsset
image_pil = ImageAsset('cherry_blossom').pil_image

import argparse
import gc
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
                    default=8192) # Need to be bigger then 5k at least to make sure we can fit the biggest images
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--prompt-len", type=int, default=64)
parser.add_argument(
    "--max-tokens",
    type=int,
    default=3,
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=1,
)
parser.add_argument("--backend",
                    type=str,
                    default='eager', # for testing for now
                    choices=['eager', 'sendnn']) 
parser.add_argument("--compare-with-cpu",
                    action=argparse.BooleanOptionalAction)
args = parser.parse_args()

if platform.machine() == "arm64":
    print("Detected arm64 running environment. "
          "Setting HF_HUB_OFFLINE=1 otherwise vllm tries to download a "
          "different version of the model using HF API which might not work "
          "locally on arm64.")
    os.environ["HF_HUB_OFFLINE"] = "1"

# TODO - handle setting warmup prompt len based on max image feature size.
os.environ["VLLM_SPYRE_WARMUP_PROMPT_LENS"] = "6400" #str(args.prompt_len)
os.environ["VLLM_SPYRE_WARMUP_NEW_TOKENS"] = str(args.max_tokens)
os.environ['VLLM_SPYRE_WARMUP_BATCH_SIZES'] = str(args.batch_size)
os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = args.backend

# Sendnn currently explodes with DtException: Insufficient sengraphs passed for decoder compilation, expected >=2 but got 0, file /project_src/deeptools/deeprt/deeprt.cpp line 3303
# Probably we are setting a wrong env var here, which is breaking in compile
sampling_params = SamplingParams(max_tokens=args.max_tokens,
                                 temperature=0.0,
                                 ignore_eos=True)
# Create an LLM.
llm = LLM(model=args.model,
          tokenizer=args.model,
          max_model_len=args.max_model_len,
          tensor_parallel_size=args.tp,
          enforce_eager=True)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("=============== GENERATE")
prompts = {
    "prompt": "<image> can you describe this image?",
    "multi_modal_data": {
        "image": image_pil,
    }
}

t0 = time.time()
outputs = llm.generate(prompts, sampling_params)
print("Time elaspsed for %d tokens is %.2f sec" %
      (len(outputs[0].outputs[0].token_ids), time.time() - t0))
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if args.tp > 1:
    # needed to prevent ugly stackdump caused by sigterm
    del llm
    gc.collect()

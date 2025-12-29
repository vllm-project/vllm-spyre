"""
This example shows how to run offline inference using continuous batching.
"""
import argparse
import os
import platform
import time

from fms.utils.spyre import paged
from fms.models import get_model
from fms.utils import serialization
from fms.utils.generation import generate as fms_generate
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
from vllm.assets.image import ImageAsset
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default="ibm-granite/granite-vision-3.3-2b")
parser.add_argument("--max_model_len",
                    "--max-model-len",
                    type=int,
                    default=8192)  # one image has a max context of ~5k
parser.add_argument("--max_num_seqs", "--max-num-seqs", type=int, default=2)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--num-prompts", "-n", type=int, default=1)
parser.add_argument(
    "--max-tokens",
    type=str,
    default="8",
    help="Comma separated list of max tokens to use for each prompt. "
    "This list is repeated until prompts are exhausted.")
parser.add_argument("--backend",
                    type=str,
                    default='sendnn',
                    choices=['eager', 'sendnn'])
parser.add_argument("--compare-with-transformers",
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--compare-with-fms",
                    action=argparse.BooleanOptionalAction)
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
        prompts.append({
            "prompt": template.format(instr),
            "multi_modal_data": {
                "image": img.resize((new_width, new_height)),
            }
        })

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

llm = LLM(model=args.model,
          tokenizer=args.model,
          max_model_len=args.max_model_len,
          max_num_seqs=max_num_seqs,
          tensor_parallel_size=args.tp)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("=============== GENERATE")
t0 = time.time()
vllm_outputs = llm.generate(prompts, sampling_params)
vllm_results = [x.outputs[0].text for x  in vllm_outputs] # raw texts
raw_prompts = [prompt["prompt"] for prompt in prompts]

print("Time elaspsed for %d tokens is %.2f sec" %
      (len(vllm_outputs[0].outputs[0].token_ids), time.time() - t0))
print("===============")
for output in vllm_results:
    print(output)
print("===============")
for prompt, generated_text in zip(prompts, vllm_results):
    prompt = prompt["prompt"]
    # Prompt contains expanded image tokens, so just print
    # what's after the last one for readability
    print(f"\nPrompt:\n {prompt}")
    print(f"\nGenerated text:\n {generated_text!r}\n")
    print("-----------------------------------")

# Alternate implentatinos to compare against; we have seen mismatches for parity
# between FMS and Transformers, so for completeness, we allow comparison against
# both Transformers on CPU and FMS.
def get_transformers_results(model_path, vllm_prompts):
    """Process the results for HF Transformers running on CPU."""
    hf_results = []
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForVision2Seq.from_pretrained(model_path)
    num_prompts = len(vllm_prompts)

    for i in range(num_prompts):
        # Assume prompts are preformatted and that we don't need to use chat template
        vllm_req = prompts[i]

        inputs = processor(
            text=vllm_req["prompt"],
            images=vllm_req["multi_modal_data"]["image"],
            return_tensors="pt"
        )

        hf_output = model.generate(**inputs, max_new_tokens=max_tokens[i])
        # NOTE: Image tokens are expanded in the llava next preprocessor;
        # be sure to check this depending on the model if this is used for
        # others, especially if they bundle their own code!
        num_expanded_toks = inputs.input_ids.shape[1]
        hf_out_toks = hf_output[0][num_expanded_toks:]

        hf_generated_text = processor.decode(hf_out_toks)
        hf_results.append(hf_generated_text)
    return hf_results

def get_fms_results(model_path, vllm_prompts):
    """Process the results for FMS running on CPU."""
    fms_results = []
    processor = AutoProcessor.from_pretrained(model_path)
    num_prompts = len(vllm_prompts)

    # head_dim expansion required for granite vision
    serialization.extend_adapter(
        "llava_next", "hf", ["weight_expansion_for_mismatched_head_dim"]
    )
    config_dict = {}
    config_dict["head_dim"] = 128

    # Load, but don't compile (compare to CPU)
    model = get_model(
        "hf_pretrained",
        model_path,
        data_type=torch.float32,
        device_type="cpu",
        fused_weights=False,
        override_hf_pretrained_config=True,
        text_config=config_dict,
    )


    for i in range(num_prompts):
        # Assume prompts are preformatted and that we don't need to use chat template
        vllm_req = prompts[i]

        inputs = processor(
            text=vllm_req["prompt"],
            images=vllm_req["multi_modal_data"]["image"],
            return_tensors="pt"
        )

        input_ids = inputs.pop("input_ids")
        inputs["attn_name"] = "sdpa_causal" # TODO need to handle padding for paged attn

        fms_output = fms_generate(
            model,
            input_ids,
            max_new_tokens=max_tokens[i],
            use_cache=True,
            do_sample=False,
            max_seq_len=input_ids.shape[1] + max_tokens[i],
            extra_kwargs=inputs,
            prepare_model_inputs_hook=model.prepare_inputs_for_generation,
        )
        num_expanded_toks = input_ids.shape[1]
        fms_out_toks = fms_output[0][num_expanded_toks:]

        fms_generated_text = processor.decode(fms_out_toks)
        fms_results.append(fms_generated_text)
    return fms_results


def compare_results(prompts: list[str], outputs_a: list[str], outputs_b: list[str], name_a: str, name_b: str):
    print(f"Comparing {name_a} results with {name_b}")
    print("===============")
    any_differ = False
    for idx, (result_a, result_b) in enumerate(zip(outputs_a, outputs_b)):
        # Assume prompts are preformatted and that we don't need to use chat template

        if result_a != result_b:
            img_tok_idx = prompts[idx].index("<image>")
            gen_prompt_idx = prompts[idx].index("<|assistant|>")
            raw_prompt = prompts[idx][img_tok_idx:gen_prompt_idx].strip()

            any_differ = True
            print(f"Results for prompt {idx} differ!")
            print(f"\nPrompt (excluding system/generation prompt):\n {repr(raw_prompt)}")
            print(
                f"\n{name_a} generated text:\n {result_a}\n")
            print(f"\n{name_b} generated text:\n {result_b}\n")
            print("-----------------------------------")

    if not any_differ:
        print("\nAll results match!\n")

if args.compare_with_transformers:
    transformers_results = get_transformers_results(
        model_path=args.model,
        vllm_prompts=prompts,
    )

    compare_results(
        prompts=raw_prompts,
        outputs_a=transformers_results,
        outputs_b=vllm_results,
        name_a="transformers [cpu]",
        name_b="vllm [spyre]",
    )

if args.compare_with_fms:
    fms_results = get_fms_results(
        model_path=args.model,
        vllm_prompts=prompts,
    )

    compare_results(
        prompts=raw_prompts,
        outputs_a=fms_results,
        outputs_b=vllm_results,
        name_a="FMS [cpu]",
        name_b="vllm [spyre]",
    )

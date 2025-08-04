import argparse
import os
import platform
import sys
import time

sys.path.insert(0, "/workspace/vllm")
sys.path.insert(0, "/workspace/vllm-spyre/tests")
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = str(12356)

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default="/workspace/models/llama-194m")
parser.add_argument("--max_model_len",
                    "--max-model-len",
                    type=int,
                    default=4096)
parser.add_argument("--max_prompt_len",
                    "--max-prompt-len",
                    type=int,
                    default=4096)
parser.add_argument("--max_num_seqs", "--max-num-seqs", type=int, default=4)
parser.add_argument("--tp", type=int, default=1)
parser.add_argument("--num-prompts", "-n", type=int, default=1)
parser.add_argument("--num_new_tokens", type=int, default=1)
parser.add_argument("--compare-with-cpu",
                    action=argparse.BooleanOptionalAction)
parser.add_argument("--trunc_print_len",
                    "--trunc-print-len",
                    type=int,
                    required=False)
args = parser.parse_args()

trunc = args.trunc_print_len

max_num_seqs = args.max_num_seqs  # defines the max batch size
assert args.max_prompt_len <= args.max_model_len

if platform.machine() == "arm64":
    print("Detected arm64 running environment. "
          "Setting HF_HUB_OFFLINE=1 otherwise vllm tries to download a "
          "different version of the model using HF API which might not work "
          "locally on arm64.")
    os.environ["HF_HUB_OFFLINE"] = "1"

if "VLLM_SPYRE_DYNAMO_BACKEND" not in os.environ:
    os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = 'eager'
os.environ['VLLM_SPYRE_USE_CB'] = '1'

# prompt = ("1 2 3 ")
prompt = ' '.join([str(i) for i in range(1, 10)])
prompt += ' '
prompt *= 228

if True:
    prompt = ' ' + prompt
prompts = [prompt]


tokenizer = AutoTokenizer.from_pretrained(args.model)


tokenized_prompts = tokenizer(prompts)["input_ids"]
tokenized_prompts = [p for p in tokenized_prompts]

n_pads = args.max_prompt_len - len(tokenized_prompts[0])
tokenized_prompts = [tokenized_prompt[:args.max_prompt_len] for tokenized_prompt in tokenized_prompts]
# print('tokenized_prompts', tokenized_prompts)



prompt_lens = [len(p) for p in tokenized_prompts]

max_prompt = max(prompt_lens)
min_prompt = min(prompt_lens)

if max_prompt < args.max_prompt_len:
    print(f"Warning, none of the prompts reach the maximum length"
          f"({args.max_prompt_len})")

print(f"All prompts have lengths between {min_prompt} and {max_prompt}")


tokens_to_generate = [args.num_new_tokens]


sampling_params = [
    SamplingParams(max_tokens=t, temperature=0.0, ignore_eos=True)
    for t in tokens_to_generate
]

vllm_token_prompts = [
    TokensPrompt(prompt_token_ids=p) for p in tokenized_prompts
]

# Create an LLM.
llm = LLM(model=args.model,
          tokenizer=args.model,
          max_model_len=args.max_model_len,
          block_size=2048,
          max_num_seqs=max_num_seqs,
          tensor_parallel_size=args.tp)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
print("=============== GENERATE")
t0 = time.time()
outputs = llm.generate(vllm_token_prompts, sampling_params)
print("Time elapsed for all prompts is %.2f sec" % (time.time() - t0))
print("===============")
for output, prompt in zip(outputs, prompts):
    generated_ids = output.outputs[0].token_ids
    tokenized_prompts_crop = tokenized_prompts[0][-18:]
    print('lenght of prompt', len(tokenized_prompts[0]))
    print('requested tokens', tokens_to_generate[0])
    print('generated', len(output.outputs[0].token_ids), 'vLLM tokens')
    print(f"\n tokenized_prompts:\n {tokenized_prompts_crop!r}")
    print(f"\nGenerated text (truncated):\n {generated_ids!r}\n")
    print("-----------------------------------")

if args.compare_with_cpu:
    print("Comparing results with HF on cpu")
    print("===============")
    any_differ = False

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model)

    for i in range(args.num_prompts):
        prompt = prompts[i]

        hf_input_tokens = torch.tensor(tokenized_prompts[i]).unsqueeze(0)
        hf_output = model.generate(hf_input_tokens,
                                   do_sample=False,
                                   min_new_tokens=tokens_to_generate[i],
                                   max_new_tokens=tokens_to_generate[i],
                                   return_dict_in_generate=True,
                                   output_scores=True)
        
        # decode output tokens after first removing input tokens (prompt)
        hf_generated_text = tokenizer.batch_decode(
            hf_output.sequences[:, len(hf_input_tokens[0]):])[0]
        print('generated', len(hf_generated_text), 'HF tokens')
        if hf_generated_text != outputs[i].outputs[0].text:
            any_differ = True
            spyre_output = outputs[i].outputs[0].text
            print(f"Results for prompt {i} differ on cpu")
            # print(f"\nPrompt:\n {prompt[-10:]!r}")
            print(f"\nSpyre generated text:\n {spyre_output[:trunc]!r}\n")
            print(f"\nCPU generated text:\n {hf_generated_text[:trunc]!r}\n")
            print("-----------------------------------")

    if not any_differ:
        print("\nAll results match!\n")

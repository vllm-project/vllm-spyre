import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

DISABLE_ASSERTS = False  # used for debugging

ISCLOSE_REL_TOL_CPU = 0.1
ISCLOSE_REL_TOL_SPYRE = 0.1


# vLLM / Spyre
def generate_spyre_vllm_output(model: str, prompts: List[str],
                               warmup_shapes: List[Tuple[int, int, int]],
                               max_model_len: int, block_size: int,
                               sampling_params: Union[SamplingParams,
                                                      List[SamplingParams]],
                               tensor_parallel_size: int,
                               backend: str) -> List[Dict[str, Any]]:

    warmup_prompt_length = [t[0] for t in warmup_shapes]
    warmup_new_tokens = [t[1] for t in warmup_shapes]
    warmup_batch_size = [t[2] for t in warmup_shapes]

    os.environ['VLLM_SPYRE_WARMUP_PROMPT_LENS'] = ','.join(
        str(val) for val in warmup_prompt_length)
    os.environ['VLLM_SPYRE_WARMUP_NEW_TOKENS'] = ','.join(
        str(val) for val in warmup_new_tokens)
    os.environ['VLLM_SPYRE_WARMUP_BATCH_SIZES'] = ','.join(
        str(val) for val in warmup_batch_size)
    os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = backend

    vllm_model = LLM(model=model,
                     tokenizer=model,
                     max_model_len=max_model_len,
                     block_size=block_size,
                     tensor_parallel_size=tensor_parallel_size)

    vllm_outputs = vllm_model.generate(prompts, sampling_params)

    results = []
    for req_output in vllm_outputs:
        result = {}
        result['text'] = req_output.outputs[0].text
        result['token_ids'] = req_output.outputs[0].token_ids
        result['tokens'] = tuple([
            req_output.outputs[0].logprobs[i][t].decoded_token
            for i, t in enumerate(result['token_ids'])
        ])
        result['logprobs'] = tuple([
            req_output.outputs[0].logprobs[i][t].logprob
            for i, t in enumerate(result['token_ids'])
        ])
        results.append(result)

    return results


# Hugging Face
def generate_hf_output(
        model: str, prompts: List[str],
        max_new_tokens: Union[int, List[int]]) -> List[Dict[str, Any]]:

    if not isinstance(max_new_tokens, list):
        max_new_tokens = [max_new_tokens] * len(prompts)

    hf_model = AutoModelForCausalLM.from_pretrained(model)
    hf_tokenizer = AutoTokenizer.from_pretrained(model)

    results = []
    for prompt_index, prompt in enumerate(prompts):
        hf_input_tokens = hf_tokenizer(prompt, return_tensors="pt").input_ids
        hf_output = hf_model.generate(
            hf_input_tokens,
            do_sample=False,
            max_new_tokens=max_new_tokens[prompt_index],
            return_dict_in_generate=True,
            output_scores=True)

        # decode output tokens after first removing input tokens (prompt)
        hf_generated_text = hf_tokenizer.batch_decode(
            hf_output.sequences[:, len(hf_input_tokens[0]):])[0]
        hf_transition_scores = hf_model.compute_transition_scores(
            hf_output.sequences, hf_output.scores, normalize_logits=True)

        # return HF generated text, tokens, token ids and logprobs
        result = {}
        result['text'] = hf_generated_text
        result['token_ids'] = []
        result['tokens'] = []
        result['logprobs'] = []
        for tok_index, hf_logprob in enumerate(hf_transition_scores[0]):
            hf_token_id = hf_output.sequences[0][tok_index +
                                                 len(hf_input_tokens[0])]
            result['token_ids'].append(hf_token_id.item())
            result['tokens'].append(hf_tokenizer.decode(hf_token_id))
            result['logprobs'].append(hf_logprob.item())
        result['token_ids'] = tuple(result['token_ids'])
        result['tokens'] = tuple(result['tokens'])
        result['logprobs'] = tuple(result['logprobs'])
        results.append(result)

    return results


# compare results
def compare_results(model: str, prompts: List[str],
                    warmup_shapes: List[Tuple[int, int,
                                              int]], tensor_parallel_size: int,
                    backend: str, vllm_results: List[Dict[str, Any]],
                    hf_results: List[Dict[str, Any]]):

    print(f"\nmodel:         {model:s}")
    print(f"warmup shapes: {warmup_shapes}")
    print(f"tp size:       {tensor_parallel_size}")
    print(f"backend:       {backend:s}")
    print(f"\n#prompts:      {len(prompts):d}")
    print(f"#HF results:   {len(hf_results):d}"
          f"{'' if len(hf_results) == len(prompts) else '  ERROR':s}")
    print(f"#vLLM results: {len(vllm_results):d}"
          f"{'' if len(vllm_results) == len(prompts) else '  ERROR':s}")
    print()

    assert DISABLE_ASSERTS or len(hf_results) == len(vllm_results)
    assert DISABLE_ASSERTS or len(hf_results) == len(prompts)

    for prompt_index, (prompt, hf_result, vllm_result) in enumerate(
            zip(prompts, hf_results, vllm_results)):
        err_msg = '' if hf_result['text'] == vllm_result['text'] else '  ERROR'
        print(f"\nprompt {prompt_index:3d}:    {repr(prompt):s}")
        print("generated:")
        print(f"        HF:    {repr(hf_result['text']):s}")
        print(f"        vLLM:  {repr(vllm_result['text']):s}{err_msg}")
        print()

        assert DISABLE_ASSERTS or backend == 'sendnn_decoder' or\
            hf_result['token_ids'] == vllm_result['token_ids']

        if len(hf_result['tokens']) > 0:
            print("   token id. token               logprob      "
                  "   token id. token               logprob")

            logprob_abs_diff_list = []
            logprob_rel_diff_list = []

            for hf_token, hf_token_id, hf_logprob, vllm_token,\
                 vllm_token_id, vllm_logprob in zip(
                    hf_result['tokens'], hf_result['token_ids'],
                    hf_result['logprobs'], vllm_result['tokens'],
                    vllm_result['token_ids'], vllm_result['logprobs']):
                logprob_abs_diff = math.fabs(hf_logprob - vllm_logprob)
                logprob_abs_diff_list.append(logprob_abs_diff)
                logprob_rel_diff = math.fabs(logprob_abs_diff / hf_logprob)
                logprob_rel_diff_list.append(logprob_rel_diff)

                print(
                    f"HF: {hf_token_id:8d} {repr(hf_token):14s} "
                    f"{hf_logprob:14f}  "
                    f"vLLM: {vllm_token_id:8d} {repr(vllm_token):14s} "
                    f"{vllm_logprob:14f}  ",
                    end='')

                if backend == 'sendnn_decoder':
                    rel_tol = ISCLOSE_REL_TOL_SPYRE
                else:
                    rel_tol = ISCLOSE_REL_TOL_CPU

                if hf_token_id != vllm_token_id:  # different tokens
                    if backend == 'sendnn_decoder' and math.isclose(
                            hf_logprob, vllm_logprob, rel_tol=rel_tol):
                        # probably still OK
                        print('DIVERGING')
                        break
                    else:
                        print('ERROR')
                        assert DISABLE_ASSERTS or False
                        break
                else:  # identical tokens
                    if math.isclose(hf_logprob, vllm_logprob, rel_tol=rel_tol):
                        print()
                    else:
                        print('ERROR')
                        assert DISABLE_ASSERTS or False
                        break

            print()
            print("logprob absolute differences: "
                  f"average={np.mean(logprob_abs_diff_list):f}  "
                  f"maximum={np.max(logprob_abs_diff_list):f}")
            print("logprob relative differences: "
                  f"average={np.mean(logprob_rel_diff_list):f}  "
                  f"maximum={np.max(logprob_rel_diff_list):f}")

        print()


# vLLM / Spyre
def spyre_vllm_embeddings(model: str, prompts: List[str],
                          warmup_shapes: List[Tuple[int,
                                                    int]], max_model_len: int,
                          block_size: int, tensor_parallel_size: int,
                          backend: str) -> List[Dict[str, Any]]:

    warmup_prompt_length = [t[0] for t in warmup_shapes]
    warmup_new_tokens = [0] * len(warmup_shapes)
    warmup_batch_size = [t[1] for t in warmup_shapes]

    os.environ['VLLM_SPYRE_WARMUP_PROMPT_LENS'] = ','.join(
        str(val) for val in warmup_prompt_length)
    os.environ['VLLM_SPYRE_WARMUP_NEW_TOKENS'] = ','.join(
        str(val) for val in warmup_new_tokens)
    os.environ['VLLM_SPYRE_WARMUP_BATCH_SIZES'] = ','.join(
        str(val) for val in warmup_batch_size)
    os.environ['VLLM_SPYRE_DYNAMO_BACKEND'] = backend

    vllm_model = LLM(model=model,
                     tokenizer=model,
                     max_model_len=max_model_len,
                     block_size=block_size,
                     tensor_parallel_size=tensor_parallel_size)

    vllm_outputs = vllm_model.encode(prompts)

    results = []
    for req_output in vllm_outputs:
        result = {}
        result["embeddings"] = req_output.outputs.embedding
        results.append(result)

    return results


# Hugging Face
def st_embeddings(model: str, prompts: List[str]) -> List[Dict[str, Any]]:

    model = SentenceTransformer(model)

    results = []
    for prompt in prompts:
        embeddings = model.encode(prompt)

        # return ST generated embeddings
        result = {}
        result['embeddings'] = embeddings
        results.append(result)

    return results


# compare results
def compare_embedding_results(model: str, prompts: List[str],
                              warmup_shapes: List[Tuple[int, int]],
                              tensor_parallel_size: int, backend: str,
                              vllm_results: List[Dict[str, Any]],
                              hf_results: List[Dict[str, Any]]):

    print(f"\nmodel:         {model:s}")
    print(f"warmup shapes: {warmup_shapes}")
    print(f"tp size:       {tensor_parallel_size}")
    print(f"backend:       {backend:s}")
    print(f"\n#prompts:      {len(prompts):d}")
    print(f"#HF results:   {len(hf_results):d}"
          f"{'' if len(hf_results) == len(prompts) else '  ERROR':s}")
    print(f"#vLLM results: {len(vllm_results):d}"
          f"{'' if len(vllm_results) == len(prompts) else '  ERROR':s}")
    print()

    assert DISABLE_ASSERTS or len(hf_results) == len(vllm_results)
    assert DISABLE_ASSERTS or len(hf_results) == len(prompts)

    for hf_result, vllm_result in zip(hf_results, vllm_results):

        sim = util.pytorch_cos_sim(hf_result["embeddings"],
                                   vllm_result["embeddings"])

        assert math.isclose(sim, 1.0, rel_tol=0.05)


# get model directory path from env, if not set then default to "/models".
def get_spyre_model_dir_path():
    model_dir_path = os.environ.get("VLLM_SPYRE_TEST_MODEL_DIR", "/models")
    return model_dir_path


# get model backend from env, if not set then default to "eager"
# For multiple values:
# export SPYRE_TEST_BACKEND_LIST="eager, inductor, sendnn_decoder"
def get_spyre_backend_list():
    test_backend_list = []
    user_backend_list = os.environ.get("VLLM_SPYRE_TEST_BACKEND_LIST",
                                       "eager,inductor,sendnn_decoder,sendnn")

    for sypre_backend in user_backend_list.split(","):
        test_backend_list.append(sypre_backend.strip())
    return test_backend_list


# get model names from env, if not set then default to "llama-194m"
# For multiple values:
# export SPYRE_TEST_MODEL_LIST="llama-194m,all-roberta-large-v1"
def get_spyre_model_list(isEmbeddings=False):
    spyre_model_dir_path = get_spyre_model_dir_path()
    test_model_list = []
    user_test_model_list = os.environ.get("VLLM_SPYRE_TEST_MODEL_LIST",
                                          "llama-194m")

    # set default to bert if testing embeddings
    if isEmbeddings:
        user_test_model_list = os.environ.get("VLLM_SPYRE_TEST_MODEL_LIST",
                                              "all-roberta-large-v1")

    for model in user_test_model_list.split(","):
        test_model_list.append(f"{spyre_model_dir_path}/{model.strip()}")
    return test_model_list

#!/usr/bin/env python3
"""Download a model from HuggingFace with revision.

> python3 tools/download_model.py -m <HF-model-id> [-r <git-tag-or-hash>]

"""

import argparse
import logging


def download_granite_or_llama(hf_model_id: str, revision: str = "main"):
    from transformers import pipeline
    pipeline('text-generation', model=hf_model_id, revision=revision)


def download_roberta(hf_model_id: str, revision: str = "main"):
    from sentence_transformers import SentenceTransformer
    SentenceTransformer(hf_model_id, revision=revision)


download_methods = {
    "ibm-ai-platform/micro-g3.3-8b-instruct-1b": download_granite_or_llama,
    "ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8": download_granite_or_llama,
    "JackFram/llama-160m": download_granite_or_llama,
    "cross-encoder/stsb-roberta-large": download_roberta,
    "sentence-transformers/all-roberta-large-v1": download_roberta,
}


def download_model_with_revision(hf_model_id: str, revision: str = "main"):
    if hf_model_id in download_methods:
        download_method = download_methods.get(hf_model_id)
        logging.info("Downloading model '%s' with revision '%s' ...",
                     hf_model_id, revision)
        download_method(hf_model_id, revision)
        logging.info("Model '%s' with revision '%s' downloaded.", hf_model_id,
                     revision)
    else:
        logging.error(
            "No `download_method` found for model '%s'."
            " Supported models: %s", hf_model_id,
            str(list(download_methods.keys())))
        exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        dest='hf_model_id',
                        help='HuggingFace model ID.')
    parser.add_argument('-r',
                        dest='revision',
                        default="main",
                        help='Git tag, hash, or branch.')

    args, _extra_args = parser.parse_known_args()

    if args.hf_model_id:
        download_model_with_revision(args.hf_model_id, args.revision)
    else:
        logging.error("Need to specify a model ID with -model.")
        exit(1)


if __name__ == '__main__':
    main()

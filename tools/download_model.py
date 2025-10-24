#!/usr/bin/env python3
"""Download a model from HuggingFace with revision.

> python3 tools/download_model.py -m <HF-model-id> [-r <git-tag-or-hash>]

"""

import argparse
import logging


def download_granite_or_llama(model: str, revision: str = "main"):
    from transformers import pipeline

    pipeline("text-generation", model=model, revision=revision)


def download_roberta(model: str, revision: str = "main"):
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(model, revision=revision)


download_methods = {
    "ibm-ai-platform/micro-g3.3-8b-instruct-1b": download_granite_or_llama,
    "ibm-ai-platform/micro-g3.3-8b-instruct-1b-FP8": download_granite_or_llama,
    "JackFram/llama-160m": download_granite_or_llama,
    "cross-encoder/stsb-roberta-large": download_roberta,
    "sentence-transformers/all-roberta-large-v1": download_roberta,
}


def download_model_with_revision(model: str, revision: str = "main"):
    if model in download_methods:
        download_method = download_methods.get(model)
        logging.info("Downloading model '%s' with revision '%s' ...", model, revision)
        download_method(model, revision)
        logging.info("Model '%s' with revision '%s' downloaded.", model, revision)
    else:
        logging.error(
            "No `download_method` found for model '%s'. Supported models: %s",
            model,
            str(list(download_methods.keys())),
        )
        exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model", help="HuggingFace model ID")
    parser.add_argument(
        "-r", dest="revision", default="main", help="Git hash, tag, or branch (default='main')"
    )
    args, _extra_args = parser.parse_known_args()

    if args.model:
        download_model_with_revision(args.model, args.revision)
    else:
        logging.error("Need to provide a HuggingFace model ID.")
        exit(1)


if __name__ == "__main__":
    main()

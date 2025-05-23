#!/bin/bash

# Env vars to set for CPU-only testing

# Need to be set for tests to run
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# Run on CPU
export VLLM_SPYRE_DYNAMO_BACKEND=eager

# TODO: Tests don't work on CPU with MP enabled?
export VLLM_ENABLE_V1_MULTIPROCESSING=0

# Test related
export VLLM_SPYRE_TEST_BACKEND_LIST=eager
# Note: Make sure model name aligns with the model that you downloaded 
export VLLM_SPYRE_TEST_MODEL_LIST="JackFram/llama-160m"
export VLLM_SPYRE_TEST_MODEL_DIR=""
# We have to use `HF_HUB_OFFLINE=1` otherwise vllm tries to download a
# different version of the model using HF API which does not work locally
export HF_HUB_OFFLINE=1

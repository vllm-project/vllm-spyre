#!/bin/bash -e

# AIU configuration needs to be done at runtime, after the pod is deployed:
source /etc/profile.d/ibm-aiu-setup.sh

# Run `vllm serve` and passthrough all args
# NB: `exec` is used here to replace the parent process so that signals are
# handled correctly
exec vllm serve "$@"

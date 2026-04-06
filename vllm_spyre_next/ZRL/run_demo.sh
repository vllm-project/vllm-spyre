#!/bin/bash
# Wrapper for Offline_demo.py that suppresses [Gloo] stdout messages.
#
# Gloo prints connectivity info via raw std::cout, bypassing
# GLOO_LOG_LEVEL entirely.  We filter stdout line-by-line while
# leaving stderr completely untouched.
#
# Usage:  bash run_demo.sh [args passed to Offline_demo.py]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec > >(grep --line-buffered -v '^\[Gloo\]')

VLLM_LOGGING_LEVEL=ERROR \
GLOO_LOG_LEVEL=ERROR \
DTCOMPILER_KEEP_EXPORT=true \
SENLIB_DEVEL_CONFIG_FILE=/home/boh/dt-inductor2/.venv/etc/senlib_config_aiusmi.json \
FLEX_DATA_TRANSFER_BUFFER_COUNT=1 \
python "$SCRIPT_DIR/Offline_demo.py" "$@"

#!/bin/bash -e

# This script sets up the runtime environment for launching a vLLM API server using Spyre AIU cards.
# Used for local development and testing in tandem with podman run command.
# Not for use on an openshift cluster.
# 1. Validates Torch sendNN cache settings.
# 2. Detects and configures available AIU devices.
# 3. Activates the Python virtual environment if not already active.
# 4. Launches the vLLM server with the computed arguments.

if [ "$TORCH_SENDNN_CACHE_ENABLE" = "1" ]; then
    if [ -z "$TORCH_SENDNN_CACHE_DIR" ]; then
        echo "Error: TORCH_SENDNN_CACHE_DIR is not set."
        exit 1
    fi

    if [ ! -d "$TORCH_SENDNN_CACHE_DIR" ]; then
        echo "Error: Cache directory $TORCH_SENDNN_CACHE_DIR does not exist."
        exit 1
    fi

    PERMS=$(stat -c "%a" "$TORCH_SENDNN_CACHE_DIR")
    if [ "$PERMS" != "777" ]; then
        echo "Error: Cache directory $TORCH_SENDNN_CACHE_DIR does not have 777 permissions. Current: $PERMS"
        exit 1
    fi
fi

# Ensure that a model path has been set
if [ -z "${VLLM_MODEL_PATH}" ]; then
    echo "VLLM_MODEL_PATH is not set. Exiting."
    exit 1
fi

# Configure AIU cards, use all if none are specified.
if [ -z "${VLLM_AIU_PCIE_IDS}" ] ; then
	export VLLM_AIU_PCIE_IDS=$(lspci -n -d 1014:06a7 | cut -d ' ' -f 1)
fi

# Create a senlib_config.json to use only specified AIU id's.
tmpfile=$(mktemp -t senlib_config_XXXXXXX.json)
cat <<EOF | jq --argjson newValues "$(for i in ${VLLM_AIU_PCIE_IDS}; do echo "$i"; done | jq -R . | jq -s .)" '.GENERAL.sen_bus_id = $newValues' > "$tmpfile"
{
  "GENERAL": {
    "target": "SOC",
    "sen_bus_id": [
    ]
  },
  "METRICS": {
    "general": {
      "enable": false
    }
  }
}
EOF
sudo mv "$tmpfile" /etc/aiu/senlib_config.json

# Reconfigure AIU's, and update the envvars to reflect changes
. /etc/bashrc-sentient-env.sh
setup_multi_aiu_env

# Activate the vllm venv if not already active
if [[ -z "$VIRTUAL_ENV" ]] ; then
    source /opt/vllm/bin/activate
fi

# Start the vllm server with given model and detected parallelism.
DEFAULT_ARGS="--model ${VLLM_MODEL_PATH} -tp ${AIU_WORLD_SIZE}"
exec python -m vllm.entrypoints.openai.api_server "${DEFAULT_ARGS}" "$@"


#!/bin/bash

# A bash script that uses `/opt/sentient/senlib/bin/senlib_unit_test` 
# to check each AIU allocated to the pod to see if 
# they work for a basic test:

cleanup_done=0
cleanup() {
  if [ "$cleanup_done" -eq 0 ] && [ -f ~/.senlib.json.bak ]; then
    echo "Restoring .senlib.json from backup"
    cp ~/.senlib.json.bak ~/.senlib.json
    cleanup_done=1
  fi
  kill -- -$PPID
  wait
  exit
}

trap cleanup EXIT SIGINT

# Create backup .senlib.json if it doesn't exist
if [ -f "$HOME"/.senlib.json ]; then
  if [ ! -f "$HOME"/.senlib.json.bak ]; then
    echo "Creating backup of $HOME/.senlib.json"
    cp "$HOME"/.senlib.json "$HOME"/.senlib.json.bak
  else
    echo "$HOME/.senlib.json.bak already exists"
  fi
fi

for device_id in $(jq -r .GENERAL.sen_bus_id[] /etc/aiu/senlib_config.json); do
  echo "======================================================================"
  echo "Checking AIU ${device_id}"
  echo "======================================================================"
  jq -n '{"GENERAL": { "sen_bus_id": "'"${device_id}"'" }}' > .senlib.json
  # run in background to not override bash signal handler
  timeout 10 /opt/sentient/senlib/bin/senlib_unit_test --gtest_filter=SmlPF1VF0.Open &
  wait
done

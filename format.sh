#!/usr/bin/env bash
# This runs the precommit checks that are configured in the repo, reformatting files and failing on linting errors that require fixing.
# We use prek which is a much faster rust port of precommit from astral

# Cause the script to exit if a single command fails
set -eo pipefail

if [ -z "$*" ]; then
  # Run all files by default if no args are passed to format.sh
  args=( "--all-files" )
else
  # Otherwise passthrough all args
  args=( "$@" )
fi

# uvx will run prek from an isolated venv without requiring any dev setup or configuration
uvx prek "${args[@]}"

if ! git diff --quiet &>/dev/null; then
    echo 
    echo "🔍🔍There are files changed by the format checker or by you that are not added and committed:"
    git --no-pager diff --name-only
    echo "🔍🔍Format checker passed, but please add, commit and push all the files above to include changes made by the format checker."

    exit 1
else
    echo "✨🎉 Format check passed! Congratulations! 🎉✨"
fi
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if (($# > 0)); then
  provider=$1
  if [[ -f "$provider" ]]; then
    provider_dir="$(cd -- "$(dirname -- "$provider")" && pwd)"
    provider="$provider_dir/$(basename -- "$provider")"
  fi
  export AUTORD_DATASET_NOTEBOOK="$provider"
fi

cd "$REPO_ROOT/nb"
exec jupyter notebook Eval.ipynb

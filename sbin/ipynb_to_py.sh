#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if (($# == 0)); then
  set -- "$REPO_ROOT/nb/Rules.ipynb"
fi

for notebook in "$@"; do
  jupyter nbconvert --to python "$notebook"
done

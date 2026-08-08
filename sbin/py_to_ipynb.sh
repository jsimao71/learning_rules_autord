#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYFILE=${1:-"$REPO_ROOT/nb/Rules.py"}
OUT=${2:-"$REPO_ROOT/nb/Rules.ipynb"}

python - "$PYFILE" "$OUT" <<'PY'
import pathlib
import sys
import nbformat as nbf

source_path, output_path = map(pathlib.Path, sys.argv[1:3])
code = source_path.read_text(encoding="utf-8")
notebook = nbf.v4.new_notebook()
notebook.cells = [
    nbf.v4.new_markdown_cell(f"# Rules - generated from {source_path.name}"),
    nbf.v4.new_code_cell(code),
]
nbf.write(notebook, output_path)
PY

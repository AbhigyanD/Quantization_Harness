#!/usr/bin/env bash
# vLLM serving benchmark in its own venv (vLLM pins its own torch).
# Usage: scripts/run_bench.sh [config]
set -eu
CFG="${1:-configs/default.yaml}"
VENV="${VLLM_VENV:-.venv-vllm}"
if [ ! -x "$VENV/bin/python" ]; then
  python -m venv "$VENV"
  "$VENV/bin/pip" install -q -r requirements-vllm.txt
fi
"$VENV/bin/python" -m bench.vllm_bench --config "$CFG"
"$VENV/bin/python" -m bench.summarize --config "$CFG"

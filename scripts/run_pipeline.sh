#!/usr/bin/env bash
# Quantize, measure, evaluate. Run on a GPU box with requirements.txt + requirements-quant.txt installed.
# Usage: scripts/run_pipeline.sh [config]   (default configs/default.yaml)
# A failing step is logged and the pipeline moves on, so one broken quantizer doesn't hide the
# other variants; eval/compare.py reports anything missing as INCOMPLETE.
set -u
CFG="${1:-configs/default.yaml}"
PY="${PYTHON:-python}"
cfgval() { $PY -c "import yaml,functools;d=yaml.safe_load(open('$CFG'));print(functools.reduce(lambda a,k:a[k],'$1'.split('.'),d))"; }
SLUG=$(cfgval model.slug); MODELS=$(cfgval paths.models_dir); RESULTS=$(cfgval paths.results_dir)
FAIL_LOG="$RESULTS/$SLUG/pipeline_failures.log"
mkdir -p "$RESULTS/$SLUG"
: > "$FAIL_LOG"
VARIANTS=$($PY -c "import yaml;print(' '.join(yaml.safe_load(open('$CFG'))['variants']))")

run() {  # run <description> <command...>
  desc="$1"; shift
  echo "=== $desc: $*"
  if ! "$@"; then
    echo "FAILED: $desc ($*)" | tee -a "$FAIL_LOG"
    return 1
  fi
}

run "download fp16 baseline" $PY -m quantize.fp16_baseline --config "$CFG" || exit 1
run "quantize nf4"  $PY -m quantize.quantize_nf4  --config "$CFG"
run "quantize gptq" $PY -m quantize.quantize_gptq --config "$CFG"

for v in $VARIANTS; do
  [ -d "$MODELS/$SLUG/$v" ] || { echo "skipping $v: no checkpoint" | tee -a "$FAIL_LOG"; continue; }
  run "footprint $v"  $PY -m quantize.measure_footprint --config "$CFG" --variant "$v"
  run "perplexity $v" $PY -m eval.perplexity --config "$CFG" --variant "$v"
  run "lm_eval $v"    $PY -m eval.run_lm_eval --config "$CFG" --variant "$v"
done

$PY -m eval.compare --config "$CFG"
echo "--- failures:"; cat "$FAIL_LOG"

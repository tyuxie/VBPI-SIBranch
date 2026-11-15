#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
TREES_DIR="$SCRIPT_DIR/L8trees"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATE_TAG="${RUN_DATE:-$(date +%Y%m%d%H%M%S)}"
MAX_ITER="${MAX_ITER:-40000}"
N_SAMPLES="${N_SAMPLES:-1000}"
MAX_DATASETS="${MAX_DATASETS:-0}"

cd "$ROOT"

find "$TREES_DIR" -type f -name 'dna.fasta' | sort | while read -r fasta; do
  rel="${fasta#$TREES_DIR/}"
  exp_name="$(dirname "$rel" | sed 's#/#__#g')"

  count="${count:-0}"
  count=$((count + 1))
  if [ "$MAX_DATASETS" -gt 0 ] && [ "$count" -gt "$MAX_DATASETS" ]; then
    break
  fi

  echo "[submit] $rel -> $exp_name (dataset $count, $MAX_ITER iterations)"
  "$PYTHON_BIN" synthetic_exp/main_synthetic.py \
    --dataset L8ALL \
    --support_dataset L8ALL \
    --input_fasta "$fasta" \
    --experiment_name "$exp_name" \
    --brlen_model iwhvi \
    --gradMethod iwhvi \
    --date "$DATE_TAG" \
    --maxIter "$MAX_ITER" \
    --tf 1000 \
    --lbf 1000  \
    --sf 10000 \
    --hdim 50 \
    --zdim 25 \
    --nwarmStart 10000 \
    --n_samples "$N_SAMPLES"
done

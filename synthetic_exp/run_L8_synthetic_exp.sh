#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
TREES_DIR="$SCRIPT_DIR/L8trees"
PYTHON_BIN="python3"
DATE_TAG="$(date +%F)"

cd "$ROOT"

find "$TREES_DIR" -type f -name 'dna.fasta' | sort | while read -r fasta; do
  rel="${fasta#$TREES_DIR/}"
  exp_name="$(dirname "$rel" | sed 's#/#__#g')"

  echo "[submit] $rel -> $exp_name"
  "$PYTHON_BIN" synthetic_exp/main_synthetic.py \
    --dataset L8ALL \
    --support_dataset L8ALL \
    --input_fasta "$fasta" \
    --experiment_name "$exp_name" \
    --brlen_model iwhvi \
    --gradMethod iwhvi \
    --date "$DATE_TAG" \
    --maxIter 40000 \
    --tf 1000 \
    --lbf 1000  \
    --sf 10000 \
    --hdim 50 \
    --zdim 25 \
    --maxIter 40000 \
    --nwarmStart 10000
done

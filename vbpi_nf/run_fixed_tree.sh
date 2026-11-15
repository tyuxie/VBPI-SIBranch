#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/python_env.sh"
resolve_nf_python
cd "$ROOT"

args=(--dataset "${DATASET:-DS1}" --feature-dim "${NF_FEATURE_DIM:-2}"
  --hidden-sizes "${NF_HIDDEN_DIM:-100}" --flow-layers "${NF_LAYERS:-10}"
  --iterations "${MAX_ITER:-200000}" --batch-size "${BATCH_SIZE:-10}"
  --learning-rate "${NF_STEPSZ_BRANCH:-0.0001}"
  --log-every "${LOG_EVERY:-1000}" --save-every "${SAVE_EVERY:-100000}"
  --bound-runs "${BOUND_RUNS:-0}"
  --branch-samples "${BRANCH_SAMPLES:-0}"
  --run-tag "${RUN_TAG:-fixed_tree_nf}"
  --workdir "${OUTPUT_ROOT:-$ROOT/results/vbpi_nf/fixed_tree}")
if [ -n "${TREE_IDS:-}" ]; then
  read -r -a tree_ids <<< "$TREE_IDS"
  args+=(--tree-ids "${tree_ids[@]}")
else
  args+=(--all-trees)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/standalone/fixed_tree.py" "${args[@]}"
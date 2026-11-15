#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/python_env.sh"
resolve_nf_python
RUN_DATE="${RUN_DATE:?Set RUN_DATE to the benchmark run date}"
RUN_TAG="${RUN_TAG:-figure5_6_nf}"
WORKDIR="${WORKDIR:-results/vbpi_nf}"
FEATURE_DIM="${NF_FEATURE_DIM:-2}"
LAYERS="${NF_LAYERS:-10}"
HIDDEN_DIM="${NF_HIDDEN_DIM:-100}"
SEED="${SEED:-2023}"
cd "$ROOT"
checkpoint="$WORKDIR/DS1/realnvp_${LAYERS}/${RUN_TAG}_${RUN_DATE}_${SEED}/final.pt"
test -f "$checkpoint" || { echo "Missing VBPI-NF checkpoint: $checkpoint" >&2; exit 2; }
for tree_id in ${TREE_IDS:-1 2 36 37}; do
  "$PYTHON_BIN" "$SCRIPT_DIR/standalone/sample_branches.py" \
    --dataset DS1 --tree-id "$tree_id" --checkpoint "$checkpoint" \
    --feature-dim "$FEATURE_DIM" --hidden-sizes "$HIDDEN_DIM" --flow-layers "$LAYERS" \
    --n-samples "${N_SAMPLES:-100000}" \
    --output "$ROOT/results/vbpi_nf/figure5_6/DS1_tree${tree_id}_branches.npy"
done
#!/usr/bin/env bash
# Reproduction helper for the VBPI-NF paper workflows.
#   bash vbpi_nf/reproduce.sh <recipe>
# Recipes and their knobs are documented in experiments/REPRODUCE_RECIPE.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d%H%M%S)}"

usage() {
  cat <<'EOF'
Usage: bash vbpi_nf/reproduce.sh <recipe>
Recipes: train, mll_benchmark, training_performance, joint_branch_ds8, joint_branch_ds1, fixed_tree
See experiments/REPRODUCE_RECIPE.md for commands, knobs, and outputs.
EOF
  exit 64
}

# RealNVP benchmark training (+ optional raw evaluation) over DATASETS x REPLICATES.
_benchmark() {
  for dataset in $DATASETS; do
    test -f "data/hohna_datasets_fasta/${dataset}.fasta" || {
      echo "Missing alignment data/hohna_datasets_fasta/${dataset}.fasta" >&2
      exit 2
    }
    for replicate in {1..10}; do
      support_file="data/ufboot_data_DS1-8/${dataset}/${dataset}_ufboot_rep_${replicate}"
      test -f "$support_file" || {
        echo "Missing UFBoot file $support_file; download the full data package first." >&2
        exit 2
      }
    done

    for replicate in $(seq 1 "${REPLICATES:-1}"); do
      tag="$RUN_TAG"
      if [ "${REPLICATES:-1}" -gt 1 ]; then tag="${RUN_TAG}_rep${replicate}"; fi
      seed=$(( ${SEED:-2023} + replicate - 1 ))
      args=(--dataset "$dataset" --flow_type realnvp
        --date "$RUN_DATE" --run-tag "$tag" --workdir "$ROOT/${WORKDIR:-results/vbpi_nf}" --seed "$seed"
        --maxIter "${MAX_ITER:-400000}" --stepszBranch "${NF_STEPSZ_BRANCH:-0.0001}"
        --eval-repeats "${EVAL_REPEATS:-100}" --eval-runs "${EVAL_RUNS:-10000}"
        --eval-ml-repeats "${EVAL_ML_REPEATS:-1000}" --empFreqTrees "${EMPFREQ_TREES:-42}"
        --nf "${NF_FEATURE_DIM:-2}" --sh "${NF_HIDDEN_DIM:-100}" --Lnf "${NF_LAYERS:-10}")
      if [ "${EMP_FREQ:-0}" = 1 ]; then args+=(--empFreq); fi
      echo "[VBPI-NF train] dataset=$dataset run=$RUN_DATE tag=$tag seed=$seed"
      "$PYTHON_BIN" "$HERE/vbpi_nf_src/main.py" "${args[@]}"
      if [ "${DO_EVALUATION:-1}" = 1 ]; then
        echo "[VBPI-NF raw evaluation] dataset=$dataset run=$RUN_DATE tag=$tag"
        "$PYTHON_BIN" "$HERE/vbpi_nf_src/main.py" "${args[@]}" --test
      fi
    done
  done
}

# Shared conditional branch sampling from a trained RealNVP checkpoint.
_samples() {
  local dataset="$1" checkpoint="$2" outdir="$3"
  for tree_id in ${TREE_IDS:?set TREE_IDS}; do
    "$PYTHON_BIN" "$HERE/vbpi_nf_src/sample_branches.py" \
      --dataset "$dataset" --tree-id "$tree_id" --checkpoint "$checkpoint" \
      --feature-dim "${NF_FEATURE_DIM:-2}" --hidden-sizes "${NF_HIDDEN_DIM:-100}" \
      --flow-layers "${NF_LAYERS:-10}" --n-samples "${N_SAMPLES:-100000}" \
      --output "$ROOT/results/vbpi_nf/$outdir/${dataset}_tree${tree_id}_branches.npy"
  done
}

case "${1:-}" in
# -------------------------------------------------------------- benchmark
train)
  DATASETS="${DATASETS:-DS1}" RUN_TAG="${RUN_TAG:-vbpi_nf}"
  _benchmark
  ;;
# ----------------------------------------------------------------- mll_benchmark
mll_benchmark)
  DATASETS="${DATASETS:-DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8}"
  RUN_TAG="${RUN_TAG:-mll_benchmark_nf}" DO_EVALUATION=1
  _benchmark
  ;;
# ---------------------------------------------------------------- training_performance
training_performance)
  DATASETS="${DATASETS:-DS1}" RUN_TAG="${RUN_TAG:-training_performance_nf}" DO_EVALUATION=0
  _benchmark
  ;;
# ------------------------------------------------------ joint_branch_ds8
joint_branch_ds8)
  checkpoint="${WORKDIR:-results/vbpi_nf}/DS8/realnvp_${NF_LAYERS:-10}/${RUN_TAG:-joint_branch_ds8_nf}_${RUN_DATE}_${SEED:-2023}/final.pt"
  test -f "$checkpoint" || { echo "Missing VBPI-NF checkpoint: $checkpoint" >&2; exit 2; }
  TREE_IDS="${TREE_IDS:-1 2}" _samples DS8 "$checkpoint" joint_branch_ds8
  ;;
# ---------------------------------------------------- joint_branch_ds1
joint_branch_ds1)
  checkpoint="${WORKDIR:-results/vbpi_nf}/DS1/realnvp_${NF_LAYERS:-10}/${RUN_TAG:-joint_branch_ds1_nf}_${RUN_DATE}_${SEED:-2023}/final.pt"
  test -f "$checkpoint" || { echo "Missing VBPI-NF checkpoint: $checkpoint" >&2; exit 2; }
  TREE_IDS="${TREE_IDS:-1 2 36 37}" _samples DS1 "$checkpoint" branch_dist
  ;;
# ------------------------------------------------------------- fixed_tree
fixed_tree)
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
  "$PYTHON_BIN" "$HERE/vbpi_nf_src/fixed_tree.py" "${args[@]}"
  ;;
# ------------------------------------------------------------------ help
""|help|-h|--help) usage ;;
*) echo "unknown recipe: ${1:-}" >&2; usage ;;
esac

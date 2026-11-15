#!/usr/bin/env bash
# Single entry point for the benchmark experiments (Tables 1-4, Figures 1-7).
#   bash experiments/run.sh <recipe>
# Recipes and their knobs are documented in experiments/RECIPES.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"
source "$ROOT/scripts/python_env.sh"
resolve_project_python

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d%H%M%S)}"
RUN_ID="${RUN_ID:-$RUN_DATE}"
DS18="${DATASETS:-DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8}"

usage() {
  sed -n '2,40p' "$HERE/RECIPES.md" | sed 's/^# /  /'
  exit 64
}

case "${1:-}" in
# ------------------------------------------------------------------ table1
table1)
  DATASETS="$DS18" MODELS="${MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG=table1 \
    WORKDIR="${WORKDIR:-results/table1}" DO_EVALUATION=1 \
    bash "$HERE/run_benchmark_matrix.sh"
  DATASETS="$DS18" RUN_DATE="$RUN_DATE" RUN_TAG=table1_nf \
    WORKDIR="${NF_WORKDIR:-results/table1_nf}" DO_EVALUATION=1 \
    bash "$ROOT/vbpi_nf/run_table1.sh"
  ;;
# ------------------------------------------------------------------ table3
table3)
  for layers in ${HL_GRID:-1 2 3 4}; do
    HL="$layers" DATASETS="${DATASETS:-DS1 DS8}" MODELS="${MODELS:-sivi iwhvi}" \
      RUN_DATE="$RUN_DATE" RUN_TAG="table3_l${layers}" \
      WORKDIR="${WORKDIR:-results/table3}" \
      bash "$HERE/run_benchmark_matrix.sh"
  done
  ;;
# ------------------------------------------------------------------ table4
table4)
  for activation in ${ACTIVATIONS:-elu relu}; do
    ACTIVATION="$activation" DATASETS="${DATASETS:-DS1 DS8}" \
      MODELS="${MODELS:-sivi iwhvi}" \
      RUN_DATE="$RUN_DATE" RUN_TAG="table4_${activation}" \
      WORKDIR="${WORKDIR:-results/table4}" \
      bash "$HERE/run_benchmark_matrix.sh"
  done
  ;;
# ----------------------------------------------------------------- figure1
figure1)
  DATASETS=DS8 MODELS="${MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG=figure1 \
    WORKDIR="${WORKDIR:-results/figure1}" DO_EVALUATION=0 \
    bash "$HERE/run_benchmark_matrix.sh"
  for model in ${MODELS:-gnn sivi iwhvi}; do
    if [ "$model" = gnn ]; then grad=vimco; else grad="$model"; fi
    checkpoint="$ROOT/${WORKDIR:-results/figure1}/DS8/$model/${grad}_10"
    if [ "$model" = gnn ]; then checkpoint="${checkpoint}_edge_sum"; fi
    if [ "$model" = sivi ]; then checkpoint="${checkpoint}_nz_50"; fi
    checkpoint="${checkpoint}_figure1_${RUN_DATE}/final.pt"
    for tree_id in 1 2; do
      "$PYTHON_BIN" "$HERE/sample_joint_branches.py" \
        --dataset DS8 --tree-id "$tree_id" --model "$model" \
        --checkpoint "$checkpoint" \
        --output "results/figure1/DS8_${model}_tree${tree_id}.npy"
    done
  done
  DATASETS=DS8 RUN_DATE="$RUN_DATE" RUN_TAG=figure1_nf \
    WORKDIR="${NF_WORKDIR:-results/figure1_nf}" DO_EVALUATION=0 \
    bash "$ROOT/vbpi_nf/run_benchmark.sh"
  RUN_DATE="$RUN_DATE" RUN_TAG=figure1_nf \
    WORKDIR="${NF_WORKDIR:-results/figure1_nf}" \
    bash "$ROOT/vbpi_nf/run_figure1_samples.sh"
  ;;
# ----------------------------------------------------------------- figure3
figure3)
  DATASETS="$DS18" MODELS="${MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG=figure3 REPLICATES="${REPLICATES:-100}" \
    WORKDIR="${WORKDIR:-results/figure3}" DO_EVALUATION=0 \
    bash "$HERE/run_benchmark_matrix.sh"
  DATASETS="$DS18" RUN_DATE="$RUN_DATE" RUN_TAG=figure3_nf \
    REPLICATES="${REPLICATES:-100}" \
    WORKDIR="${NF_WORKDIR:-results/figure3_nf}" DO_EVALUATION=0 \
    bash "$ROOT/vbpi_nf/run_figure3.sh"
  ;;
# ----------------------------------------------------------------- figure7
figure7)
  for nz in ${NZ_GRID:-20 50 100}; do
    NZ="$nz" DATASETS="$DS18" MODELS="${MODELS:-sivi iwhvi}" \
      RUN_DATE="$RUN_DATE" RUN_TAG="figure7_j${nz}" \
      WORKDIR="${WORKDIR:-results/figure7}" \
      bash "$HERE/run_benchmark_matrix.sh"
  done
  ;;
# ---------------------------------------------------------- fig4_table2
fig4_table2)
  RUN_TAG=figure4_table2
  WORKDIR="${WORKDIR:-results/figure4_table2}"
  DATASETS=DS1 MODELS="${JOINT_MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" \
    EMP_FREQ=1 DO_EVALUATION=1 \
    bash "$HERE/run_benchmark_matrix.sh"
  DATASETS=DS1 RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" \
    EMP_FREQ=1 DO_EVALUATION=1 \
    bash "$ROOT/vbpi_nf/run_benchmark.sh"

  tree_args=(--all-trees)
  if [ -n "${TREE_IDS:-}" ]; then tree_args=(--tree-ids $TREE_IDS); fi
  for model in ${MODELS:-lognorm sivi iwhvi}; do
    "$PYTHON_BIN" "$HERE/run_fixed_tree.py" --dataset DS1 --model "$model" \
      "${tree_args[@]}" --run-id "${RUN_ID}_figure4_table2_${model}" \
      --iterations "${MAX_ITER:-200000}" --bound-runs "${BOUND_RUNS:-10000}" \
      --z-batch-size "${NZ:-50}" --eval-z-batch-size "${EVAL_NZ:-1000}" \
      --output-root "${OUTPUT_ROOT:-results/fixed_tree}"
  done
  DATASET=DS1 TREE_IDS="${TREE_IDS:-}" RUN_TAG="${RUN_ID}_figure4_table2_nf" \
    MAX_ITER="${MAX_ITER:-200000}" BOUND_RUNS="${BOUND_RUNS:-10000}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/vbpi_nf/fixed_tree}" \
    bash "$ROOT/vbpi_nf/run_fixed_tree.sh"
  ;;
# ------------------------------------------------------------ figure5_6
figure5_6)
  RUN_TAG=figure5_6
  WORKDIR="${WORKDIR:-results/figure5_6}"
  if [ -z "${MRBAYES_TREE_ROOT:-}" ]; then
    echo "Set MRBAYES_TREE_ROOT to the directory containing tree1/tree1.nex.t through tree42/tree42.nex.t (see mrbayes/README.md) before starting the fixed-tree runs." >&2
    exit 2
  fi
  for tree_id in $(seq 1 42); do
    test -f "$MRBAYES_TREE_ROOT/tree${tree_id}/tree${tree_id}.nex.t" || {
      echo "Missing MrBayes tree file: $MRBAYES_TREE_ROOT/tree${tree_id}/tree${tree_id}.nex.t" >&2
      exit 2
    }
  done

  DATASETS=DS1 MODELS="${JOINT_MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" DO_EVALUATION=0 \
    bash "$HERE/run_benchmark_matrix.sh"
  DATASETS=DS1 RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" \
    DO_EVALUATION=0 bash "$ROOT/vbpi_nf/run_benchmark.sh"

  TREE_IDS="${FIGURE5_TREE_IDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42}"
  N_SAMPLES="${BRANCH_SAMPLES:-1000000}"
  for model in ${MODELS:-gnn sivi iwhvi}; do
    if [ "$model" = gnn ]; then grad=vimco; else grad="$model"; fi
    checkpoint="$WORKDIR/DS1/$model/${grad}_10"
    if [ "$model" = gnn ]; then checkpoint="${checkpoint}_edge_sum"; fi
    if [ "$model" = sivi ]; then checkpoint="${checkpoint}_nz_50"; fi
    checkpoint="${checkpoint}_${RUN_TAG}_${RUN_DATE}/final.pt"
    test -f "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
    for tree_id in $TREE_IDS; do
      "$PYTHON_BIN" "$HERE/sample_joint_branches.py" \
        --dataset DS1 --tree-id "$tree_id" --model "$model" \
        --checkpoint "$checkpoint" --n-samples "$N_SAMPLES" \
        --output "results/figure5_6/joint_${model}/tree_${tree_id}_branches.npy"
    done
  done
  RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" \
    TREE_IDS="$TREE_IDS" N_SAMPLES="$N_SAMPLES" \
    bash "$ROOT/vbpi_nf/run_figure5_6_samples.sh"

  tree_args=(--all-trees)
  if [ -n "${TREE_IDS:-}" ]; then tree_args=(--tree-ids $TREE_IDS); fi
  for model in ${MODELS:-lognorm sivi iwhvi}; do
    "$PYTHON_BIN" "$HERE/run_fixed_tree.py" --dataset DS1 --model "$model" \
      "${tree_args[@]}" --run-id "${RUN_ID}_figure5_6_${model}" \
      --iterations "${MAX_ITER:-200000}" --branch-samples "${BRANCH_SAMPLES:-1000000}" \
      --output-root "${OUTPUT_ROOT:-results/fixed_tree}"
  done
  DATASET=DS1 TREE_IDS="${TREE_IDS:-}" RUN_TAG="${RUN_ID}_figure5_6_nf" \
    MAX_ITER="${MAX_ITER:-200000}" BRANCH_SAMPLES="${BRANCH_SAMPLES:-1000000}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/vbpi_nf/fixed_tree}" \
    bash "$ROOT/vbpi_nf/run_fixed_tree.sh"

  for tree_id in $(seq 1 42); do
    "$PYTHON_BIN" "$HERE/prepare_mrbayes_branch_samples.py" \
      --tree-id "$tree_id" \
      --nexus-tree-file "$MRBAYES_TREE_ROOT/tree${tree_id}/tree${tree_id}.nex.t" \
      --output "${REFERENCE_ROOT:-data/fixed_tree_mcmc/DS1}/tree${tree_id}.npy" \
      --burnin "${BURNIN:-2500}"
  done
  ;;
# ------------------------------------------------------------------ help
""|help|-h|--help) usage ;;
*) echo "unknown recipe: ${1:-}" >&2; usage ;;
esac

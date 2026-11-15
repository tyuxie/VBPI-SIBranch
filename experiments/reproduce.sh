#!/usr/bin/env bash
# Reproduction helper for the paper experiments (Tables 1-4, Figures 1-7).
#   bash experiments/reproduce.sh <recipe>
# Recipes and their knobs are documented in experiments/REPRODUCE_RECIPE.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python}"

RUN_DATE="${RUN_DATE:-$(date +%Y%m%d%H%M%S)}"
RUN_ID="${RUN_ID:-$RUN_DATE}"
DS18="${DATASETS:-DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8}"

usage() {
  sed -n '2,40p' "$HERE/REPRODUCE_RECIPE.md" | sed 's/^# /  /'
  exit 64
}

# Final checkpoint written by run_benchmark_matrix.sh.
checkpoint_path() { # dataset model workdir tag date
  local dataset=$1 model=$2 workdir=$3 tag=$4 date=$5 grad=$2 suffix= nz=${NZ:-50}
  if [ "$model" = gnn ]; then grad=vimco; suffix=_edge_sum; fi
  if [ "$model" = sivi ]; then suffix="_nz_${nz}"; fi
  printf '%s/%s/%s/%s_10%s_%s_%s/final.pt\n' \
    "$workdir" "$dataset" "$model" "$grad" "$suffix" "$tag" "$date"
}

case "${1:-}" in
# ------------------------------------------------------------------ mll_benchmark
mll_benchmark)
  DATASETS="$DS18" MODELS="${MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG=mll_benchmark \
    WORKDIR="${WORKDIR:-results/mll_benchmark}" DO_EVALUATION=1 \
    bash "$HERE/run_benchmark_matrix.sh"
  DATASETS="$DS18" RUN_DATE="$RUN_DATE" RUN_TAG=mll_benchmark_nf \
    WORKDIR="${NF_WORKDIR:-results/mll_benchmark_nf}" DO_EVALUATION=1 \
    bash "$ROOT/vbpi_nf/reproduce.sh" mll_benchmark
  ;;
# ------------------------------------------------------------------ mp_rounds_ablation
mp_rounds_ablation)
  for layers in ${HL_GRID:-1 2 3 4}; do
    HL="$layers" DATASETS="${DATASETS:-DS1 DS8}" MODELS="${MODELS:-sivi iwhvi}" \
      RUN_DATE="$RUN_DATE" RUN_TAG="mp_rounds_ablation_l${layers}" \
      WORKDIR="${WORKDIR:-results/mp_rounds_ablation}" \
      bash "$HERE/run_benchmark_matrix.sh"
  done
  ;;
# ------------------------------------------------------------------ activation_ablation
activation_ablation)
  for activation in ${ACTIVATIONS:-elu relu}; do
    ACTIVATION="$activation" DATASETS="${DATASETS:-DS1 DS8}" \
      MODELS="${MODELS:-sivi iwhvi}" \
      RUN_DATE="$RUN_DATE" RUN_TAG="activation_ablation_${activation}" \
      WORKDIR="${WORKDIR:-results/activation_ablation}" \
      bash "$HERE/run_benchmark_matrix.sh"
  done
  ;;
# ----------------------------------------------------------------- joint_branch_ds8
joint_branch_ds8)
  WORKDIR="${WORKDIR:-results/joint_branch_ds8}"
  DATASETS=DS8 MODELS="${MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG=joint_branch_ds8 \
    WORKDIR="$WORKDIR" DO_EVALUATION=0 \
    bash "$HERE/run_benchmark_matrix.sh"
  for model in ${MODELS:-gnn sivi iwhvi}; do
    checkpoint=$(checkpoint_path DS8 "$model" "$WORKDIR" joint_branch_ds8 "$RUN_DATE")
    for tree_id in 1 2; do
      "$PYTHON_BIN" "$HERE/sample_joint_branches.py" \
        --dataset DS8 --tree-id "$tree_id" --model "$model" \
        --checkpoint "$checkpoint" \
        --output "$WORKDIR/DS8_${model}_tree${tree_id}.npy"
    done
  done
  DATASETS=DS8 RUN_DATE="$RUN_DATE" RUN_TAG=joint_branch_ds8_nf \
    WORKDIR="${NF_WORKDIR:-results/joint_branch_ds8_nf}" DO_EVALUATION=0 \
    bash "$ROOT/vbpi_nf/reproduce.sh" train
  RUN_DATE="$RUN_DATE" RUN_TAG=joint_branch_ds8_nf \
    WORKDIR="${NF_WORKDIR:-results/joint_branch_ds8_nf}" \
    bash "$ROOT/vbpi_nf/reproduce.sh" joint_branch_ds8
  ;;
# ----------------------------------------------------------------- training_performance
training_performance)
  DATASETS="$DS18" MODELS="${MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG=training_performance REPLICATES="${REPLICATES:-10}" \
    WORKDIR="${WORKDIR:-results/training_performance}" DO_EVALUATION=0 \
    bash "$HERE/run_benchmark_matrix.sh"
  DATASETS="$DS18" RUN_DATE="$RUN_DATE" RUN_TAG=training_performance_nf \
    REPLICATES="${REPLICATES:-10}" \
    WORKDIR="${NF_WORKDIR:-results/training_performance_nf}" DO_EVALUATION=0 \
    bash "$ROOT/vbpi_nf/reproduce.sh" training_performance
  ;;
# ----------------------------------------------------------------- nz_ablation
nz_ablation)
  for nz in ${NZ_GRID:-20 50 100}; do
    NZ="$nz" DATASETS="$DS18" MODELS="${MODELS:-sivi iwhvi}" \
      RUN_DATE="$RUN_DATE" RUN_TAG="nz_ablation_j${nz}" \
      WORKDIR="${WORKDIR:-results/nz_ablation}" \
      bash "$HERE/run_benchmark_matrix.sh"
  done
  ;;
# ---------------------------------------------------------- fixedtree_ds1
fixedtree_ds1)
  RUN_TAG=fixedtree_ds1
  WORKDIR="${WORKDIR:-results/fixedtree_ds1}"
  DATASETS=DS1 MODELS="${JOINT_MODELS:-gnn sivi iwhvi}" \
    RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" \
    EMP_FREQ=1 DO_EVALUATION=1 \
    bash "$HERE/run_benchmark_matrix.sh"
  DATASETS=DS1 RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" \
    EMP_FREQ=1 DO_EVALUATION=1 \
    bash "$ROOT/vbpi_nf/reproduce.sh" train

  tree_args=(--all-trees)
  if [ -n "${TREE_IDS:-}" ]; then tree_args=(--tree-ids $TREE_IDS); fi
  for model in ${MODELS:-lognorm sivi iwhvi}; do
    "$PYTHON_BIN" "$HERE/run_fixed_tree.py" --dataset DS1 --model "$model" \
      "${tree_args[@]}" --run-id "${RUN_ID}_fixedtree_ds1_${model}" \
      --iterations "${MAX_ITER:-200000}" --bound-runs "${BOUND_RUNS:-10000}" \
      --z-batch-size "${NZ:-50}" --eval-z-batch-size "${EVAL_NZ:-1000}" \
      --output-root "${OUTPUT_ROOT:-results/fixed_tree}"
  done
  DATASET=DS1 TREE_IDS="${TREE_IDS:-}" RUN_TAG="${RUN_ID}_fixedtree_ds1_nf" \
    MAX_ITER="${MAX_ITER:-200000}" BOUND_RUNS="${BOUND_RUNS:-10000}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/vbpi_nf/fixed_tree}" \
    bash "$ROOT/vbpi_nf/reproduce.sh" fixed_tree
  ;;
# ------------------------------------------------------------ branch_dist
branch_dist)
  RUN_TAG=branch_dist
  WORKDIR="${WORKDIR:-results/branch_dist}"
  if [ -z "${MRBAYES_TREE_ROOT:-}" ]; then
    echo "Set MRBAYES_TREE_ROOT to the directory containing tree1/tree1.nex.t through tree42/tree42.nex.t (see experiments/REPRODUCE_RECIPE.md) before starting the fixed-tree runs." >&2
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
    DO_EVALUATION=0 bash "$ROOT/vbpi_nf/reproduce.sh" train

  TREE_IDS="${FIGURE5_TREE_IDS:-1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42}"
  N_SAMPLES="${BRANCH_SAMPLES:-1000000}"
  for model in ${MODELS:-gnn sivi iwhvi}; do
    checkpoint=$(checkpoint_path DS1 "$model" "$WORKDIR" "$RUN_TAG" "$RUN_DATE")
    test -f "$checkpoint" || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
    for tree_id in $TREE_IDS; do
      "$PYTHON_BIN" "$HERE/sample_joint_branches.py" \
        --dataset DS1 --tree-id "$tree_id" --model "$model" \
        --checkpoint "$checkpoint" --n-samples "$N_SAMPLES" \
        --output "$WORKDIR/joint_${model}/tree_${tree_id}_branches.npy"
    done
  done
  RUN_DATE="$RUN_DATE" RUN_TAG="$RUN_TAG" WORKDIR="$WORKDIR" \
    TREE_IDS="$TREE_IDS" N_SAMPLES="$N_SAMPLES" \
    bash "$ROOT/vbpi_nf/reproduce.sh" joint_branch_ds1

  tree_args=(--all-trees)
  if [ -n "${TREE_IDS:-}" ]; then tree_args=(--tree-ids $TREE_IDS); fi
  for model in ${MODELS:-lognorm sivi iwhvi}; do
    "$PYTHON_BIN" "$HERE/run_fixed_tree.py" --dataset DS1 --model "$model" \
      "${tree_args[@]}" --run-id "${RUN_ID}_branch_dist_${model}" \
      --iterations "${MAX_ITER:-200000}" --branch-samples "${BRANCH_SAMPLES:-1000000}" \
      --output-root "${OUTPUT_ROOT:-results/fixed_tree}"
  done
  DATASET=DS1 TREE_IDS="${TREE_IDS:-}" RUN_TAG="${RUN_ID}_branch_dist_nf" \
    MAX_ITER="${MAX_ITER:-200000}" BRANCH_SAMPLES="${BRANCH_SAMPLES:-1000000}" \
    OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/results/vbpi_nf/fixed_tree}" \
    bash "$ROOT/vbpi_nf/reproduce.sh" fixed_tree

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

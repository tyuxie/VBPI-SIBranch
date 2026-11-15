#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
source "$ROOT/scripts/python_env.sh"
resolve_project_python
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d%H%M%S)}"
DATASETS="${DATASETS:-DS1}"
MODELS="${MODELS:-gnn sivi iwhvi}"
MAX_ITER="${MAX_ITER:-400000}"
HDIM="${HDIM:-100}"
ZDIM="${ZDIM:-50}"
HL="${HL:-2}"
NZ="${NZ:-50}"
ACTIVATION="${ACTIVATION:-elu}"
RUN_TAG="${RUN_TAG:-paper}"
WORKDIR="${WORKDIR:-results/paper}"
EMP_FREQ="${EMP_FREQ:-0}"
DO_EVALUATION="${DO_EVALUATION:-1}"
EVAL_REPEATS="${EVAL_REPEATS:-100}"
EVAL_RUNS="${EVAL_RUNS:-10000}"
EMPFREQ_TREES="${EMPFREQ_TREES:-42}"
REPLICATES="${REPLICATES:-1}"
SEED="${SEED:-2023}"

cd "$ROOT"

check_dataset() {
  local dataset="$1"
  test -f "data/hohna_datasets_fasta/${dataset}.fasta" || {
    echo "Missing alignment: data/hohna_datasets_fasta/${dataset}.fasta" >&2
    exit 2
  }
  for replicate in {1..10}; do
    test -f "data/ufboot_data_DS1-8/${dataset}/${dataset}_ufboot_rep_${replicate}" || {
      echo "Missing UFBoot replicate: data/ufboot_data_DS1-8/${dataset}/${dataset}_ufboot_rep_${replicate}. Download the full data package and preserve its directory structure." >&2
      exit 2
    }
  done
}

for dataset in $DATASETS; do
  check_dataset "$dataset"
  for model in $MODELS; do
    case "$model" in
      gnn) grad_method=vimco ;;
      sivi|iwhvi) grad_method="$model" ;;
      *) echo "Unsupported model '$model'. Available in this launcher: gnn sivi iwhvi." >&2; exit 2 ;;
    esac
    for replicate in $(seq 1 "$REPLICATES"); do
      tag="$RUN_TAG"
      if [ "$REPLICATES" -gt 1 ]; then tag="${RUN_TAG}_rep${replicate}"; fi
      seed=$((SEED + replicate - 1))
      args=(--dataset "$dataset" --brlen_model "$model" --gradMethod "$grad_method"
        --date "$RUN_DATE" --run_tag "$tag" --workdir "$WORKDIR"
        --maxIter "$MAX_ITER" --hdim "$HDIM" --zdim "$ZDIM" --hL "$HL"
        --nz "$NZ" --activation "$ACTIVATION" --seed "$seed"
        --eval-repeats "$EVAL_REPEATS" --eval-runs "$EVAL_RUNS"
        --empFreqTrees "$EMPFREQ_TREES")
      if [ "$EMP_FREQ" = 1 ]; then args+=(--empFreq); fi
      echo "[train] dataset=$dataset model=$model run=$RUN_DATE tag=$tag seed=$seed"
      "$PYTHON_BIN" main.py "${args[@]}"
      if [ "$DO_EVALUATION" = 1 ]; then
        echo "[raw-eval] dataset=$dataset model=$model run=$RUN_DATE tag=$tag"
        "$PYTHON_BIN" main.py "${args[@]}" --test
      fi
    done
  done
done
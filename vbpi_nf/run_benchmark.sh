#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/python_env.sh"
resolve_nf_python
RUN_DATE="${RUN_DATE:-$(date +%Y%m%d%H%M%S)}"
RUN_TAG="${RUN_TAG:-vbpi_nf}"
WORKDIR="${WORKDIR:-results/vbpi_nf}"
DATASETS="${DATASETS:-DS1}"
MAX_ITER="${MAX_ITER:-400000}"
NF_FEATURE_DIM="${NF_FEATURE_DIM:-2}"
NF_HIDDEN_DIM="${NF_HIDDEN_DIM:-100}"
NF_LAYERS="${NF_LAYERS:-10}"
NF_STEPSZ_BRANCH="${NF_STEPSZ_BRANCH:-0.0001}"
DO_EVALUATION="${DO_EVALUATION:-1}"
EMP_FREQ="${EMP_FREQ:-0}"
EVAL_REPEATS="${EVAL_REPEATS:-100}"
EVAL_RUNS="${EVAL_RUNS:-10000}"
EVAL_ML_REPEATS="${EVAL_ML_REPEATS:-1000}"
EMPFREQ_TREES="${EMPFREQ_TREES:-42}"
REPLICATES="${REPLICATES:-1}"
SEED="${SEED:-2023}"
cd "$ROOT"

for dataset in $DATASETS; do
	test -f "data/hohna_datasets_fasta/${dataset}.fasta" || {
		echo "Missing alignment data/hohna_datasets_fasta/${dataset}.fasta" >&2
		exit 2
	}
	for replicate in {1..10}; do
		support_file="data/ufboot_data_DS1-8/${dataset}/${dataset}_ufboot_rep_${replicate}"
		test -f "$support_file" || {
			echo "Missing UFBoot file $support_file; download the full data package before running the requested datasets." >&2
			exit 2
		}
	done

	for replicate in $(seq 1 "$REPLICATES"); do
		tag="$RUN_TAG"
		if [ "$REPLICATES" -gt 1 ]; then tag="${RUN_TAG}_rep${replicate}"; fi
		seed=$((SEED + replicate - 1))
		args=(--dataset "$dataset" --flow_type realnvp
			--date "$RUN_DATE" --run-tag "$tag" --workdir "$ROOT/$WORKDIR" --seed "$seed"
			--maxIter "$MAX_ITER" --stepszBranch "$NF_STEPSZ_BRANCH"
			--eval-repeats "$EVAL_REPEATS" --eval-runs "$EVAL_RUNS"
			--eval-ml-repeats "$EVAL_ML_REPEATS" --empFreqTrees "$EMPFREQ_TREES"
			--nf "$NF_FEATURE_DIM" --sh "$NF_HIDDEN_DIM" --Lnf "$NF_LAYERS")
		if [ "$EMP_FREQ" = 1 ]; then args+=(--empFreq); fi
		echo "[VBPI-NF train] dataset=$dataset run=$RUN_DATE tag=$tag seed=$seed"
		"$PYTHON_BIN" "$SCRIPT_DIR/standalone/main.py" "${args[@]}"
		if [ "$DO_EVALUATION" = 1 ]; then
			echo "[VBPI-NF raw evaluation] dataset=$dataset run=$RUN_DATE tag=$tag"
			"$PYTHON_BIN" "$SCRIPT_DIR/standalone/main.py" "${args[@]}" --test
		fi
	done
done
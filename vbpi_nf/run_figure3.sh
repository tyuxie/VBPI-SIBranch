#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASETS="${DATASETS:-DS1}"
export RUN_TAG="${RUN_TAG:-figure3_nf}"
export DO_EVALUATION=0
exec bash "$SCRIPT_DIR/run_benchmark.sh"
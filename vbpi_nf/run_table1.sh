#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASETS="${DATASETS:-DS1 DS2 DS3 DS4 DS5 DS6 DS7 DS8}"
export RUN_TAG="${RUN_TAG:-table1_nf}"
export DO_EVALUATION=1
exec bash "$SCRIPT_DIR/run_benchmark.sh"
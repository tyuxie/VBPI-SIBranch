#!/usr/bin/env bash

resolve_nf_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    export PYTHON_BIN
    return
  fi
  if [ -x "$HOME/.conda/envs/vbpisibranch/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/vbpisibranch/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != base ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    echo "Set PYTHON_BIN or create the vbpisibranch Conda environment from environment.yml." >&2
    return 2
  fi
  export PYTHON_BIN
}
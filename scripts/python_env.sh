#!/usr/bin/env bash

resolve_project_python() {
  if [ -n "${PYTHON_BIN:-}" ]; then
    export PYTHON_BIN
    return
  fi

  if [ -x "$HOME/.conda/envs/vbpisibranch/bin/python" ]; then
    PYTHON_BIN="$HOME/.conda/envs/vbpisibranch/bin/python"
  elif [ -n "${CONDA_EXE:-}" ] && [ -x "${CONDA_EXE%/bin/conda}/envs/vbpisibranch/bin/python" ]; then
    PYTHON_BIN="${CONDA_EXE%/bin/conda}/envs/vbpisibranch/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_DEFAULT_ENV:-}" != "base" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  elif command -v conda >/dev/null 2>&1; then
    PYTHON_BIN="$(conda run -n vbpisibranch python -c 'import sys; print(sys.executable)' 2>/dev/null)" || {
      echo "Could not resolve Python from the vbpisibranch Conda environment. Create it with: conda env create -f environment.yml" >&2
      return 2
    }
  else
    echo "Set PYTHON_BIN or activate the vbpisibranch Conda environment before running this workflow." >&2
    return 2
  fi

  export PYTHON_BIN
}
#!/usr/bin/env bash
# Generate the 42 DS1 fixed-topology MrBayes launch files and run them.
#
# The launch files are assembled by make_nex.py from the shared template and
# topology list, then `mb` is run inside each treeN/ directory. MrBayes names
# its outputs after the input file, so each run yields treeN/treeN.nex.t in
# place -- the exact layout experiments/run_fig5_6.sh expects via
# MRBAYES_TREE_ROOT=<workdir>.
#
# Usage:
#   MRBAYES_ROOT=/path/to/workdir bash mrbayes/run_mrbayes.sh
#   MRBAYES_ROOT=/path/to/workdir MAXJOBS=8 FRESH=1 bash mrbayes/run_mrbayes.sh
#
# Env vars:
#   MRBAYES_ROOT  (required) work directory that will hold tree1..tree42
#   MAXJOBS       max parallel MrBayes processes (default: 8)
#   FRESH=1       delete existing treeN.nex.{ckp,ckp~,p,t,mcmc} before running
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${MRBAYES_ROOT:?set MRBAYES_ROOT to the work directory that will hold tree1..tree42}"
MAXJOBS="${MAXJOBS:-8}"
command -v mb >/dev/null || {
  echo "MrBayes (mb) not found. Install it first, e.g.: brew install mrbayes" >&2
  exit 2
}

# 1) (re)generate the 42 launch files from the unified sources
python3 "$HERE/make_nex.py" --src "$HERE" --root "$ROOT"

# 2) run one MrBayes job per tree, up to MAXJOBS in parallel.
#    `printf 'y'` answers the checkpoint-resume prompt; the scripts set
#    `autoclose=yes`, so MrBayes exits on its own when the run finishes.
for t in $(seq 1 42); do
  dir="$ROOT/tree$t"
  if [ "${FRESH:-0}" = 1 ]; then
    rm -f "$dir"/tree$t.nex.ckp "$dir"/tree$t.nex.ckp~ \
          "$dir"/tree$t.nex.p "$dir"/tree$t.nex.t "$dir"/tree$t.nex.mcmc
  fi
  ( cd "$dir" && printf 'y\n' | mb tree$t.nex > tree$t.mb.log 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do wait -n || true; done
done
wait

# 3) verify every run produced its tree log
bad=0
for t in $(seq 1 42); do
  test -s "$ROOT/tree$t/tree$t.nex.t" || { echo "MISSING: tree$t.nex.t" >&2; bad=1; }
done
if [ "$bad" = 0 ]; then
  echo "OK: all 42 tree*.nex.t produced under $ROOT"
  echo "Feed them to the Figure 5/6 launcher with: MRBAYES_TREE_ROOT=$ROOT"
fi

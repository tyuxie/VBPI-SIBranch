# VBPI-NF

VBPI-NF is a standalone implementation under this top-level directory. Its
trainer, data handling, phylogenetic likelihood, SBN topology model, branch-flow
models, conditional sampler, and fixed-tree runner are all in `standalone/`.
It does not import the general implementation from `src/`, root `main.py`, or
the general Python experiment utilities. Its launchers use the repository data
files and the dedicated Conda environment.

From the repository root, create the project Conda environment once:

```bash
conda env create -f environment.yml
```

NF launchers automatically select the dedicated `vbpisibranch` interpreter;
set `PYTHON_BIN` to override it.

## Figure 3

The NF-only portion of the training-trajectory experiment defaults to DS1:

```bash
DATASETS=DS1 RUN_DATE="$(date +%Y%m%d%H%M%S)" bash vbpi_nf/run_figure3.sh
```

The repository-level `experiments/run_figure3.sh` invokes this NF workflow and
the GNN/SIVI/IWHVI workflows with the same run date. It defaults to 100 seeded
training replicates per method; set `REPLICATES=1` and `MAX_ITER=1` for a
launcher smoke test.

## Table 1

Train and run the raw benchmark evaluation for the NF method across DS1-DS8:

```bash
RUN_DATE="$(date +%Y%m%d%H%M%S)" bash vbpi_nf/run_table1.sh
```

The workflow requires all DS1-DS8 alignments and ten UFBoot replicate files
per dataset under `data/`. The launcher stops with the missing file path if
the complete data package has not been downloaded and placed in the repository
layout. Set `DATASETS=DS1` for a reduced test run.

The NF CLI defaults to the paper's RealNVP flow with feature dimension 2,
hidden width 100, and 10 coupling layers. Change these through
`NF_FEATURE_DIM`, `NF_HIDDEN_DIM`, and `NF_LAYERS`.

`MAX_ITER`, `EVAL_REPEATS`, `EVAL_RUNS`, and `EVAL_ML_REPEATS` can reduce the
work for smoke tests. Result files are stored under `results/vbpi_nf/` by
default, with separate dataset/flow/run-tag/date/seed directories.

The repository-level `experiments/run_table1.sh` invokes this standalone NF
Table 1 workflow alongside the GNN, SIVI, and IWHVI benchmark runs.

## Fixed-tree NF

The NF-only fixed-tree implementation also lives inside this directory:

```bash
DATASET=DS1 TREE_IDS="1" MAX_ITER=1 BOUND_RUNS=1 \
	RUN_TAG=fixed-tree-smoke bash vbpi_nf/run_fixed_tree.sh
```

Set `TREE_IDS` to posterior topology IDs to restrict a run; omit it to fit all
42 DS1 topologies. Outputs include `final.pt`, `training_trace.csv`, optional
`fixed_tree_lower_bound_samples.npy`, and optional
`variational_branch_samples.npy` under `results/vbpi_nf/fixed_tree/`.

## Conditional branch samples

After training, Figure 1 conditional samples for DS8 trees 1 and 2 can be
generated with the same run date/tag:

```bash
RUN_DATE=<training-run-date> RUN_TAG=figure1_nf bash vbpi_nf/run_figure1_samples.sh
```

The Figure 5/6 NF comparison samples for DS1 posterior trees 1, 2, 36, and 37
use the run tag used to train the DS1 NF checkpoint. The full Figure 5/6
workflow uses `figure5_6`:

```bash
RUN_DATE=<training-run-date> RUN_TAG=figure5_6 bash vbpi_nf/run_figure5_6_samples.sh
```

These commands use NF's own `standalone/sample_branches.py` and save raw
positive branch-length arrays under `results/vbpi_nf/`. They do not calculate
KL/TV summaries or create manuscript plots/tables.

## Source

The standalone implementation follows Cheng Zhang, *Improved Variational
Bayesian Phylogenetic Inference with Normalizing Flows*, NeurIPS 2020. The
upstream implementation supplied with this repository is vendored in
`standalone/` so NF runs do not depend on the general VBPI source package.
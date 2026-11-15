# Experiment recipes

One command per paper experiment; `run.sh` is the only entry point.

| Target | Command | Produces |
|---|---|---|
| Table 1 | `bash experiments/run.sh table1` | GNN/SIVI/IWHVI + NF marginal-likelihood runs on DS1-DS8, with raw Monte Carlo evaluations |
| Table 3 | `bash experiments/run.sh table3` | SIVI/IWHVI with 1-4 message-passing rounds on DS1, DS8 |
| Table 4 | `bash experiments/run.sh table4` | SIVI/IWHVI with ELU and ReLU on DS1, DS8 |
| Figure 1 | `bash experiments/run.sh figure1` | DS8 joint-branch samples for GNN/SIVI/IWHVI + NF |
| Figure 3 | `bash experiments/run.sh figure3` | 100 replicates per model on DS1-DS8 (GNN/SIVI/IWHVI + NF) |
| Figure 4, Table 2 | `bash experiments/run.sh fig4_table2` | DS1 fixed-topology branch models for the 42 posterior topologies (GNN/SIVI/IWHVI/lognorm + NF) |
| Figure 5, Figure 6 | `MRBAYES_TREE_ROOT=/path/to/DS1-larger bash experiments/run.sh figure5_6` | Fixed-tree and joint branch posteriors vs. MrBayes reference (needs the long-chain tree logs, see `mrbayes/README.md`) |
| Figure 7 | `bash experiments/run.sh figure7` | SIVI/IWHVI at NZ = 20/50/100 on DS1-DS8 |
| Figure 9 | `bash synthetic_exp/run_L8_synthetic_exp.sh` | Synthetic recovery experiment (separate workflow) |

`run.sh table1/table3/table4/figure1/figure3/figure7` train via
`run_benchmark_matrix.sh` (paper scale: `MAX_ITER=400000`, `EVAL_RUNS=10000`);
`fig4_table2`/`figure5_6` add fixed-tree fits at `MAX_ITER=200000` and joint
sampling (`N_SAMPLES=1000000`).

## Common knobs (env vars)

| Var | Default | Meaning |
|---|---|---|
| `RUN_DATE` | now (`YYYYmmddHHMMSS`) | shared run id; reusing it groups all stages of one run |
| `WORKDIR` / `NF_WORKDIR` | `results/<recipe>` | output roots for the VBPI / VBPI-NF legs |
| `MODELS` | recipe-dependent | e.g. `gnn sivi iwhvi`, `sivi iwhvi`, `lognorm sivi iwhvi` |
| `DATASETS` | recipe-dependent | e.g. `DS1 DS8` |
| `MAX_ITER` | 400000 | training iterations (engine) |
| `REPLICATES` | 1 | replicates per dataset/model |
| `HL_GRID` | `1 2 3 4` | Table 3 message-passing rounds |
| `NZ_GRID` | `20 50 100` | Figure 7 NZ grid |
| `ACTIVATIONS` | `elu relu` | Table 4 activations |
| `TREE_IDS` | all 42 | restrict `fig4_table2`/`figure5_6` to subsets |
| `BRANCH_SAMPLES` | 1000000 | joint / fixed-tree sample count |
| `MRBAYES_TREE_ROOT` | required for `figure5_6` | long-chain MrBayes tree logs (see `mrbayes/README.md`) |

## Smoke testing

Everything above is paper scale (weeks of GPU). To verify a recipe end to
end in minutes, shrink the knobs first:

```bash
MAX_ITER=1 EVAL_RUNS=10 EVAL_REPEATS=1 REPLICATES=1 TREE_IDS=1 BRANCH_SAMPLES=100 \
  bash experiments/run.sh table1      # and so on for the other recipes
```

`figure5_6` additionally needs `MRBAYES_TREE_ROOT` even for a smoke test; the
launcher only checks that the 42 `.nex.t` files exist.

## Relation to the old launchers

`run.sh` replaces `run_table1.sh`, `run_table3.sh`, `run_table4.sh`,
`run_figure1.sh`, `run_figure3.sh`, `run_figure7.sh`, `run_fig4_table2.sh`,
and `run_fig5_6.sh`; each old wrapper's env overrides map to the knobs above.
`run_benchmark_matrix.sh`, `run_fixed_tree.py`, `sample_joint_branches.py`,
and `prepare_mrbayes_branch_samples.py` remain the underlying engines.

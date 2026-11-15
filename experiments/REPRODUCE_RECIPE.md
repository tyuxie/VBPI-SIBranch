# Reproduction recipes

Each command below runs one experiment and writes its outputs under
`results/`. Run from the repository root with the `vbpisibranch` environment
activated; `PYTHON_BIN` selects another interpreter. `DATA.md` documents the
input data.

## Main-text experiments

| Target | Command | Produces |
|---|---|---|
| Table 1 | `bash experiments/reproduce.sh mll_benchmark` | GNN/SIVI/IWHVI + NF marginal-likelihood runs on DS1-DS8, with raw Monte Carlo evaluations |
| Table 3 | `bash experiments/reproduce.sh mp_rounds_ablation` | SIVI/IWHVI with 1-4 message-passing rounds on DS1, DS8 |
| Table 4 | `bash experiments/reproduce.sh activation_ablation` | SIVI/IWHVI with ELU and ReLU on DS1, DS8 |
| Figure 1 | `bash experiments/reproduce.sh joint_branch_ds8` | DS8 joint-branch samples for GNN/SIVI/IWHVI + NF |
| Figure 3 | `bash experiments/reproduce.sh training_performance` | Training performance on DS1-DS8 (GNN/SIVI/IWHVI + NF) |
| Figure 4, Table 2 | `bash experiments/reproduce.sh fixedtree_ds1` | DS1 fixed-topology branch models for the 42 posterior topologies (GNN/SIVI/IWHVI/lognorm + NF) |
| Figure 5, Figure 6 | `MRBAYES_TREE_ROOT=/path/to/DS1-larger bash experiments/reproduce.sh branch_dist` | Fixed-tree and joint branch posteriors vs. the MrBayes reference |
| Figure 7 | `bash experiments/reproduce.sh nz_ablation` | SIVI/IWHVI at NZ = 20/50/100 on DS1-DS8 |

`mll_benchmark`/`mp_rounds_ablation`/`activation_ablation`/`joint_branch_ds8`/`training_performance`/`nz_ablation` train via
`run_benchmark_matrix.sh` (paper scale: `MAX_ITER=400000`, `EVAL_RUNS=10000`);
`fixedtree_ds1`/`branch_dist` add fixed-tree fits at `MAX_ITER=200000` and joint
sampling (`N_SAMPLES=1000000`). Figures 2 and 8 are static illustrations
without computational content.

## Synthetic experiments (Figure 9)

```bash
export RUN_DATE="full-$(date +%Y%m%d%H%M%S)"
bash synthetic_exp/run_L8_synthetic_exp.sh
```

The script discovers every `dna.fasta` under `synthetic_exp/L8trees/`, trains
the MIWLB model (`iwhvi` branch-length model and gradient method) for 40,000
iterations, and writes 1,000 posterior trees per dataset. Result directories
are not overwritten; `RUN_DATE` separates runs. Each run directory
holds the checkpoint, the training trace, and
`final_sampled_<N>_trees.newick`. The synthetic data are described in
`DATA.md`.

## MrBayes reference runs

Figures 5-6 compare against long-chain fixed-topology MrBayes samples for the
42 DS1 posterior topologies. `mrbayes/` contains a template, the 42 topologies,
and a generator that writes the 42 per-tree launch files:

| File | Contents |
|---|---|
| `DS1_fixed_tree.nex` | Template: TreeBASE DS1 alignment (27 taxa, 1949 sites), `TREES` block, and `MRBAYES` block (JC69, topology fixed, Exp(10) branch-length prior, 4 chains); the fixed-topology declaration is a `__MYSTARTTREE_LINE__` placeholder |
| `DS1_topologies.txt` | The 42 fixed topologies, one Newick string per line (`tree1`..`tree42`) |
| `make_nex.py` | Reassembles `treeN/treeN.nex` from the template and topology list |
| `run_mrbayes.sh` | Generates the launch files, runs all 42 MrBayes jobs in parallel, and verifies the outputs |

```bash
python3 mrbayes/make_nex.py --root /path/to/workdir        # launch files only
MRBAYES_ROOT=/path/to/workdir MAXJOBS=8 bash mrbayes/run_mrbayes.sh
```

## VBPI-NF baseline experiments

The `vbpi_nf/` directory contains a vendored VBPI-NF implementation (Zhang,
NeurIPS 2020). Its commands are:

| Target | Command | Produces |
|---|---|---|
| NF training (generic) | `bash vbpi_nf/reproduce.sh train` | RealNVP training (+ optional raw evaluation) over `DATASETS` |
| Table 1 (NF leg) | `bash vbpi_nf/reproduce.sh mll_benchmark` | DS1-DS8 RealNVP runs with raw Monte Carlo evaluations |
| Figure 3 (NF leg) | `REPLICATES=100 bash vbpi_nf/reproduce.sh training_performance` | Training performance on DS1-DS8 (NF) |
| Figure 1 (NF samples) | `RUN_DATE=<date> bash vbpi_nf/reproduce.sh joint_branch_ds8` | Joint branch samples for DS8 trees from the Figure 1 NF checkpoint |
| Figures 5-6 (NF samples) | `RUN_DATE=<date> bash vbpi_nf/reproduce.sh joint_branch_ds1` | Joint branch samples for DS1 trees from the Figures 5-6 NF checkpoint |
| Figures 4-6 (NF fixed-tree) | `TREE_IDS="1 2 3" bash vbpi_nf/reproduce.sh fixed_tree` | Fixed-topology branch fits (all 42 trees unless `TREE_IDS` is set) |

The NF training recipes run at paper scale (`MAX_ITER=400000`,
`EVAL_RUNS=10000`); `fixed_tree` at `MAX_ITER=200000`.

## Common environment variables

| Var | Default | Meaning |
|---|---|---|
| `RUN_DATE` | now (`YYYYmmddHHMMSS`) | shared run id; reusing it groups all stages of one run |
| `WORKDIR` / `NF_WORKDIR` | `results/<recipe>` | output roots for the VBPI / VBPI-NF workflows |
| `MODELS` | recipe-dependent | e.g. `gnn sivi iwhvi`, `sivi iwhvi`, `lognorm sivi iwhvi` |
| `DATASETS` | recipe-dependent | e.g. `DS1 DS8` |
| `MAX_ITER` | 400000 (benchmark) / 200000 (fixed-tree) | training iterations |
| `REPLICATES` | 1 | seeded runs per dataset/model |
| `HL_GRID` / `NZ_GRID` / `ACTIVATIONS` | `1 2 3 4` / `20 50 100` / `elu relu` | Table 3 rounds, Figure 7 NZ grid, Table 4 activations |
| `TREE_IDS` | all 42 | fixed-tree topology subset for `fixedtree_ds1`/`branch_dist`/NF `fixed_tree` |
| `N_SAMPLES` / `BRANCH_SAMPLES` / `BOUND_RUNS` | 100000 / 1000000 / 0 | sampling and fixed-tree evaluation options |
| `NF_LAYERS` / `NF_HIDDEN_DIM` / `NF_FEATURE_DIM` | 10 / 100 / 2 | NF flow depth, hidden size, feature dimension |
| `NF_STEPSZ_BRANCH` | 0.0001 | NF branch-length learning rate |
| `DO_EVALUATION` / `EMP_FREQ` | 1 / 0 | run raw evaluation after training / use empirical topology frequencies |
| `MRBAYES_TREE_ROOT` | required for `branch_dist` | long-chain MrBayes tree logs (see `DATA.md`) |
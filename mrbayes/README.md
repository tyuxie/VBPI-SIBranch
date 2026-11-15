# DS1 fixed-topology MrBayes runs (Figures 5-6 reference samples)

The Figure 5/6 workflow compares VBPI branch-length posteriors against
ground-truth distributions from long MrBayes runs on the 42 DS1 posterior
topologies. This directory holds the unified source for those 42 runs; the
per-tree launch files are generated from it rather than stored in duplicate.

## Layout

| File | Contents |
|---|---|
| `DS1_fixed_tree.nex` | Shared template: TreeBASE DS1 alignment (27 taxa, 1949 sites), `TREES` block, and `MRBAYES` block (JC69, topology fixed, Exp(10) branch-length prior, 4 chains). The fixed-topology declaration is a `__MYSTARTTREE_LINE__` placeholder. |
| `DS1_topologies.txt` | The 42 fixed topologies, one Newick string per line (`tree1`..`tree42`). |
| `make_nex.py` | Reassembles `treeN/treeN.nex` from the template and topology list. |
| `run_mrbayes.sh` | Generates the launch files and runs all 42 MrBayes jobs (parallel, checkpoint-resumable), then verifies the outputs. |

## Usage

```bash
# generate the 42 launch files only
python3 mrbayes/make_nex.py --root /path/to/workdir

# generate + run (requires MrBayes: brew install mrbayes)
MRBAYES_ROOT=/path/to/workdir MAXJOBS=8 bash mrbayes/run_mrbayes.sh
```

`run_mrbayes.sh` accepts `MAXJOBS` (parallel jobs, default 8) and `FRESH=1`
(delete existing checkpoints/outputs and start over). MrBayes derives output
names from the input path, so each run writes `treeN/treeN.nex.t` next to its
launch file — exactly the layout `experiments/run_fig5_6.sh` expects; point it
there with `MRBAYES_TREE_ROOT=/path/to/workdir`.

## Short vs. long chain

The template's `mcmc` line runs 10 million generations (`ngen=10000000`,
`samplef=1000`, 4 chains), yielding 10,001 samples per tree — the *short*
chain. The manuscript's Figures 5-6 reference distributions come from the
*long* chain (`ngen=1000000000`), whose per-tree tree logs (~1M samples) are
distributed in the complete data package (see `synthetic_exp/DATA.md`), not
re-generated locally: re-running the long chain takes on the order of months
of CPU per tree. Raising `ngen` in the template (or in generated launch files)
before running reproduces it, if ever needed.

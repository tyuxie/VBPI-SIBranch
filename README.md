### <div align="center"><font size=5> Variational Bayesian Phylogenetic Inference with Semi-implicit Branch Length Distributions (VBPI-SIBranch)</font><div> 

<div align="center">
Tianyu Xie, Frederick A. Matsen, Marc A. Suchard, Cheng Zhang
</div>

## Installation

This repository is a light CPU-based implementation of VBPI-SIBranch. Create
the torch environment with:

```
conda env create -f environment.yml
conda activate vbpisibranch
```

## Training

Run the VBPI baseline ([Zhang, 2023](https://arxiv.org/abs/2302.08840)):

```
python main.py --dataset DS1 --brlen_model gnn --gradMethod vimco
```

Train the VBPI-SIBranch SIVI branch-length model:

```
python main.py --dataset DS1 --brlen_model sivi --gradMethod sivi --nParticle 10 --nz 50
```

Train VBPI-SIBranch with MIWLB:

```
python main.py --dataset DS1 --brlen_model iwhvi --gradMethod iwhvi --nParticle 10 --nz 50
```

`--nParticle` sets the number of particles ($K$) and `--nz` the number of
extra samples ($J$). `--empFreq` monitors the KL divergence from the empirical
topology distribution during training.

## Evaluation

Evaluate a trained variational approximation:

```
python main.py --dataset DS1 --brlen_model gnn --test
python main.py --dataset DS1 --brlen_model sivi --test
python main.py --dataset DS1 --brlen_model iwhvi --test
```

These commands auto-load the trained model and compute the evidence lower
bound, the 10-sample lower bound, and marginal-likelihood estimates. For DS1,
adding `--empFreq` computes evidence lower bounds on individual trees.

## Performance

Training time (hours) for 400,000 iterations with the default settings, on a
single core of an Intel Xeon Platinum 9242:

| Method \ Data set | DS1|DS2|DS3|DS4|DS5|DS6|DS7|DS8|
|----|----|----|----|----|----|----|----|----|
|VBPI|11.8|13.2|15.7|17.3|20.0|20.5|26.0|26.2|
|VBPI-SIBranch (MIWLB)|23.6|27.4|34.2|36.3|43.8|41.4|52.2|55.6|

Each training run writes `<run>_times.npy`, the cumulative wall-clock seconds
per logging block; the "Computation time" section of
[`experiments/REPRODUCE_RECIPE.md`](experiments/REPRODUCE_RECIPE.md) describes
how to aggregate these arrays.

## Experiments

Experiment commands and environment variables are listed in
[`experiments/REPRODUCE_RECIPE.md`](experiments/REPRODUCE_RECIPE.md); input
data and provenance are documented in [`DATA.md`](DATA.md).

## Repository layout

| Path | Contents |
|---|---|
| `main.py`, `src/` | VBPI-SIBranch implementation: SBN topology model, branch-length models, phylogenetic likelihood, training and evaluation driver |
| `experiments/` | Benchmarks and fixed-tree/sampling launchers for the main-text experiments |
| `vbpi_nf/` | Vendored VBPI-NF baseline implementation (Zhang, NeurIPS 2020) |
| `synthetic_exp/` | Synthetic eight-taxon data and workflow (Figure 9) |
| `mrbayes/` | MrBayes fixed-topology launch files used for the branch-length reference |
| `data/`, `DATA.md` | Benchmark alignments, UFBoot support trees, and data documentation |

## References

Zhang C. *Learnable Topological Features For Phylogenetic Inference via Graph Neural Networks*. ICLR 2023.

Cock PJA, Antao T, Chang JT, et al. *Biopython: freely available Python tools for computational molecular biology and bioinformatics*. Bioinformatics. 2009.

Huerta-Cepas J, Serra F, Bork P. *ETE 3: Reconstruction, analysis, and visualization of phylogenomic data*. Molecular Biology and Evolution. 2016.

Minh BQ, Nguyen MAT, von Haeseler A. *Ultrafast approximation for phylogenetic bootstrap*. Molecular Biology and Evolution. 2013.

Paszke A, Gross S, Massa F, et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. NeurIPS. 2019.

Ronquist F, Teslenko M, van der Mark P, et al. *MrBayes 3.2: Efficient Bayesian phylogenetic inference and model choice across a large model space*. Systematic Biology. 2012.

--- 

Citation:
```
@article{xie2024vbpisibranch,
  title={Variational Bayesian Phylogenetic Inference with Semi-implicit Branch Length Distributions},
  author={Xie, Tianyu and Matsen IV, Frederick A and Suchard, Marc A and Zhang, Cheng},
  journal={arXiv preprint arXiv:2408.05058},
  year={2024}
}
```

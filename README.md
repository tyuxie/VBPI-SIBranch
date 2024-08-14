### <div align="center"><font size=5> Variational Bayesian Phylogenetic Inference with Semi-implicit Branch Length Distributions (VBPI-SIBranch)</font><div> 

<div align="center">
Tianyu Xie, Frederick A. Matsen, Marc A. Suchard, Cheng Zhang
</div>


## Installation
This repository is a light CPU-based implementation of VBPI-SIBranch.
To create the torch environment, use the following command:
```
conda env create -f environment.yml
```

## Training
One can use the following command to reproduce the VBPI baseline ([Zhang, 2023](https://arxiv.org/abs/2302.08840)):
```
python main.py --dataset DS1 --brlen_model gnn --gradMethod vimco
```
To reproduce the results of VBPI-SIBranch with MSILB, run the command:
```
python main.py --dataset DS1 --brlen_model sivi --gradMethod sivi --nParticle 10 --nz 50
```
To reproduce the results of VBPI-SIBranch with MIWLB, run the command:
```
python main.py --dataset DS1 --brlen_model iwhvi --gradMethod iwhvi --nParticle 10 --nz 50
```

You can freely specify the number of particles ($K$) in "--nParticle" and the number of extra samples ($J$) in "--nz" for ablation studies. 
The KL divergence from the tree topology ground truth to its variational approximation can be monitered during training by adding the "--empFreq" argument.

Here we provide the time comsumption (in hours) for completing the training process (400K itertions) with the default setting, on a single core of Intel Xeon Platinum 9242 processor.

| Method \ Data set | DS1|DS2|DS3|DS4|DS5|DS6|DS7|DS8|
|----|----|----|----|----|----|----|----|----|
|VBPI|11.8|13.2|15.7|17.3|20.0|20.5|26.0|26.2|
|VBPI-SIBranch (MSILB)|19.5|21.7|25.1|28.2|34.9|33.9|42.6|43.8|
|VBPI-SIBranch (MIWLB)|23.6|27.4|34.2|36.3|43.8|41.4|52.2|55.6|

## Evaluation
Run the following code to evaluate the variational approximation:
```
python main.py --dataset DS1 --brlen_model gnn --test
python main.py --dataset DS1 --brlen_model sivi --test
python main.py --dataset DS1 --brlen_model iwhvi --test
```
These commands will auto-load the trained model and compute the evidence lower bound, 10-sample lower bound and marginal likelihood estimates.

For DS1, this codebase also supports computing the evidence lower bounds on individual trees by adding the "--empFreq" argument.

## References
Zhang C. *Learnable Topological Features For Phylogenetic Inference via Graph Neural Networks*. ICLR 2023.

--- 

Please consider citing our work if you find this codebase useful:
```
@article{xie2024vbpisibranch,
  title={Variational Bayesian Phylogenetic Inference with Semi-implicit Branch Length Distributions},
  author={Xie, Tianyu and Matsen IV, Frederick A and Suchard, Marc A and Zhang, Cheng},
  journal={arXiv preprint arXiv:2408.05058},
  year={2024}
}
```
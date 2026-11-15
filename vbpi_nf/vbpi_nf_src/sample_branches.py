import argparse
from pathlib import Path

import numpy as np
import torch

from dataManipulation import loadData
from treeManipulation import namenum
from utils import BitArray, get_support_from_mcmc, summary, summary_raw, tree_process
from vbpi import VBPI

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description='Sample NF branch lengths conditional on a posterior topology.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tree-id', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--workdir', type=Path, default=REPO_ROOT / 'results' / 'vbpi_nf')
    parser.add_argument('--flow-type', choices=('realnvp', 'planar', 'identity'), default='realnvp')
    parser.add_argument('--feature-dim', type=int, default=2)
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[100])
    parser.add_argument('--flow-layers', type=int, default=10)
    parser.add_argument('--n-samples', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    data_root = REPO_ROOT / 'data'
    fasta = data_root / 'hohna_datasets_fasta' / '{}.fasta'.format(args.dataset)
    data, taxa = loadData(str(fasta), 'fasta')
    support_path = data_root / 'ufboot_data_DS1-8'
    support_trees, support_names = summary_raw(args.dataset, str(support_path) + '/')
    rootsplit, subsplit = get_support_from_mcmc(taxa, support_trees, support_names)
    posterior_path = data_root / 'raw_data_DS1-8'
    posterior_trees, posterior_names, _ = summary(args.dataset, str(posterior_path) + '/')
    target_name = 'tree_{}'.format(args.tree_id)
    if target_name not in posterior_trees:
        raise ValueError('Posterior tree {} not found for {}'.format(target_name, args.dataset))
    tree = posterior_trees[target_name].copy()
    tree_process(tree, BitArray(taxa))
    namenum(tree, taxa)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = VBPI(
        taxa,
        rootsplit,
        subsplit,
        data,
        pden=np.ones(4) / 4.0,
        subModel=('JC', 1.0),
        feature_dim=args.feature_dim,
        hidden_sizes=args.hidden_sizes,
        num_of_layers_nf=args.flow_layers,
        flow_type=args.flow_type,
    )
    model.load_from(str(args.checkpoint))
    with torch.no_grad():
        samples = model.branch_model.sample_branch_fixtree(tree, args.n_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, samples.exp().cpu().numpy())
    print('Saved {} conditional NF samples to {}'.format(args.n_samples, args.output))


if __name__ == '__main__':
    main()
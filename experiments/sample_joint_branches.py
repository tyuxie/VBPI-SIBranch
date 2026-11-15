import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataManipulation import loadData
from src.treeManipulation import namenum
from src.utils import get_support_from_mcmc, summary, summary_raw, BitArray, tree_process
from src.vbpi import VBPI


def main():
    parser = argparse.ArgumentParser(description='Sample branch lengths conditional on a posterior topology from a fitted joint VBPI model.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--tree-id', type=int, required=True)
    parser.add_argument('--model', choices=('gnn', 'sivi', 'iwhvi'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--n-samples', type=int, default=1000000)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hidden-dim', type=int, default=100)
    parser.add_argument('--latent-dim', type=int, default=50)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--activation', choices=('elu', 'relu'), default='elu')
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    support_root = args.data_root / 'ufboot_data_DS1-8'
    raw_root = args.data_root / 'raw_data_DS1-8'
    fasta = args.data_root / 'hohna_datasets_fasta' / '{}.fasta'.format(args.dataset)
    support_trees, support_names = summary_raw(args.dataset, str(support_root) + '/')
    taxa = [record.id for record in SeqIO.parse(str(fasta), 'fasta')]
    rootsplit, subsplit = get_support_from_mcmc(taxa, support_trees, support_names)
    data, taxa = loadData(str(fasta), 'fasta')
    posterior_trees, posterior_names, _ = summary(args.dataset, str(raw_root) + '/')
    tree_name = 'tree_{}'.format(args.tree_id)
    if tree_name not in posterior_trees:
        raise ValueError('Posterior tree {} not found for {}'.format(tree_name, args.dataset))
    tree = posterior_trees[tree_name].copy()
    tree_process(tree, BitArray(taxa))
    namenum(tree, taxa)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = VBPI(
        taxa,
        rootsplit,
        subsplit,
        data,
        pden=np.ones(4) / 4.0,
        subModel=('JC', 1.0),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        branch_model=args.model,
        gnn_type='edge',
        aggr='sum',
        activation=args.activation,
    )
    model.load_from(str(args.checkpoint))
    with torch.no_grad():
        log_samples = model.branch_model.sample_branch(tree, args.n_samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, log_samples.exp().cpu().numpy())
    print('Saved {} conditional branch samples to {}'.format(args.n_samples, args.output))


if __name__ == '__main__':
    main()
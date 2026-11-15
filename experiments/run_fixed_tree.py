import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from Bio import Phylo
from ete3 import Tree
from io import StringIO

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('OMP_NUM_THREADS', '1')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataManipulation import loadData
from src.fixed_tree import FixedTreeVBPI
from src.treeManipulation import namenum
from src.utils import summary, tree_process, BitArray


def load_fixed_tree(dataset, tree_id, data_root):
    raw_root = data_root / 'raw_data_DS1-8'
    fasta_path = data_root / 'hohna_datasets_fasta' / '{}.fasta'.format(dataset)
    trees, names, weights = summary(dataset, str(raw_root) + '/', samp_size=750001)
    key = 'tree_{}'.format(tree_id)
    if key not in trees:
        raise ValueError('Tree ID {} is unavailable for {}'.format(tree_id, dataset))
    data, taxa = loadData(str(fasta_path), 'fasta')
    tree = trees[key].copy()
    tree_process(tree, BitArray(taxa))
    namenum(tree, taxa)
    return tree, data, taxa


def write_records(path, records):
    if not records:
        return
    with path.open('w', newline='', encoding='utf-8') as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def run_one(args, tree_id):
    tree, data, taxa = load_fixed_tree(args.dataset, tree_id, args.data_root)
    output_dir = args.output_root / args.dataset / 'tree_{}'.format(tree_id) / args.model / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    logger = logging.getLogger('fixed-tree-{}-{}'.format(args.model, tree_id))
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(output_dir / 'run.log')
    logger.addHandler(handler)
    logger.info('dataset=%s tree_id=%s model=%s iterations=%s seed=%s', args.dataset, tree_id, args.model, args.iterations, args.seed)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model = FixedTreeVBPI(
        taxa,
        tree,
        data,
        branch_model=args.model,
        latent_dim=args.latent_dim,
        hidden_units=args.hidden_units,
        activation=args.activation,
    )
    records = model.fit(
        iterations=args.iterations,
        batch_size=args.batch_size,
        z_batch_size=args.z_batch_size,
        learning_rate=args.learning_rate,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=output_dir,
    )
    write_records(output_dir / 'training_trace.csv', records)
    if args.bound_runs:
        bounds = model.lower_bound_samples(args.bound_runs, args.eval_z_batch_size)
        np.save(output_dir / 'fixed_tree_lower_bound_samples.npy', bounds)
    if args.branch_samples:
        samples = model.branch_samples(args.branch_samples)
        np.save(output_dir / 'variational_branch_samples.npy', samples)
    logger.info('finished output_dir=%s', output_dir)
    handler.close()
    logger.removeHandler(handler)
    print(output_dir)


def main():
    parser = argparse.ArgumentParser(description='Train a fixed-topology branch-length model for Figure 4-6 and Table 2.')
    parser.add_argument('--dataset', default='DS1')
    parser.add_argument('--tree-id', type=int, default=1)
    parser.add_argument('--all-trees', action='store_true', help='run tree IDs 1 through 42 sequentially')
    parser.add_argument('--tree-ids', type=int, nargs='+', help='run an explicit list of posterior tree IDs')
    parser.add_argument('--model', choices=('sivi', 'iwhvi', 'lognorm'), required=True)
    parser.add_argument('--activation', choices=('elu', 'relu'), default='elu')
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--output-root', type=Path, default=Path('results/fixed_tree'))
    parser.add_argument('--run-id', default='full')
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--z-batch-size', type=int, default=50)
    parser.add_argument('--eval-z-batch-size', type=int, default=1000)
    parser.add_argument('--latent-dim', type=int, default=50)
    parser.add_argument('--hidden-units', type=int, default=100)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=100000)
    parser.add_argument('--bound-runs', type=int, default=0)
    parser.add_argument('--branch-samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    args = parser.parse_args()

    if args.tree_ids:
        tree_ids = args.tree_ids
    elif args.all_trees:
        tree_ids = range(1, 43)
    else:
        tree_ids = [args.tree_id]
    for tree_id in tree_ids:
        run_one(args, tree_id)


if __name__ == '__main__':
    main()
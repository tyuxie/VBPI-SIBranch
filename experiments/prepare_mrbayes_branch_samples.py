import argparse
from io import StringIO
from pathlib import Path
import sys

import numpy as np
from Bio import Phylo
from ete3 import Tree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.dataManipulation import loadData
from src.treeManipulation import namenum


def main():
    parser = argparse.ArgumentParser(description='Convert per-fixed-tree MrBayes Nexus samples to branch vectors.')
    parser.add_argument('--dataset', default='DS1')
    parser.add_argument('--tree-id', type=int, required=True)
    parser.add_argument('--nexus-tree-file', type=Path, required=True)
    parser.add_argument('--fasta', type=Path, default=Path('data/hohna_datasets_fasta/DS1.fasta'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--burnin', type=int, default=2500)
    args = parser.parse_args()

    _, taxa = loadData(str(args.fasta), 'fasta')
    branch_rows = []
    for phylo_tree in Phylo.parse(str(args.nexus_tree_file), 'nexus'):
        buffer = StringIO()
        Phylo.write(phylo_tree, buffer, 'newick')
        tree = Tree(buffer.getvalue().strip(), format=1)
        namenum(tree, taxa)
        row = np.zeros(2 * len(taxa) - 3, dtype=float)
        for node in tree.traverse('postorder'):
            if not node.is_root():
                row[int(node.name)] = float(node.dist)
        if np.any(row <= 0):
            raise ValueError('Nonpositive branch length found in tree {}'.format(args.tree_id))
        branch_rows.append(row)

    samples = np.asarray(branch_rows, dtype=float)
    if args.burnin >= len(samples):
        raise ValueError('Burn-in {} removes all {} samples'.format(args.burnin, len(samples)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, samples[args.burnin:])
    print('Saved {} post-burn-in trees to {}'.format(len(samples) - args.burnin, args.output))


if __name__ == '__main__':
    main()
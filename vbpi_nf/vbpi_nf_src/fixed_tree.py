import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch

from dataManipulation import loadData
from deep_branchModel import DeepModel
from phyloModel import PHY
from treeManipulation import namenum
from utils import BitArray, get_support_from_mcmc, summary, summary_raw, tree_process
from vector_sbnModel import SBN

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_inputs(dataset, tree_id):
    data_root = REPO_ROOT / 'data'
    data, taxa = loadData(str(data_root / 'hohna_datasets_fasta' / '{}.fasta'.format(dataset)), 'fasta')
    support_trees, support_names = summary_raw(dataset, str(data_root / 'ufboot_data_DS1-8') + '/')
    rootsplit, subsplit = get_support_from_mcmc(taxa, support_trees, support_names)
    topology_model = SBN(taxa, rootsplit, subsplit)
    posterior_trees, posterior_names, posterior_weights = summary(
        dataset, str(data_root / 'raw_data_DS1-8') + '/'
    )
    name = 'tree_{}'.format(tree_id)
    if name not in posterior_trees:
        raise ValueError('Posterior topology {} is missing for {}'.format(name, dataset))
    tree = posterior_trees[name].copy()
    tree_process(tree, BitArray(taxa))
    namenum(tree, taxa)
    return data, taxa, tree, topology_model.rs_map, topology_model.ss_map


def main():
    parser = argparse.ArgumentParser(description='Fit the standalone VBPI-NF branch model on a fixed DS1 topology.')
    parser.add_argument('--dataset', default='DS1')
    parser.add_argument('--tree-id', type=int, default=1)
    parser.add_argument('--all-trees', action='store_true')
    parser.add_argument('--tree-ids', type=int, nargs='+')
    parser.add_argument('--feature-dim', type=int, default=2)
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[100])
    parser.add_argument('--flow-layers', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    parser.add_argument('--log-every', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=100000)
    parser.add_argument('--bound-runs', type=int, default=0)
    parser.add_argument('--branch-samples', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--run-tag', default='figure4_6_nf')
    parser.add_argument('--workdir', type=Path, default=REPO_ROOT / 'results' / 'vbpi_nf' / 'fixed_tree')
    args = parser.parse_args()

    if args.tree_ids:
        tree_ids = args.tree_ids
    elif args.all_trees:
        tree_ids = range(1, 43)
    else:
        tree_ids = [args.tree_id]
    for tree_id in tree_ids:
        data, taxa, tree, rootsplit, subsplit = load_inputs(args.dataset, tree_id)
        output_dir = args.workdir / args.dataset / 'tree_{}'.format(tree_id) / args.run_tag
        output_dir.mkdir(parents=True, exist_ok=False)
        logger = logging.getLogger('vbpi-nf-fixed-tree-{}'.format(tree_id))
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(output_dir / 'training.log')
        logger.addHandler(handler)
        torch.manual_seed(args.seed + tree_id - 1)
        np.random.seed(args.seed + tree_id - 1)
        model = DeepModel(
            len(taxa), rootsplit, subsplit, psp=True, feature_dim=args.feature_dim,
            hidden_sizes=args.hidden_sizes, num_of_layers_nf=args.flow_layers,
            flow_type='realnvp',
        )
        phylo = PHY(data, taxa, np.ones(4) / 4.0, ('JC', 1.0), scale=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        trace = []
        recent = []
        for iteration in range(1, args.iterations + 1):
            log_branch, logq = model([tree] * args.batch_size)
            log_likelihood = torch.stack([phylo.loglikelihood(row, tree) for row in log_branch])
            log_prior = phylo.logprior(log_branch)
            bound = log_likelihood + log_prior - logq
            loss = -bound.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recent.append(float(bound.detach().mean()))
            if iteration % args.log_every == 0 or iteration == args.iterations:
                trace.append({'iteration': iteration, 'lower_bound_mean': float(np.mean(recent))})
                logger.info('iteration=%d lower_bound=%.6f', iteration, trace[-1]['lower_bound_mean'])
                recent.clear()
            if iteration % args.save_every == 0:
                torch.save(model.state_dict(), output_dir / '{}.pt'.format(iteration))

        torch.save(model.state_dict(), output_dir / 'final.pt')
        with (output_dir / 'training_trace.csv').open('w', newline='', encoding='utf-8') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=['iteration', 'lower_bound_mean'])
            writer.writeheader()
            writer.writerows(trace)
        if args.bound_runs:
            bound_samples = []
            with torch.no_grad():
                for _ in range(args.bound_runs):
                    log_branch, logq = model([tree])
                    log_likelihood = phylo.loglikelihood(log_branch[0], tree)
                    log_prior = phylo.logprior(log_branch)[0]
                    bound_samples.append(float(log_likelihood + log_prior - logq[0]))
            np.save(output_dir / 'fixed_tree_lower_bound_samples.npy', np.asarray(bound_samples))
        if args.branch_samples:
            with torch.no_grad():
                branch_samples = model.sample_branch_fixtree(tree, args.branch_samples)
            np.save(output_dir / 'variational_branch_samples.npy', branch_samples.exp().cpu().numpy())
        handler.close()
        logger.removeHandler(handler)
        print(output_dir)


if __name__ == '__main__':
    main()
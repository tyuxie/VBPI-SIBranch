import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _path in (_REPO_ROOT, _THIS_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from copy import deepcopy
from multiprocessing import Pool
from src.dataManipulation import *
from support_preprocessing import filter_subsplit_support_for_sampling
from src.utils import summary, summary_raw, mcmc_treeprob, get_support_from_mcmc, BitArray, tree_process
from src.vbpi import VBPI
import time
import numpy as np
import datetime
import pdb
import torch
import logging


def _normalize_experiment_name(path):
    stem = os.path.splitext(path)[0]
    return stem.replace('\\', '/').strip('/').replace('/', '__')


def resolve_data_sources(args):
    ufboot_support_path = 'data/ufboot_data_DS1-8/'
    ground_truth_path, samp_size = 'data/raw_data_DS1-8/', 750001
    data_path = 'data/hohna_datasets_fasta/' + args.dataset + '.fasta'
    support_dataset = args.dataset
    experiment_name = args.dataset

    if getattr(args, 'input_fasta', None):
        input_fasta = args.input_fasta
        if not os.path.isabs(input_fasta):
            input_fasta = os.path.abspath(input_fasta)
        if not os.path.exists(input_fasta):
            raise FileNotFoundError('Input FASTA not found: {}'.format(input_fasta))

        data_path = input_fasta
        support_dataset = args.support_dataset if getattr(args, 'support_dataset', None) else args.dataset
        experiment_name = getattr(args, 'experiment_name', None) or _normalize_experiment_name(os.path.relpath(input_fasta, os.getcwd()))

        # synthetic L8ALL tree support lives alongside this script, not under data/
        if support_dataset == 'L8ALL':
            ufboot_support_path = os.path.join(_THIS_DIR, 'ufboot') + os.sep

    args.data_path = data_path
    args.support_dataset = support_dataset
    args.experiment_name = experiment_name
    return ufboot_support_path, ground_truth_path, samp_size


def parse_args():
    parser = argparse.ArgumentParser()

    ######### Data arguments
    parser.add_argument('--dataset', required=True, help=' DS1 | DS2 | DS3 | DS4 | DS5 | DS6 | DS7 | DS8 ')
    parser.add_argument('--empFreq', default=False, action='store_true', help='emprical frequence for KL computation')
    parser.add_argument('--input_fasta', type=str, default=None, help='optional absolute or relative FASTA path for an independent experiment run')
    parser.add_argument('--support_dataset', type=str, default=None, help='dataset name used only for loading the shared tree support set')
    parser.add_argument('--experiment_name', type=str, default=None, help='unique experiment tag used to separate result and log folders')


    ######### Model arguments
    parser.add_argument('--hdim', type=int, default=100, help='hidden dimension for node embedding net')
    parser.add_argument('--zdim', type=int, default=50, help='dimension for the latent variable z')
    parser.add_argument('--hL', type=int, default=2, help='number of hidden layers for node embedding net')
    parser.add_argument('--brlen_model', type=str, default='sivi', help='branch length models')
    parser.add_argument('--gnn_type', type=str, default='edge', help='gcn | sage | gin | ggnn')
    parser.add_argument('--aggr', type=str, default='sum', help='sum | mean | max')
    parser.add_argument('--proj', default=False, action='store_true', help='use projection first in SAGEConv')
    parser.add_argument('--test', default=False, action='store_true', help='turn on the test mode')
    parser.add_argument('--date', type=str, default='2022-01-01', help=' 2020-04-01 | 2020-04-02 | ...... ')
    parser.add_argument('--seed', type=int, default=2023)

    ######### Optimizer arguments
    parser.add_argument('--stepszTree', type=float, default=0.001, help=' step size for tree topology parameters ')
    parser.add_argument('--stepszBranch', type=float, default=0.001, help=' stepsz for branch length parameters ')
    parser.add_argument('--maxIter', type=int, default=400000, help=' number of iterations for training, default=400000')
    parser.add_argument('--invT0', type=float, default=0.001, help=' initial inverse temperature for annealing schedule, default=0.001')
    parser.add_argument('--nwarmStart', type=float, default=100000, help=' number of warm start iterations, default=100000')
    parser.add_argument('--nParticle', type=int, default=10, help='number of particles for variational objectives, default=10')
    parser.add_argument('--nz', type=int, default=50, help='batch size for latent variable z, defalut=50')
    parser.add_argument('--ar', type=float, default=0.75, help='step size anneal rate, default=0.75')
    parser.add_argument('--af', type=int, default=20000, help='step size anneal frequency, default=20000')
    parser.add_argument('--tf', type=int, default=1000, help='monitor frequency during training, default=1000')
    parser.add_argument('--sf', type=int, default=100000, help='save frequency, default=100000')
    parser.add_argument('--lbf', type=int, default=5000, help='lower bound test frequency, default=5000')
    parser.add_argument('--workdir', type=str, default='synthetic_exp/results')
    parser.add_argument('--gradMethod', type=str, default='sivi', help=' vimco | rws ')

    args = parser.parse_args()
    return args

def file_prepare(args):
    dataset_folder = args.support_dataset if getattr(args, 'input_fasta', None) else args.dataset
    experiment_folder = args.experiment_name if getattr(args, 'input_fasta', None) else args.dataset
    args.result_folder = args.workdir + '/' + dataset_folder + '/' + experiment_folder + '/' + args.brlen_model

    args.result_folder = args.result_folder + '/' + args.gradMethod + '_' + str(args.nParticle)
    if args.brlen_model == 'gnn':
        args.result_folder = args.result_folder + '_' + args.gnn_type + '_' + args.aggr
    if args.brlen_model == 'sivi':
        args.result_folder = args.result_folder + '_nz_' + str(args.nz) 
    if args.proj:
        args.result_folder = args.result_folder + '_proj'
    args.result_folder = args.result_folder + '_' + args.date
    args.load_from_path = args.result_folder + '/final.pt'
    args.save_to_path = args.result_folder + '/final.pt'
    args.logpath = args.result_folder + '/final.log'

    if not args.test:
        os.makedirs(args.result_folder, exist_ok=False)


def main(args):
    file_prepare(args)
    if args.brlen_model == 'gnn':
        args.nz = None

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(args.logpath)
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)

    if not args.test:
        logger.info('Training with the following settings:')
    # else:
    #     logger.info('Testing with the following settings:')
    for name, value in vars(args).items():
        logger.info('{} : {}'.format(name, value))

    ufboot_support_path, ground_truth_path, samp_size = resolve_data_sources(args)

    ###### Load Data
    print('\nLoading Data set: {} ......'.format(args.experiment_name))
    run_time = -time.time()

    support_hpd = 1.0 if args.support_dataset == 'L8ALL' else 0.95
    tree_dict_support, tree_names_support = summary_raw(args.support_dataset, ufboot_support_path, hpd=support_hpd)

    data, taxa = loadData(args.data_path, 'fasta')

    run_time += time.time()
    print('Support loaded in {:.1f} seconds'.format(run_time))

    if args.empFreq:
        print('\nLoading empirical posterior estimates ......')
        run_time = -time.time()
        tree_dict_total, tree_names_total, tree_wts_total = summary(args.support_dataset, ground_truth_path, samp_size=samp_size)
        emp_tree_freq = {tree_dict_total[tree_name]:tree_wts_total[i] for i, tree_name in enumerate(tree_names_total)}
        run_time += time.time()
        print('Empirical estimates from MrBayes loaded in {:.1f} seconds'.format(run_time))
    else:
        emp_tree_freq = None

    rootsplit_supp_dict, subsplit_supp_dict = get_support_from_mcmc(taxa, tree_dict_support, tree_names_support)
    if args.support_dataset == 'L8ALL':
        subsplit_supp_dict = filter_subsplit_support_for_sampling(taxa, subsplit_supp_dict)
    del tree_dict_support, tree_names_support

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = VBPI(taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden=np.ones(4)/4., subModel=('JC', 1.0),
                    emp_tree_freq=emp_tree_freq, latent_dim=args.zdim, hidden_dim=args.hdim, num_layers=args.hL, branch_model=args.brlen_model, gnn_type=args.gnn_type, aggr=args.aggr, project=args.proj)

        
    if not args.test:
        logger.info('Parameter Info:')
        for param in model.parameters():
            logger.info(param.dtype)
            logger.info(param.size())

        logger.info('\nThis is a version of \'output mean std of each edge, and the aggregating them\' ')
        logger.info('\nVBPI running, results will be saved to: {}\n'.format(args.save_to_path))
        test_lb, test_kl_div, ts = model.learn({'tree':args.stepszTree,'branch':args.stepszBranch}, args.maxIter, test_freq=args.tf, lb_test_freq=args.lbf, n_particles=args.nParticle, anneal_freq=args.af, save_freq=args.sf, init_inverse_temp=args.invT0, warm_start_interval=args.nwarmStart, batch_size_z=args.nz if args.nz==None else args.nz + 1, method=args.gradMethod, save_to_path=args.save_to_path, logger=logger)

        sampled_output_path = args.save_to_path.replace('.pt', '_sampled_1000_trees.newick')
        logger.info('Sampling 1000 trees with branch lengths to: {}'.format(sampled_output_path))
        sampled_trees = []
        with torch.no_grad():
            for i in range(1000):
                tree = model.tree_model.sample_tree()
                namenum(tree, taxa)
                if args.brlen_model == 'iwhvi':
                    samp_log_branch, _, _ = model.branch_model([tree], 50)
                else:
                    samp_log_branch, _ = model.branch_model([tree])
                log_branch_vec = samp_log_branch[0]
                for node in tree.traverse('postorder'):
                    if not node.is_root():
                        node.dist = float(torch.exp(log_branch_vec[node.name]).item())
                    else:
                        node.dist = 0.0
                sampled_trees.append(tree.write(format=5))
                if (i + 1) % 100 == 0:
                    logger.info('Sampled {} / 1000 trees'.format(i + 1))

        with open(sampled_output_path, 'w') as f:
            for newick in sampled_trees:
                f.write(newick + '\n')
        logger.info('Saved sampled trees to: {}'.format(sampled_output_path))
                
        np.save(args.save_to_path.replace('.pt', '_test_lb.npy'), test_lb)
        np.save(args.save_to_path.replace('.pt', '_times.npy'), ts)
        if args.empFreq:
            np.save(args.save_to_path.replace('.pt', '_kl_div.npy'), test_kl_div)
    else:
        logger.info('\nThis is a version of \'output mean std of each edge, and the aggregating them\' ')
        logger.info('Loading parameters from: {}\n'.format(args.load_from_path))
        model.load_from(args.load_from_path)
        
        if args.brlen_model in ['sivi', 'iwhvi']:
            batch_size_z = 1000 + 1
        elif args.brlen_model == 'gnn':
            batch_size_z = None
        
        if not args.empFreq:
            logger.info('Computing one sample lower bounds\n')
            ELBOs = torch.stack([model.fast_lower_bound(total_runs=10000, batch_size_z=batch_size_z) for i in range(100)])

            lower_bound_1_sample = ELBOs.reshape((1000, 1000)).mean(-1).numpy()
            np.save(args.load_from_path.replace('.pt', '_lower_bound_1.npy'), lower_bound_1_sample)
            logger.info('1-sample lower bound. MEAN:{} STD:{}'.format(np.mean(lower_bound_1_sample), np.std(lower_bound_1_sample)))

            lower_bound_10_sample = torch.logsumexp(ELBOs.reshape((100, 1000, 10)) - np.log(10), dim=-1).mean(-1).numpy()
            np.save(args.load_from_path.replace('.pt', '_lower_bound_10.npy'), lower_bound_10_sample)
            logger.info('10-sample lower bound. MEAN:{} STD:{}'.format(np.mean(lower_bound_10_sample), np.std(lower_bound_10_sample)))

            marginal_likelihood_est = torch.logsumexp(ELBOs.reshape((1000, 1000)) - np.log(1000), dim=-1).numpy()
            np.save(args.load_from_path.replace('.pt', '_marginal_likelihood_est.npy'), marginal_likelihood_est)
            logger.info('marginal likelihood. MEAN:{} STD:{}'.format(np.mean(marginal_likelihood_est), np.std(marginal_likelihood_est)))
        else:
            tree_ci_index = np.argsort(tree_wts_total)[::-1]
            logger.info('Computing 95% confidence interval tree lower bound\n')
            lower_bound_ci = []
            toBitArr = BitArray(taxa)
            for i in tree_ci_index[:42]:
                test_tree = tree_dict_total[tree_names_total[i]].copy()
                tree_process(test_tree, toBitArr)
                lower_bound_ci.append(model.tree_lower_bound(test_tree, n_runs=10000))
                logger.info('tree {} ELBO: {}'.format(i, lower_bound_ci[-1]))
            np.save(args.save_to_path.replace('.pt', '_tree_lower_bound.npy'), lower_bound_ci)

if __name__ == '__main__':
    args = parse_args()
    main(args)
import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
from tqdm import tqdm
import numpy as np
from src.utils import namenum
from src.gnn_branchModel import GNNModel
from src.sivi_branchModel import SIVIModel, IWHVIModel
from src.vector_sbnModel import SBN
from src.phyloModel import PHY


class VBPI(nn.Module):
    EPS = np.finfo(float).eps
    def __init__(self, taxa, rootsplit_supp_dict, subsplit_supp_dict, data, pden, subModel, emp_tree_freq=None,
                 scale=0.1,latent_dim=50, hidden_dim=100, num_layers=1, branch_model='gnn', gnn_type='gcn', aggr='sum', project=False):
        super().__init__()
        self.taxa, self.emp_tree_freq = taxa, emp_tree_freq
        if emp_tree_freq:
            self.trees, self.emp_freqs = zip(*emp_tree_freq.items())
            self.emp_freqs = np.array(self.emp_freqs)
            self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        
        self.ntips = len(data)
        self.scale = scale
        self.phylo_model = PHY(data, taxa, pden, subModel, scale=scale)
        self.log_p_tau = - np.sum(np.log(np.arange(3, 2*self.ntips-3, 2)))
        
        self.tree_model = SBN(taxa, rootsplit_supp_dict, subsplit_supp_dict)
        self.rs_embedding_map, self.ss_embedding_map = self.tree_model.rs_map, self.tree_model.ss_map     
        self.branch_type = branch_model
        if branch_model == 'gnn':
            self.branch_model = GNNModel(self.ntips, hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr, project=project)
        elif branch_model == 'sivi':
            self.branch_model = SIVIModel(self.ntips, latent_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr)
        elif branch_model == 'iwhvi':
            self.branch_model = IWHVIModel(self.ntips, latent_dim, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr)
        else:
            raise NotImplementedError
        
        torch.set_num_threads(1)

    def load_from(self, state_dict_path):
        with torch.no_grad():
            self.load_state_dict(torch.load(state_dict_path))
            self.eval()
            self.tree_model.update_CPDs()
                
    def kl_div(self):
        kl_div = 0.0
        for tree, wt in self.emp_tree_freq.items():
            kl_div += wt * np.log(max(np.exp(self.tree_model.loglikelihood(tree)), self.EPS))
        kl_div = self.negDataEnt - kl_div
        return kl_div
    
    def logq_tree(self, tree):
        return self.tree_model(tree)
    
    def lower_bound(self, n_particles=1, n_runs=1000, batch_size_z=None):
        lower_bounds = []
        with torch.no_grad():
            for run in range(n_runs):
                samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
                [namenum(tree, self.taxa) for tree in samp_trees]    
                if self.branch_type == 'iwhvi':
                    samp_log_branch, logq_branch, logq_reverse = self.branch_model(samp_trees, batch_size_z)
                    logq_branch = torch.logsumexp(logq_branch -  logq_reverse - math.log(batch_size_z), dim=1)
                else:
                    samp_log_branch, logq_branch = self.branch_model(samp_trees, batch_size_z)
                    if batch_size_z is not None:
                        logq_branch = torch.logsumexp(logq_branch - math.log(batch_size_z), dim=1)

                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])       
                lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0))            
            
            lower_bound = torch.stack(lower_bounds).mean()
            
        return lower_bound.item()

    def fast_lower_bound(self, total_runs, batch_size_z=None):
        ELBOs = []
        with torch.no_grad():
            for run in range(total_runs):
                samp_trees = [self.tree_model.sample_tree()]
                [namenum(tree, self.taxa) for tree in samp_trees]    
                if self.branch_type == 'iwhvi':
                    samp_log_branch, logq_branch, logq_reverse = self.branch_model(samp_trees, batch_size_z)
                    logq_branch = torch.logsumexp(logq_branch -  logq_reverse - math.log(batch_size_z), dim=1)
                else:
                    samp_log_branch, logq_branch = self.branch_model(samp_trees, batch_size_z)
                    if batch_size_z is not None:
                        logq_branch = torch.logsumexp(logq_branch - math.log(batch_size_z), dim=1)

                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])       
                ELBOs.append(torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau, 0))            
            
            ELBOs = torch.stack(ELBOs)
            
        return ELBOs
    
    def tree_lower_bound(self, tree, n_particles=1, n_runs=1000, batch_size_z=1000, name_to_num=True):
        lower_bounds = []
        if name_to_num:
            namenum(tree, self.taxa)
        with torch.no_grad():
            for run in range(n_runs):
                test_trees = [tree for particle in range(n_particles)]
                if self.branch_type == 'iwhvi':
                    samp_log_branch, logq_branch, logq_reverse = self.branch_model(test_trees, batch_size_z)
                    logq_branch = torch.logsumexp(logq_branch -  logq_reverse - math.log(batch_size_z), dim=1)
                else:
                    samp_log_branch, logq_branch = self.branch_model(test_trees, batch_size_z)
                    if batch_size_z is not None:
                        logq_branch = torch.logsumexp(logq_branch - math.log(batch_size_z), dim=1)

                logll = torch.stack([self.phylo_model.loglikelihood(log_branch, test_tree) for log_branch, test_tree in zip(*[samp_log_branch, test_trees])])
                logp_prior = self.phylo_model.logprior(samp_log_branch)
                lower_bounds.append(torch.logsumexp(logll + logp_prior - logq_branch, 0) - math.log(n_particles))
                
            lower_bound = torch.stack(lower_bounds).mean()

        return lower_bound.item()

    def vimco_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        
        samp_log_branch, logq_branch = self.branch_model(samp_trees)
        
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree - logq_branch
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)
        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)
        
        
    def rws_lower_bound(self, inverse_temp=1.0, n_particles=10):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        
        samp_log_branch, logq_branch = self.branch_model(samp_trees)
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree.detach() - logq_branch
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        snis_wts = torch.softmax(l_signal, dim=0)
        rws_fake_term = torch.sum(snis_wts.detach() * logq_tree, dim=0)
        return temp_lower_bound, rws_fake_term, lower_bound, torch.max(logll)
        
    
    def sivi_lower_bound(self, inverse_temp=1.0, n_particles=10, batch_size_z=50):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]
        
        samp_log_branch, logq_branch_batch = self.branch_model(samp_trees, batch_size_z)
        logq_branch = torch.logsumexp(logq_branch_batch - math.log(batch_size_z), dim=1)
        
        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree - logq_branch
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)
        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)
        
    def iwhvi_lower_bound(self, inverse_temp=1.0, n_particles=10, batch_size_z=50):
        samp_trees = [self.tree_model.sample_tree() for particle in range(n_particles)]
        [namenum(tree, self.taxa) for tree in samp_trees]

        samp_log_branch, logq_branch_batch, logq_reverse_batch = self.branch_model(samp_trees, batch_size_z)
        logq_branch = torch.logsumexp(logq_branch_batch - logq_reverse_batch - math.log(batch_size_z), dim=1)

        logll = torch.stack([self.phylo_model.loglikelihood(log_branch, tree) for log_branch, tree in zip(*[samp_log_branch, samp_trees])])
        logp_prior = self.phylo_model.logprior(samp_log_branch)
        logp_joint = inverse_temp * logll + logp_prior
        logq_tree = torch.stack([self.logq_tree(tree) for tree in samp_trees])
        lower_bound = torch.logsumexp(logll + logp_prior - logq_tree - logq_branch + self.log_p_tau - math.log(n_particles), 0)
        
        l_signal = logp_joint - logq_tree - logq_branch
        mean_exclude_signal = (torch.sum(l_signal) - l_signal) / (n_particles-1.)
        control_variates = torch.logsumexp(l_signal.view(-1,1).repeat(1, n_particles) - l_signal.diag() + mean_exclude_signal.diag() - math.log(n_particles), dim=0)
        temp_lower_bound = torch.logsumexp(l_signal - math.log(n_particles), dim=0)
        vimco_fake_term = torch.sum((temp_lower_bound - control_variates).detach() * logq_tree, dim=0)
        return temp_lower_bound, vimco_fake_term, lower_bound, torch.max(logll)

    def learn(self, stepsz, maxiter=100000, test_freq=1000, lb_test_freq=5000, anneal_freq=20000, anneal_rate=0.75, n_particles=10,
              batch_size_z=None, save_freq=100000, init_inverse_temp=0.001, warm_start_interval=50000, method='vimco', save_to_path=None, logger=None):
        lbs, lls = [], []
        test_kl_div, test_lb, ts = [], [], []
        
        if not isinstance(stepsz, dict):
            stepsz = {'tree': stepsz, 'branch': stepsz}
        
        if method in ['sivi', 'iwhvi']: assert batch_size_z is not None, f"please specify batch_size_z!!!"
        
        optimizer = torch.optim.Adam([
                    {'params': self.tree_model.parameters(), 'lr':stepsz['tree']},
                    {'params': self.branch_model.parameters(), 'lr': stepsz['branch']}
                ])
        run_time = -time.time()
        for it in tqdm(range(1, maxiter+1)):
            inverse_temp = min(1., init_inverse_temp + it * 1.0/warm_start_interval)
            if method == 'vimco':
                temp_lower_bound, vimco_fake_term, lower_bound, logll = self.vimco_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - vimco_fake_term
            elif method == 'rws':
                temp_lower_bound, rws_fake_term, lower_bound, logll = self.rws_lower_bound(inverse_temp, n_particles)
                loss = - temp_lower_bound - rws_fake_term
            elif method== 'sivi':
                temp_lower_bound, vimco_fake_term, lower_bound, logll = self.sivi_lower_bound(inverse_temp, n_particles, batch_size_z)
                loss = - temp_lower_bound - vimco_fake_term
            elif method== 'iwhvi':
                temp_lower_bound, vimco_fake_term, lower_bound, logll = self.iwhvi_lower_bound(inverse_temp, n_particles, batch_size_z)
                loss = - temp_lower_bound - vimco_fake_term
            else:
                raise NotImplementedError

            lbs.append(lower_bound.item())
            lls.append(logll.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            self.tree_model.update_CPDs()
            
            if it % test_freq == 0:
                run_time += time.time()
                if self.emp_tree_freq:
                    test_kl_div.append(self.kl_div())
                    msg = 'Iter {}:({:.4f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f} | KL: {:.6f}'.format(it, run_time, np.mean(lbs), np.max(lls), test_kl_div[-1])
                    logger.info(msg)
                    tqdm.write(msg)
                else:
                    msg = 'Iter {}:({:.4f}s) Lower Bound: {:.4f} | Loglikelihood: {:.4f}'.format(it, run_time, np.mean(lbs), np.max(lls))
                    logger.info(msg)
                    tqdm.write(msg)
                ts.append(run_time)
                if it % lb_test_freq == 0:
                    run_time = -time.time()
                    test_lb.append([self.lower_bound(n_particles=1, n_runs=1000, batch_size_z=batch_size_z), self.lower_bound(n_particles=10, n_runs=100, batch_size_z=batch_size_z)])
                    run_time += time.time()
                    msg = '>>> Iter {}:({:.4f}s) Test Lower Bound 1: {:.4f} Test Lower Bound 10: {:.4f}'.format(it, run_time, test_lb[-1][0], test_lb[-1][1])
                    logger.info(msg)
                    tqdm.write(msg)
                    
                run_time = -time.time()
                lbs, lls = [], []
            
            if it % anneal_freq == 0:
                for g in optimizer.param_groups:
                    g['lr'] *= anneal_rate
            
            if it % save_freq == 0:
                torch.save(self.state_dict(), save_to_path.replace('final', str(it)))
        if save_to_path is not None:
            torch.save(self.state_dict(), save_to_path)
            
        return test_lb, test_kl_div, ts
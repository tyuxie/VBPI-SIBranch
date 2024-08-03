import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

from src.gnnModels_slim import GNNStack, GatedGraphConv, IDConv


class MeanStdPooling(nn.Module):
    def __init__(self, in_channels, hidden_channels, bias=True, **kwargs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_channels, hidden_channels, bias=bias),
                                 nn.ELU(),
                                 nn.Linear(hidden_channels, hidden_channels, bias=bias),
                                 nn.ELU(),)       
    
    def forward(self, x, parent_index):
        mean_std = self.net(x)           
        mean_std = torch.max(mean_std[:-1], mean_std[parent_index])                
        
        return mean_std
        
        
class GNNEncoder(nn.Module):
    def __init__(self, ntips, hidden_dim=100, num_layers=1, gnn_type='gcn', aggr='sum', bias=True, **kwargs):
        super().__init__()
        self.ntips = ntips
        self.leaf_features = torch.eye(self.ntips)
        
        if gnn_type == 'identity':
            self.gnn = IDConv()
        elif gnn_type != 'ggnn':
            self.gnn = GNNStack(self.ntips, hidden_dim, num_layers=num_layers, bias=bias, gnn_type=gnn_type, aggr=aggr)
        else:
            self.gnn = GatedGraphConv(hidden_dim, num_layers=num_layers, bias=bias)
            
        self.mean_std_net = MeanStdPooling(hidden_dim, hidden_dim, bias=bias)


    def node_embedding(self, tree):
        for node in tree.traverse('postorder'):
            if node.is_leaf():
                node.c = 0
                node.d = self.leaf_features[node.name]
            else:
                child_c, child_d = 0., 0.
                for child in node.children:
                    child_c += child.c
                    child_d += child.d
                node.c = 1./(3. - child_c)
                node.d = node.c * child_d
        
        node_features, node_idx_list, edge_index = [], [], []            
        for node in tree.traverse('preorder'):
            neigh_idx_list = []
            if not node.is_root():
                node.d = node.c * node.up.d + node.d
                neigh_idx_list.append(node.up.name)
                
                if not node.is_leaf():
                    neigh_idx_list.extend([child.name for child in node.children])
                else:
                    neigh_idx_list.extend([-1, -1])              
            else:
                neigh_idx_list.extend([child.name for child in node.children])
            
            edge_index.append(neigh_idx_list)                
            node_features.append(node.d)
            node_idx_list.append(node.name)
        
        branch_idx_map = torch.sort(torch.LongTensor(node_idx_list), dim=0, descending=False)[1]
        edge_index = torch.LongTensor(edge_index)
        
        return torch.index_select(torch.stack(node_features), 0, branch_idx_map), edge_index[branch_idx_map]
    
    
    def forward(self, tree, **kwargs):
        node_features, edge_index = self.node_embedding(tree)
        node_features = self.gnn(node_features, edge_index)

        return self.mean_std_net(node_features, edge_index[:-1, 0])


class SIVIModel(nn.Module):
    def __init__(self, ntips, latent_dim=50, hidden_dim=100, num_layers=1, gnn_type='gcn', aggr='sum', bias=True, **kwargs):
        super().__init__()
        self.ntips, self.latent_dim = ntips, latent_dim
        self.ndim = 2*self.ntips - 3
        self.encoder = GNNEncoder(ntips, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr, bias=bias)
        self.readout = nn.Sequential(nn.Linear(hidden_dim+latent_dim, hidden_dim, bias=bias),
                                        nn.ELU(),
                                        nn.Linear(hidden_dim, 2, bias=bias),)
        
    def forward(self, tree_list, batch_size_z, log_std_min=-3.):
        samp_z = torch.randn(len(tree_list), batch_size_z, self.ndim, self.latent_dim)

        mean_std = torch.stack([self.encoder(tree) for tree in tree_list])
        mean_std = self.readout(torch.cat((mean_std.unsqueeze(1).repeat(1, batch_size_z, 1, 1), samp_z), dim=-1))
        mean, log_std = mean_std[:,:,:,0], mean_std[:,:,:,1]
        log_std = log_std.clamp(log_std_min)
        samp_x_raw = torch.randn(len(tree_list), self.ndim)
        samp_log_branch = samp_x_raw * log_std[:,0,:].exp() + mean[:,0,:] - 2.0
        logq_branch_batch = -0.5*torch.sum(math.log(2*math.pi) + ((samp_log_branch.unsqueeze(1) - mean + 2.0)/log_std.exp())**2, dim=-1) - torch.sum(log_std, dim=-1)
        
        return samp_log_branch, logq_branch_batch

    def sample_branch(self, tree, batchsize, log_std_min=-3.):
        samp_z = torch.randn(batchsize, self.ndim, self.latent_dim)
        mean_std = self.encoder(tree)
        mean_std = self.readout(torch.cat([mean_std.unsqueeze(0).repeat(batchsize, 1, 1), samp_z], dim=-1))
        mean, log_std = mean_std[:,:,0], mean_std[:,:,1]
        log_std = log_std.clamp(log_std_min)
        samp_log_branch = torch.randn_like(log_std) * log_std.exp() + mean - 2.0
        return samp_log_branch


class IWHVIModel(nn.Module):
    def __init__(self, ntips, latent_dim=50, hidden_dim=100, num_layers=1, gnn_type='gcn', aggr='sum', bias=True, **kwargs):
        super().__init__()
        self.ntips, self.latent_dim = ntips, latent_dim
        self.ndim = 2*self.ntips - 3
        self.encoder = GNNEncoder(ntips, hidden_dim=hidden_dim, num_layers=num_layers, gnn_type=gnn_type, aggr=aggr, bias=bias)
        self.readout = nn.Sequential(nn.Linear(hidden_dim+latent_dim, hidden_dim, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(hidden_dim, 2, bias=bias),)
        self.reverse = nn.Sequential(nn.Linear(hidden_dim+1, hidden_dim, bias=bias),
                                     nn.ELU(),
                                     nn.Linear(hidden_dim, 2*latent_dim, bias=bias),)
        
    def forward(self, tree_list, batch_size_z, log_std_min=-3.):
        samp_z = torch.randn(len(tree_list), self.ndim, self.latent_dim) 
        edge_info = torch.stack([self.encoder(tree) for tree in tree_list]) 
        mean_std = self.readout(torch.cat((edge_info, samp_z), dim=-1))
        mean, log_std = mean_std[:,:,0], mean_std[:,:,1]
        log_std = log_std.clamp(log_std_min)
        samp_x_raw = torch.randn(len(tree_list), self.ndim)
        samp_log_branch = samp_x_raw * log_std.exp() + mean - 2.0 

        reverse_mean_std = self.reverse(torch.cat((edge_info, samp_log_branch.unsqueeze(-1)), dim=-1))
        reverse_mean, reverse_log_std = reverse_mean_std[:,:,:self.latent_dim], reverse_mean_std[:,:,self.latent_dim:] 
        reverse_log_std = reverse_log_std.clamp(log_std_min)
        reverse_samp_z = reverse_mean.unsqueeze(1) + torch.randn(len(tree_list), batch_size_z-1, self.ndim, self.latent_dim) * reverse_log_std.unsqueeze(1).exp()

        all_samp_z = torch.cat((samp_z.unsqueeze(1), reverse_samp_z), dim=1) 
        all_mean_std = self.readout(torch.cat((edge_info.unsqueeze(1).repeat(1, batch_size_z, 1, 1), all_samp_z), dim=-1))
        all_mean, all_log_std = all_mean_std[:,:,:,0], all_mean_std[:,:,:,1]
        all_log_std = all_log_std.clamp(log_std_min)
        logq_z_batch = - 0.5 * torch.sum(all_samp_z**2 + math.log(2*math.pi), dim=(-1,-2))
        logq_branch_batch = -0.5*torch.sum(math.log(2*math.pi) + ((samp_log_branch.unsqueeze(1) - all_mean + 2.0)/all_log_std.exp())**2, dim=-1) - torch.sum(all_log_std, dim=-1)
        logq_reverse_batch = -0.5*torch.sum(math.log(2*math.pi) + ((all_samp_z - reverse_mean.unsqueeze(1))/reverse_log_std.unsqueeze(1).exp())**2, dim=(-1,-2)) - torch.sum(reverse_log_std.unsqueeze(1), dim=(-1,-2))
            
        return samp_log_branch, logq_branch_batch + logq_z_batch, logq_reverse_batch
    

    def sample_branch(self, tree, batchsize, log_std_min=-3.):
        samp_z = torch.randn(batchsize, self.ndim, self.latent_dim)
        mean_std = self.encoder(tree)
        mean_std = self.readout(torch.cat([mean_std.unsqueeze(0).repeat(batchsize, 1, 1), samp_z], dim=-1))
        mean, log_std = mean_std[:,:,0], mean_std[:,:,1]
        log_std = log_std.clamp(log_std_min)
        samp_log_branch = torch.randn_like(log_std) * log_std.exp() + mean - 2.0
        return samp_log_branch
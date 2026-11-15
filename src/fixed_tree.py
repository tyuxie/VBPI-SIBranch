import math

import numpy as np
import torch
import torch.nn as nn

from src.phyloModel import PHY


class _Encoder(nn.Module):
    def __init__(self, ndim, latent_dim, hidden_units, activation='elu'):
        super().__init__()
        activation_fn = nn.ELU if activation == 'elu' else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            activation_fn(),
            nn.Linear(hidden_units, hidden_units),
            activation_fn(),
            nn.Linear(hidden_units, 2 * ndim),
        )

    def forward(self, latent):
        return torch.chunk(self.net(latent), 2, dim=-1)


class _Decoder(nn.Module):
    def __init__(self, ndim, latent_dim, hidden_units, activation='elu'):
        super().__init__()
        activation_fn = nn.ELU if activation == 'elu' else nn.ReLU
        self.net = nn.Sequential(
            nn.Linear(ndim, hidden_units),
            activation_fn(),
            nn.Linear(hidden_units, hidden_units),
            activation_fn(),
            nn.Linear(hidden_units, 2 * latent_dim),
        )

    def forward(self, log_branch):
        return torch.chunk(self.net(log_branch), 2, dim=-1)


class _FixedSIVI(nn.Module):
    def __init__(self, ndim, latent_dim, hidden_units, activation='elu'):
        super().__init__()
        self.ndim = ndim
        self.latent_dim = latent_dim
        self.encoder = _Encoder(ndim, latent_dim, hidden_units, activation)

    def forward(self, batch_size, z_batch_size):
        first_z = torch.randn(batch_size, self.latent_dim)
        mean, log_std = self.encoder(first_z)
        log_std = log_std.clamp(min=-3.0)
        base = torch.randn(batch_size, self.ndim)
        log_branch = base * log_std.exp() + mean - 2.0
        logq_first = (-0.5 * (math.log(2 * math.pi) + base.square())).sum(-1) - log_std.sum(-1)

        other_z = torch.randn(z_batch_size, self.latent_dim)
        other_mean, other_log_std = self.encoder(other_z)
        other_log_std = other_log_std.clamp(min=-3.0)
        logq_mix = -0.5 * (
            math.log(2 * math.pi)
            + ((log_branch.unsqueeze(1) - other_mean.unsqueeze(0) + 2.0) / other_log_std.exp().unsqueeze(0)).square()
        ).sum(-1) - other_log_std.sum(-1).unsqueeze(0)
        return log_branch, logq_first, logq_mix

    def sample(self, n_samples):
        latent = torch.randn(n_samples, self.latent_dim)
        mean, log_std = self.encoder(latent)
        return torch.randn(n_samples, self.ndim) * log_std.clamp(min=-3.0).exp() + mean - 2.0


class _FixedIWHVI(nn.Module):
    def __init__(self, ndim, latent_dim, hidden_units, activation='elu'):
        super().__init__()
        self.ndim = ndim
        self.latent_dim = latent_dim
        self.encoder = _Encoder(ndim, latent_dim, hidden_units, activation)
        self.decoder = _Decoder(ndim, latent_dim, hidden_units, activation)

    def forward(self, batch_size, z_batch_size):
        latent = torch.randn(batch_size, self.latent_dim)
        mean, log_std = self.encoder(latent)
        log_std = log_std.clamp(min=-3.0)
        base = torch.randn(batch_size, self.ndim)
        log_branch = base * log_std.exp() + mean - 2.0
        reverse_mean, reverse_log_std = self.decoder(log_branch)
        reverse_log_std = reverse_log_std.clamp(min=-3.0)
        reverse_latent = reverse_mean.unsqueeze(1) + torch.randn(batch_size, z_batch_size, self.latent_dim) * reverse_log_std.unsqueeze(1).exp()
        all_latent = torch.cat((latent.unsqueeze(1), reverse_latent), dim=1)
        all_mean, all_log_std = self.encoder(all_latent.reshape(-1, self.latent_dim))
        all_mean = all_mean.reshape(batch_size, z_batch_size + 1, self.ndim)
        all_log_std = all_log_std.reshape(batch_size, z_batch_size + 1, self.ndim).clamp(min=-3.0)
        logq_latent = -0.5 * (all_latent.square() + math.log(2 * math.pi)).sum(-1)
        logq_branch = -0.5 * (
            math.log(2 * math.pi)
            + ((log_branch.unsqueeze(1) - all_mean + 2.0) / all_log_std.exp()).square()
        ).sum(-1) - all_log_std.sum(-1)
        logq_reverse = -0.5 * (
            math.log(2 * math.pi)
            + ((all_latent - reverse_mean.unsqueeze(1)) / reverse_log_std.unsqueeze(1).exp()).square()
        ).sum(-1) - reverse_log_std.sum(-1).unsqueeze(1)
        return log_branch, logq_branch + logq_latent, logq_reverse

    def sample(self, n_samples):
        latent = torch.randn(n_samples, self.latent_dim)
        mean, log_std = self.encoder(latent)
        return torch.randn(n_samples, self.ndim) * log_std.clamp(min=-3.0).exp() + mean - 2.0


class _FixedLogNormal(nn.Module):
    def __init__(self, ndim):
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(ndim))
        self.log_std = nn.Parameter(torch.zeros(ndim))

    def forward(self, batch_size):
        log_std = self.log_std.clamp(min=-3.0)
        base = torch.randn(batch_size, self.mean.numel())
        log_branch = base * log_std.exp() + self.mean - 2.0
        logq = (-0.5 * (math.log(2 * math.pi) + base.square())).sum(-1) - log_std.sum()
        return log_branch, logq

    def sample(self, n_samples):
        return torch.randn(n_samples, self.mean.numel()) * self.log_std.clamp(min=-3.0).exp() + self.mean - 2.0


class FixedTreeVBPI(nn.Module):
    def __init__(self, taxa, tree, data, branch_model, latent_dim=50, hidden_units=512, activation='elu'):
        super().__init__()
        self.taxa = taxa
        self.tree = tree
        self.branch_type = branch_model
        if activation not in ('elu', 'relu'):
            raise ValueError('Unsupported activation: {}'.format(activation))
        self.phylo_model = PHY(data, taxa, np.ones(4) / 4.0, ('JC', 1.0), scale=0.1)
        ndim = 2 * len(taxa) - 3
        if branch_model == 'sivi':
            self.branch_model = _FixedSIVI(ndim, latent_dim, hidden_units, activation)
        elif branch_model == 'iwhvi':
            self.branch_model = _FixedIWHVI(ndim, latent_dim, hidden_units, activation)
        elif branch_model == 'lognorm':
            self.branch_model = _FixedLogNormal(ndim)
        else:
            raise ValueError('Unsupported fixed-tree model: {}'.format(branch_model))

    def _bound(self, batch_size, z_batch_size):
        if self.branch_type == 'sivi':
            log_branch, logq_first, logq_mix = self.branch_model(batch_size, z_batch_size - 1)
            logq = torch.logsumexp(torch.cat((logq_first.unsqueeze(-1), logq_mix), dim=-1) - math.log(z_batch_size), dim=-1)
        elif self.branch_type == 'iwhvi':
            log_branch, logq, logq_reverse = self.branch_model(batch_size, z_batch_size - 1)
            logq = torch.logsumexp(logq - logq_reverse - math.log(z_batch_size), dim=-1)
        else:
            log_branch, logq = self.branch_model(batch_size)
        log_likelihood = torch.stack([self.phylo_model.loglikelihood(row, self.tree) for row in log_branch])
        log_prior = self.phylo_model.logprior(log_branch)
        return log_likelihood + log_prior - logq, torch.max(log_likelihood)

    def fit(self, iterations, batch_size, z_batch_size, learning_rate, log_every, save_every, output_dir):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        lower_bounds = []
        max_log_likelihoods = []
        records = []
        for iteration in range(1, iterations + 1):
            inverse_temperature = 1.0
            if self.branch_type == 'iwhvi':
                log_branch, logq_branch, logq_reverse = self.branch_model(batch_size, z_batch_size - 1)
                logq = torch.logsumexp(logq_branch - logq_reverse - math.log(z_batch_size), dim=-1)
            elif self.branch_type == 'sivi':
                log_branch, logq_first, logq_mix = self.branch_model(batch_size, z_batch_size - 1)
                logq = torch.logsumexp(torch.cat((logq_first.unsqueeze(-1), logq_mix), dim=-1) - math.log(z_batch_size), dim=-1)
            else:
                log_branch, logq = self.branch_model(batch_size)
            log_likelihood = torch.stack([self.phylo_model.loglikelihood(row, self.tree) for row in log_branch])
            log_prior = self.phylo_model.logprior(log_branch)
            bound = log_likelihood + log_prior - logq
            loss = -(inverse_temperature * log_likelihood + log_prior - logq).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lower_bounds.extend(bound.detach().cpu().tolist())
            max_log_likelihoods.append(float(log_likelihood.detach().max()))
            if iteration % log_every == 0 or iteration == iterations:
                records.append({'iteration': iteration, 'lower_bound_mean': float(np.mean(lower_bounds)), 'max_log_likelihood': float(np.max(max_log_likelihoods))})
                lower_bounds.clear()
                max_log_likelihoods.clear()
            if iteration % save_every == 0:
                torch.save(self.state_dict(), output_dir / '{}.pt'.format(iteration))
        torch.save(self.state_dict(), output_dir / 'final.pt')
        return records

    def lower_bound_samples(self, n_runs, z_batch_size):
        values = []
        with torch.no_grad():
            for _ in range(n_runs):
                bound, _ = self._bound(1, z_batch_size)
                values.append(float(bound[0]))
        return np.asarray(values)

    def branch_samples(self, n_samples):
        with torch.no_grad():
            samples = self.branch_model.sample(n_samples)
            return samples.exp().cpu().numpy()
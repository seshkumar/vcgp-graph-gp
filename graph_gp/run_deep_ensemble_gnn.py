"""Deep Ensemble of Graph Neural Networks with mean+variance heads.

Baseline following the Deep Ensembles recipe
(Lakshminarayanan, Pritzel, Blundell, 2017): train M independent GNNs,
each outputting a per-node Gaussian (mean mu_i, variance sigma_i^2),
using the Gaussian NLL loss. At test time the ensemble's predictive
distribution is itself Gaussian with mean = mean of means and variance
= mean of variances + variance of means.

Each GNN is a 3-layer GCN with hidden dimension 64, trained for 300
epochs on the training-node labels. One node feature per sensor --
the auxiliary field d_i (so the GNN sees the same input information
as the lgamma family). Input also includes the log-degree.

Output: graph_gp/results/results_degnn_{dataset}_seed{seed}.json

Usage:
    python graph_gp/run_deep_ensemble_gnn.py --dataset metrla --seed 1729
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')

from torch_geometric.nn import GCNConv

from run_all_methods import (
    load_caweather,
    load_laqn,
    load_metrla,
    load_pemsbay,
    load_pm25,
)

# run_all_methods sets default dtype to float64 on import; override after
# so the GNN uses float32 (PyG layers expect it by default).
torch.set_default_dtype(torch.float32)


M_MODELS = 5
HIDDEN = 64
EPOCHS = 300
LR = 1e-2
WEIGHT_DECAY = 5e-4


class GCNMeanVar(torch.nn.Module):
    """3-layer GCN with per-node mean + log-variance heads."""

    def __init__(self, in_dim, hidden=HIDDEN):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.head_mean = torch.nn.Linear(hidden, 1)
        self.head_logvar = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=0.1, training=self.training)
        h = F.relu(self.conv3(h, edge_index))
        mu = self.head_mean(h).squeeze(-1)
        log_var = self.head_logvar(h).squeeze(-1)
        # Clamp log-var for stability (sigma^2 in [1e-4, 1e2])
        log_var = torch.clamp(log_var, min=-9.0, max=4.6)
        return mu, log_var


def gaussian_nll_per_node(mu, log_var, y):
    return 0.5 * log_var + 0.5 * (y - mu) ** 2 / torch.exp(log_var)


def build_edge_index(A):
    """Convert dense symmetric adjacency to 2 x num_edges LongTensor."""
    rows, cols = np.where(A > 0)
    ei = np.stack([rows, cols], axis=0)
    return torch.tensor(ei, dtype=torch.long)


def train_one(seed_offset, x, edge_index, y_train, train_idx, n, device='cpu'):
    torch.manual_seed(seed_offset)
    np.random.seed(seed_offset)
    model = GCNMeanVar(in_dim=x.size(1)).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY,
    )
    x = x.to(device); edge_index = edge_index.to(device); y_train = y_train.to(device)
    train_idx_t = torch.tensor(train_idx, dtype=torch.long, device=device)
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        mu, log_var = model(x, edge_index)
        loss = gaussian_nll_per_node(
            mu[train_idx_t], log_var[train_idx_t], y_train
        ).mean()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        mu, log_var = model(x, edge_index)
    return mu.cpu().numpy(), np.exp(log_var.cpu().numpy())


def recal_scalar(mu, var, y, idx):
    from scipy.optimize import minimize_scalar
    r = (y[idx] - mu[idx]) ** 2
    v = var[idx]
    def loss(log_c):
        c = np.exp(log_c)
        return 0.5 * np.sum(np.log(2 * np.pi * c * v) + r / (c * v))
    res = minimize_scalar(loss, bounds=(-6, 6), method='bounded')
    return float(res.fun), float(np.exp(res.x))


def load_dataset(name, day=None):
    if name == 'metrla':
        return load_metrla()
    if name == 'pemsbay':
        return load_pemsbay()
    if name == 'laqn':
        return load_laqn()
    if name == 'pm25':
        return load_pm25()
    if name == 'caweather':
        return load_caweather(day=day)
    raise ValueError(name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        choices=['metrla', 'pemsbay', 'laqn', 'pm25', 'caweather'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--day', type=int, default=257)
    parser.add_argument('--outdir', default='graph_gp/results')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    day_arg = args.day if args.dataset == 'caweather' else None
    A, L, n, y, d_raw, coords_km, feat = load_dataset(args.dataset, day=day_arg)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]
    xi = perm[int(0.7 * n):]

    # Features: d_i and log(deg+1), normalised
    deg = A.sum(axis=1).flatten()
    x = np.stack([
        (d_raw - d_raw.mean()) / (d_raw.std() + 1e-8),
        (np.log(deg + 1) - np.log(deg + 1).mean()) / (np.log(deg + 1).std() + 1e-8),
    ], axis=1).astype(np.float32)
    x_t = torch.tensor(x, dtype=torch.float32)
    edge_index = build_edge_index(A)
    y32 = y.astype(np.float32)
    yt = torch.tensor(y32[ti], dtype=torch.float32)

    print(f'{args.dataset} seed={args.seed}'
          f'{(" day="+str(args.day)) if day_arg is not None else ""} '
          f'n={n}, training M={M_MODELS} GNNs...', flush=True)

    mus, vars_ = [], []
    for m in range(M_MODELS):
        offs = args.seed * 1000 + m
        mu, var = train_one(offs, x_t, edge_index, yt, ti, n)
        mus.append(mu); vars_.append(var)
        print(f'  model {m+1}/{M_MODELS} done', flush=True)

    mus = np.stack(mus, axis=0)
    vars_ = np.stack(vars_, axis=0)

    # Ensemble predictive: mean of means, mean var + spread of means.
    mu_e = mus.mean(axis=0)
    var_e = vars_.mean(axis=0) + mus.var(axis=0)

    # Test metrics on xi
    nll = float(np.sum(
        0.5 * np.log(2 * np.pi * var_e[xi])
        + 0.5 * (y[xi] - mu_e[xi]) ** 2 / var_e[xi]
    ))
    rmse = float(np.sqrt(np.mean((y[xi] - mu_e[xi]) ** 2)))
    rec, c_star = recal_scalar(mu_e, var_e, y, xi)

    result = {
        'seed': args.seed,
        'dataset': args.dataset,
        'n': int(n),
        'M': M_MODELS,
        'hidden': HIDDEN,
        'epochs': EPOCHS,
        'nll_degnn': nll,
        'rmse_degnn': rmse,
        'recal_degnn': rec,
        'c_degnn': c_star,
    }
    print(f'  ensemble: NLL={nll:.2f}, RMSE={rmse:.3f}, Recal={rec:.2f}, '
          f'c*={c_star:.3f}', flush=True)

    if args.dataset == 'caweather':
        out = f'{args.outdir}/results_degnn_caweather_day{args.day}_seed{args.seed}.json'
    else:
        out = f'{args.outdir}/results_degnn_{args.dataset}_seed{args.seed}.json'
    with open(out, 'w') as f:
        json.dump(result, f)
    print(f'Saved {out}', flush=True)

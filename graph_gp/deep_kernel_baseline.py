"""Deep kernel learning baseline: MLP feature map + stationary GP.

A learned-representation baseline. Uses a small MLP to learn
a non-linear feature map, then applies a stationary RBF kernel in
the learned feature space. This is a standard non-stationary GP
approach, included as a learned-representation comparison.

Simpler variant: input warping via sigmoid transform of node features.
"""

import sys, os, pickle, gzip, numpy as np, pandas as pd, h5py, torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
sys.path.insert(0, os.path.dirname(__file__))

from baselines import gp_posterior, evaluate_gp

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SEEDS = list(range(42, 62))

def nll_test(mu, var, y, idx):
    yt = y[idx]; mt = mu[idx]; vt = var[idx]
    return np.sum(0.5 * np.log(2 * np.pi * vt) + 0.5 * (yt - mt)**2 / vt)


class DeepKernelGP(nn.Module):
    """MLP feature map + RBF kernel."""

    def __init__(self, input_dim, hidden=16, latent=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, latent),
        ).double()
        self.log_ls = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        self.log_noise = nn.Parameter(torch.tensor(-4.0, dtype=torch.float64))

    def kernel(self, X):
        Z = self.mlp(X)  # (n, latent)
        ls = torch.exp(self.log_ls)
        scale = torch.exp(self.log_scale)
        # RBF in latent space
        dists = torch.cdist(Z, Z)
        K = scale * torch.exp(-0.5 * dists**2 / ls**2)
        return K

    def forward(self, X, y, obs_idx):
        n = X.shape[0]; m = len(obs_idx)
        K = self.kernel(X)
        noise = torch.exp(self.log_noise)
        K_nn = K[obs_idx][:, obs_idx] + noise * torch.eye(m, dtype=torch.float64)
        L = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(m, dtype=torch.float64))
        alpha = torch.linalg.solve_triangular(
            L.T, torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False),
            upper=True).squeeze(-1)
        nll = 0.5 * y @ alpha + torch.sum(torch.log(torch.diag(L)))
        return nll, K, noise


def train_deep_kernel(features, y_train, train_idx, n, n_iter=300, lr=0.01):
    """Train deep kernel GP and return kernel matrix + noise."""
    X = torch.tensor(features, dtype=torch.float64)
    y = torch.tensor(y_train, dtype=torch.float64)
    obs = torch.tensor(train_idx, dtype=torch.long)

    model = DeepKernelGP(features.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_nll = float('inf')
    best_state = None

    for it in range(n_iter):
        optimizer.zero_grad()
        nll, K, noise = model(X, y, obs)
        nll.backward()
        optimizer.step()
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    with torch.no_grad():
        _, K, noise = model(X, y, obs)
        K_np = K.numpy()
        K_np = (K_np + K_np.T) / 2
        noise_np = noise.item()

    return K_np, noise_np


def run_dataset(name, features, y_norm, n):
    """Run deep kernel baseline on one dataset."""
    log(f'\n{name}: Deep Kernel Baseline')
    nlls = []; rmses = []

    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.RandomState(seed); perm = rng.permutation(n)
        ti = perm[:int(0.7 * n)]; xi = perm[int(0.7 * n):]; yt = y_norm[ti]

        K, noise = train_deep_kernel(features, yt, ti, n)
        mu, var, _ = gp_posterior(K, yt, ti, noise)
        nll = nll_test(mu, var, y_norm, xi)
        rmse, _, _ = evaluate_gp(y_norm, mu, var, xi)
        nlls.append(nll); rmses.append(rmse)
        log(f'  seed {seed}: NLL={nll:.1f}, RMSE={rmse:.3f}')

    nlls = np.array(nlls); rmses = np.array(rmses)
    log(f'  Mean: NLL={nlls.mean():.2f}+/-{nlls.std():.2f}, RMSE={rmses.mean():.3f}+/-{rmses.std():.3f}')
    return nlls, rmses


if __name__ == '__main__':
    OUTFILE = 'graph_gp/deep_kernel_results.txt'
    f = open(OUTFILE, 'w')
    def log(msg): f.write(msg + '\n'); f.flush()

    # ==================== METR-LA ====================
    log('Deep Kernel GP Baseline')
    log('=' * 60)

    DATA_DIR = os.path.join(os.path.dirname(__file__), 'data_cache', 'metr_la')
    with open(os.path.join(DATA_DIR, 'adj_mx.pkl'), 'rb') as fp:
        _, _, adj = pickle.load(fp, encoding='latin1')
    with h5py.File(os.path.join(DATA_DIR, 'metr-la.h5'), 'r') as fp:
        speeds = fp['df']['block0_values'][:]
    n_m = speeds.shape[1]
    y_m = speeds[:288, :].mean(axis=0); y_m = (y_m - y_m.mean()) / y_m.std()
    # Features: mean speed, temporal std, temporal CV, degree
    A_m = (adj > 0).astype(float); A_m = np.maximum(A_m, A_m.T); np.fill_diagonal(A_m, 0)
    deg_m = A_m.sum(1)
    feat_m = np.stack([
        (speeds.mean(0) - speeds.mean(0).mean()) / speeds.mean(0).std(),
        (speeds.std(0) - speeds.std(0).mean()) / speeds.std(0).std(),
        np.log(deg_m.clip(1)),
    ], axis=1)
    feat_m[:, 2] = (feat_m[:, 2] - feat_m[:, 2].mean()) / (feat_m[:, 2].std() + 1e-8)

    nlls_m, rmses_m = run_dataset('METR-LA', feat_m, y_m, n_m)

    # ==================== LAQN ====================
    from laqn_regression import fetch_laqn_sites, fetch_laqn_data, build_graph, build_laplacian as build_lap2
    sites = fetch_laqn_sites(); values = fetch_laqn_data(sites, 'NO2', '2024-06-15')
    sites = [s for s in sites if s['code'] in values]
    ys_l = np.array([values[s['code']] for s in sites])
    type_nums = np.array([{'Rural':0,'Suburban':1,'Urban Background':2,'Industrial':3,'Roadside':4,'Kerbside':5}.get(s['type'],2) for s in sites], dtype=float)
    coords = np.array([[s['lat'], s['lon']] for s in sites])
    ys_ln = (ys_l - ys_l.mean()) / ys_l.std()
    A_l, _ = build_graph(coords, 5.0); L_l = build_lap2(A_l)
    deg_l = A_l.sum(1); conn = deg_l > 0
    ys_ln = ys_ln[conn]; type_nums = type_nums[conn]; coords = coords[conn]
    A_l = A_l[np.ix_(conn, conn)]; n_l = int(conn.sum()); deg_l = A_l.sum(1)
    feat_l = np.stack([
        (type_nums - type_nums.mean()) / (type_nums.std() + 1e-8),
        np.log(deg_l.clip(1)),
        (coords[:, 0] - coords[:, 0].mean()) / (coords[:, 0].std() + 1e-8),
        (coords[:, 1] - coords[:, 1].mean()) / (coords[:, 1].std() + 1e-8),
    ], axis=1)
    feat_l[:, 1] = (feat_l[:, 1] - feat_l[:, 1].mean()) / (feat_l[:, 1].std() + 1e-8)

    nlls_l, rmses_l = run_dataset('LAQN', feat_l, ys_ln, n_l)

    # ==================== CA Weather ====================
    stations = []
    with open(os.path.join(os.path.dirname(__file__), 'data_cache', 'weather', 'ghcnd-stations.txt')) as fh:
        for line in fh:
            sid = line[:11].strip(); lat = float(line[12:20]); lon = float(line[21:30])
            if sid.startswith('US') and 32 < lat < 42 and -125 < lon < -114:
                stations.append({'id': sid, 'lat': lat, 'lon': lon})
    ca_ids = set(s['id'] for s in stations); id_to_info = {s['id']: s for s in stations}
    rows = []
    with gzip.open(os.path.join(os.path.dirname(__file__), 'data_cache', 'weather', '2023.csv.gz'), 'rt') as fh:
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) >= 4 and parts[0] in ca_ids and parts[2] == 'TMAX':
                rows.append({'id': parts[0], 'date': parts[1], 'tmax': float(parts[3]) / 10})
    df = pd.DataFrame(rows); pivot = df.pivot_table(index='date', columns='id', values='tmax')
    good = pivot.columns[pivot.notna().sum() > 300]; pivot = pivot[good].fillna(pivot[good].mean())
    coords_w = np.array([[id_to_info[sid]['lat'], id_to_info[sid]['lon']] for sid in good])
    coords_km = coords_w.copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    Dd = cdist(coords_km, coords_km); A_w = ((Dd < 50) & (Dd > 0)).astype(float)
    deg_w = A_w.sum(1); conn_w = deg_w > 0
    A_w = A_w[np.ix_(conn_w, conn_w)]; vals_w = pivot.values[:, np.where(conn_w)[0]]
    coords_w = coords_w[np.where(conn_w)[0]]
    n_w = int(conn_w.sum()); deg_w = A_w.sum(1)
    if n_w > 250:
        rng_sub = np.random.RandomState(0); sub = rng_sub.choice(n_w, 200, replace=False); sub.sort()
        A_w = A_w[np.ix_(sub, sub)]; vals_w = vals_w[:, sub]; coords_w = coords_w[sub]
        deg_w = A_w.sum(1); c2 = deg_w > 0
        A_w = A_w[np.ix_(c2, c2)]; vals_w = vals_w[:, np.where(c2)[0]]
        coords_w = coords_w[np.where(c2)[0]]
        n_w = int(c2.sum()); deg_w = A_w.sum(1)
    y_w = vals_w[180, :]; y_w = (y_w - y_w.mean()) / (y_w.std() + 1e-8)
    feat_w = np.stack([
        (vals_w.var(0) - vals_w.var(0).mean()) / (vals_w.var(0).std() + 1e-8),
        np.log(deg_w.clip(1)),
        (coords_w[:, 0] - coords_w[:, 0].mean()) / (coords_w[:, 0].std() + 1e-8),
        (coords_w[:, 1] - coords_w[:, 1].mean()) / (coords_w[:, 1].std() + 1e-8),
    ], axis=1)
    feat_w[:, 1] = (feat_w[:, 1] - feat_w[:, 1].mean()) / (feat_w[:, 1].std() + 1e-8)

    nlls_w, rmses_w = run_dataset('CA Weather', feat_w, y_w, n_w)

    log('\nDone.')
    f.close()

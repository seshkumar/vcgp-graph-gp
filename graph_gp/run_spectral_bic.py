"""Unnormalized-L spectral clustering + BIC K-selection, single
(dataset, seed [, day]) job.

For a given dataset/seed:
  1. Load data, build unnormalized L = D - A, compute smallest-eigvec
     embedding.
  2. Sweep K over a dataset-adaptive grid. For each K:
       a. Spectral labels = KMeans(K) on the first K eigenvectors of L.
       b. Block k-NN Laplacian on geographic coords intra-cluster.
       c. Het warm-start, then learn_gamma_per_cluster (vanilla nu=0.5).
       d. Compute training log-marginal-likelihood logMLL_train(K).
          Record BIC(K) = -2 logMLL_train + K * log(n_train).
       e. Compute held-out NLL/RMSE/recalNLL on 30% test nodes.
  3. K* = argmin BIC(K). Record (K*, NLL*, RMSE*).

Output JSON: graph_gp/results/spectral_bic_{dataset}{_day}_seed{seed}.json
  { seed, dataset, day?, n, n_train, K_grid, by_K: {K: {...}},
    K_star, K_star_metrics: {...} }

Usage:
  python graph_gp/run_spectral_bic.py --dataset metrla --seed 1729
  python graph_gp/run_spectral_bic.py --dataset caweather --seed 1729 --day 195
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import argparse, json, sys, time
import numpy as np
import torch
torch.set_default_dtype(torch.float64)
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import minimize_scalar

sys.path.insert(0, 'graph_gp')
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from run_all_methods import (
    load_metrla, load_pemsbay, load_laqn, load_pm25, load_caweather,
    fit_het_joint, learn_gamma_per_cluster,
)
from baselines import gp_posterior


LOADERS = {
    'metrla':  load_metrla,
    'pemsbay': load_pemsbay,
    'laqn':    load_laqn,
    'pm25':    load_pm25,
}


def _log(m): print(m, flush=True)


def load_dataset(name, day=None):
    if name == 'caweather':
        return load_caweather(day=day)
    return LOADERS[name]()


def unnorm_laplacian(A):
    d = A.sum(1)
    return np.diag(d) - A


def smallest_eigvecs(L_unnorm, top_m):
    L_unnorm = 0.5 * (L_unnorm + L_unnorm.T)
    vals, vecs = np.linalg.eigh(L_unnorm)
    order = np.argsort(vals)
    return vals[order][:top_m], vecs[:, order][:, :top_m]


def spectral_labels(vecs, K, seed=0):
    emb = vecs[:, :K]
    km = KMeans(n_clusters=K, n_init=10, random_state=seed)
    return km.fit_predict(emb)


def build_block_laplacian(coords_km, labels, k_nn=8):
    n = len(labels)
    A = np.zeros((n, n))
    K = int(labels.max()) + 1
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2: continue
        nn = NearestNeighbors(n_neighbors=min(k_nn + 1, len(idx)))
        nn.fit(coords_km[idx])
        _, nb = nn.kneighbors(coords_km[idx])
        for li, neigh in enumerate(nb):
            for lj in neigh[1:]:
                A[idx[li], idx[lj]] = 1.0
    A = np.maximum(A, A.T)
    return A, np.diag(A.sum(1)) - A


def lgamma_block_kernel(L, n, d_raw, tau, scale, gammas, labels):
    d = np.maximum(d_raw, 1e-8)
    d_pow = np.clip(np.power(d, gammas[labels] / 2.0), 1e-4, 1e4)
    D_inv = np.diag(1.0 / d); D_pow = np.diag(d_pow)
    Q = D_inv + 2.0 * tau * D_pow @ L @ D_pow
    K = scale * np.linalg.inv(Q + 1e-8 * np.eye(n))
    return 0.5 * (K + K.T)


def train_nll_from_params(L_bl, n, d_raw, yt, ti, labels,
                          tau, scale, noise, gammas):
    Lt = torch.tensor(L_bl)
    d_t = torch.tensor(np.maximum(d_raw, 1e-8))
    D_inv = torch.diag(1.0 / d_t)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    gamma_t = torch.tensor(gammas, dtype=torch.float64)
    c_t = torch.tensor(labels, dtype=torch.long)
    gamma_per_node = gamma_t[c_t]
    d_pow = torch.pow(d_t, gamma_per_node / 2.0).clamp(1e-4, 1e4)
    Q = D_inv + 2 * tau * (d_pow[:, None] * Lt * d_pow[None, :])
    K_ker = scale * torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
    K_ker = (K_ker + K_ker.T) / 2
    Kn = K_ker[ot][:, ot] + noise * torch.eye(m, dtype=torch.float64)
    Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
    a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
        Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
    quad = 0.5 * (yy @ a).item()
    logdet = torch.sum(torch.log(torch.diag(Lc))).item()
    return quad + logdet + 0.5 * m * np.log(2 * np.pi)


def nll_test(mu, var, y, idx):
    return float(np.sum(0.5 * np.log(2 * np.pi * var[idx])
                        + 0.5 * (y[idx] - mu[idx]) ** 2 / var[idx]))


def recal_nll(mu, var, y, idx):
    r = (y[idx] - mu[idx]) ** 2
    v = var[idx]
    def loss(log_c):
        c = np.exp(log_c)
        return 0.5 * np.sum(np.log(2 * np.pi * c * v) + r / (c * v))
    res = minimize_scalar(loss, bounds=(-6, 6), method='bounded')
    return float(res.fun), float(np.exp(res.x))


def K_grid_for(name, n):
    """Dataset-adaptive K grid. Keep K <= n//6 and at least 2 clusters."""
    if name == 'laqn':
        return [2, 3, 4, 5, 7]
    if name == 'metrla':
        return [2, 3, 5, 7, 9, 11, 13, 17, 20]
    if name == 'pemsbay':
        return [2, 3, 5, 7, 9, 11, 13, 17, 20]
    if name == 'caweather':
        return [2, 3, 5, 7, 10, 13, 17, 20]
    if name == 'pm25':
        return [2, 5, 9, 11, 13, 17, 20]
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=['metrla', 'pemsbay', 'laqn', 'pm25', 'caweather'])
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--day', type=int, default=None,
                    help='CA Weather day-of-year (only for caweather)')
    ap.add_argument('--outdir', default='graph_gp/results')
    ap.add_argument('--k_nn', type=int, default=8,
                    help='intra-cluster k-NN for block Laplacian')
    args = ap.parse_args()

    if args.dataset == 'caweather' and args.day is None:
        ap.error('--day required for caweather')
    if args.dataset != 'caweather' and args.day is not None:
        ap.error('--day only used for caweather')

    t0 = time.time()
    tag = f'{args.dataset}' + (f'_day{args.day}' if args.day is not None else '')
    _log(f'=== spectral_bic {tag} seed={args.seed} ===')

    # 1. Data
    if args.day is not None:
        data = load_dataset(args.dataset, day=args.day)
    else:
        data = load_dataset(args.dataset)
    A, _L, n, y, d_raw, coords_km, _feat = data
    _log(f'loaded n={n}  ({time.time()-t0:.1f}s)')

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]; xi = perm[int(0.7 * n):]
    yt = y[ti]
    m = len(ti)
    _log(f'split  n_train={m}  n_test={len(xi)}')

    # 2. Unnormalized Laplacian + eigendecomp
    L_unnorm = unnorm_laplacian(A)
    K_grid = K_grid_for(args.dataset, n)
    top_m = max(K_grid) + 3
    t1 = time.time()
    _vals, vecs = smallest_eigvecs(L_unnorm, top_m)
    _log(f'eigendecomp ({top_m} smallest) done ({time.time()-t1:.1f}s)')
    _log(f'K grid: {K_grid}')

    by_K = {}
    for K_try in K_grid:
        tK = time.time()
        labels = spectral_labels(vecs, K_try, seed=0)
        sizes = sorted(np.bincount(labels, minlength=K_try).tolist(), reverse=True)
        _, L_bl = build_block_laplacian(coords_km, labels, k_nn=args.k_nn)

        # het warm-start
        het = fit_het_joint(L_bl, n, d_raw, yt, ti)
        if het is None:
            torch.manual_seed(args.seed + 1)
            het = fit_het_joint(L_bl, n, d_raw, yt, ti)
        if het is None:
            het = (0.1, 1.0, 0.1)
        th, sh, noh = het

        params = learn_gamma_per_cluster(L_bl, n, d_raw, yt, ti, labels,
                                         init=(th, sh, noh))
        if params is None:
            _log(f'  K={K_try:2d}  FAIL')
            by_K[int(K_try)] = dict(status='fail')
            continue
        tau, scale, noise, gammas = params

        # training marginal likelihood + BIC
        nll_train = train_nll_from_params(L_bl, n, d_raw, yt, ti, labels,
                                          tau, scale, noise, gammas)
        bic = 2 * nll_train + K_try * np.log(m)

        # held-out metrics
        Kmat = lgamma_block_kernel(L_bl, n, d_raw, tau, scale, gammas, labels)
        mu, var, _ = gp_posterior(Kmat, yt, ti, noise)
        nll = nll_test(mu, var, y, xi)
        rmse = float(np.sqrt(np.mean((y[xi] - mu[xi]) ** 2)))
        rec, c_rec = recal_nll(mu, var, y, xi)
        elapsed = time.time() - tK

        _log(f'  K={K_try:2d}  nll_train={nll_train:7.2f}  BIC={bic:8.2f}  '
             f'NLL={nll:7.2f}  RMSE={rmse:.3f}  recal={rec:7.2f}  '
             f'gamma=[{gammas.min():+.2f},{gammas.max():+.2f}]  ({elapsed:.1f}s)')

        by_K[int(K_try)] = dict(
            status='ok',
            nll_train=float(nll_train), bic=float(bic),
            nll=float(nll), rmse=float(rmse), recal=float(rec),
            c_recal=float(c_rec),
            sizes=sizes,
            tau=float(tau), scale=float(scale), noise=float(noise),
            gammas=[float(g) for g in gammas],
            elapsed_s=float(elapsed),
        )

    # 3. BIC pick
    ok = {k: v for k, v in by_K.items() if v.get('status') == 'ok'}
    if ok:
        k_star = min(ok, key=lambda k: ok[k]['bic'])
        star_row = ok[k_star]
        _log(f'BIC pick: K*={k_star}  BIC={star_row["bic"]:.2f}  '
             f'NLL={star_row["nll"]:.2f}  RMSE={star_row["rmse"]:.3f}')
    else:
        k_star = None; star_row = None

    os.makedirs(args.outdir, exist_ok=True)
    fname = f'spectral_bic_{tag}_seed{args.seed}.json'
    with open(f'{args.outdir}/{fname}', 'w') as f:
        json.dump(dict(
            seed=args.seed, dataset=args.dataset, day=args.day,
            n=int(n), n_train=int(m), K_grid=list(K_grid),
            by_K=by_K, K_star=k_star, K_star_row=star_row,
            elapsed_total_s=float(time.time() - t0),
        ), f, indent=2)
    _log(f'wrote {args.outdir}/{fname}  (total {time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()

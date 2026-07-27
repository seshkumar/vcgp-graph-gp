"""CA Weather lgamma_block with augmented K-means.

Features = [lat_std, lon_std, w * d_i_std]. K-means on the augmented
feature vector; per-cluster k-NN graph uses geographic (lat, lon) only;
lgamma_block fits K gamma_k via joint MLL.

One (day, K, w, seed) per job.

Usage:
    python graph_gp/run_caweather_augmented.py --day 257 --K 5 --w 0.5 --seed 1729
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')
torch.set_default_dtype(torch.float64)

from baselines import gp_posterior
from run_all_methods import (
    fit_het_joint, learn_gamma_per_cluster, load_caweather,
)


def build_block_laplacian(coords_km, labels, k_nn=8):
    n = len(labels)
    A = np.zeros((n, n))
    K = int(labels.max()) + 1
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2:
            continue
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
    d_pow = np.power(d, gammas[labels] / 2.0)
    d_pow = np.clip(d_pow, 1e-4, 1e4)
    D_inv = np.diag(1.0 / d)
    D_pow = np.diag(d_pow)
    Q = D_inv + 2.0 * tau * D_pow @ L @ D_pow
    K = scale * np.linalg.inv(Q + 1e-8 * np.eye(n))
    return 0.5 * (K + K.T)


def nll_test(mu, var, y, idx):
    return float(np.sum(
        0.5 * np.log(2 * np.pi * var[idx])
        + 0.5 * (y[idx] - mu[idx]) ** 2 / var[idx]
    ))


def recal_nll(mu, var, y, idx):
    from scipy.optimize import minimize_scalar
    r = (y[idx] - mu[idx]) ** 2
    v = var[idx]
    def loss(log_c):
        c = np.exp(log_c)
        return 0.5 * np.sum(np.log(2 * np.pi * c * v) + r / (c * v))
    res = minimize_scalar(loss, bounds=(-6, 6), method='bounded')
    return float(res.fun), float(np.exp(res.x))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--day', type=int, required=True)
    parser.add_argument('--K', type=int, required=True,
                        choices=[2, 5, 10, 20])
    parser.add_argument('--w', type=float, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--outdir', default='graph_gp/results')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    A, L, n, y, d_raw, coords_km, _ = load_caweather(day=args.day)

    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]
    xi = perm[int(0.7 * n):]
    yt = y[ti]

    coords_std = (coords_km - coords_km.mean(0)) / coords_km.std(0)
    d_std = ((d_raw - d_raw.mean()) / d_raw.std()).reshape(-1, 1)
    features = np.column_stack([coords_std, args.w * d_std])
    labels = KMeans(n_clusters=args.K, random_state=0, n_init=10).fit(features).labels_
    cluster_sizes = np.bincount(labels, minlength=args.K).tolist()

    print(f'caweather aug day={args.day} K={args.K} w={args.w} seed={args.seed} n={n}',
          flush=True)
    print(f'  cluster sizes: {sorted(cluster_sizes, reverse=True)}', flush=True)

    A_bl, L_bl = build_block_laplacian(coords_km, labels)
    tau_h, scale_h, noise_h = fit_het_joint(L_bl, n, d_raw, yt, ti)
    params = learn_gamma_per_cluster(
        L_bl, n, d_raw, yt, ti, labels,
        init=(tau_h, scale_h, noise_h),
    )
    if params is None:
        result = {
            'seed': args.seed, 'dataset': 'caweather_aug', 'day': args.day,
            'K': args.K, 'w': args.w, 'n': int(n),
            'nll_aug': float('nan'), 'rmse_aug': float('nan'),
            'recal_aug': float('nan'),
        }
    else:
        tau, scale, noise, gammas = params
        Kmat = lgamma_block_kernel(L_bl, n, d_raw, tau, scale, gammas, labels)
        mu, var, _ = gp_posterior(Kmat, yt, ti, noise)
        nll = nll_test(mu, var, y, xi)
        rmse = float(np.sqrt(np.mean((y[xi] - mu[xi]) ** 2)))
        rec, c_star = recal_nll(mu, var, y, xi)
        result = {
            'seed': args.seed, 'dataset': 'caweather_aug',
            'day': args.day, 'K': args.K, 'w': args.w, 'n': int(n),
            'cluster_sizes': cluster_sizes,
            'tau_aug': tau, 'scale_aug': scale, 'noise_aug': noise,
            'gamma_per_cluster': gammas.tolist(),
            'nll_aug': nll, 'rmse_aug': rmse,
            'recal_aug': rec, 'c_aug': c_star,
        }
        print(f'  NLL={nll:.2f}, RMSE={rmse:.3f}, '
              f'gamma range [{gammas.min():+.2f}, {gammas.max():+.2f}]',
              flush=True)

    out = (f'{args.outdir}/results_caweather_aug_day{args.day}_K{args.K}'
           f'_w{args.w:.1f}_seed{args.seed}.json')
    with open(out, 'w') as f:
        json.dump(result, f)
    print(f'Saved {out}', flush=True)

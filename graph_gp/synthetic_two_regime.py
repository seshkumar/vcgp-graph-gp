"""Two-regime synthetic: motivates the variance-coupled family.

Analogous to the homophilic/heterophilic split in GNN literature. A single
stochastic-block-model graph with two communities is built so that:

  * community A is SPARSE (low intra-block density -> low-degree nodes);
  * community B is DENSE  (high intra-block density -> high-degree nodes).

Data is sampled from a GP with precision

    Q_gamma = D^{-1} + 2*tau * diag(d^{gamma(x)/2}) L diag(d^{gamma(x)/2}),

where gamma(x) is piecewise constant: gamma_A = +1 on community A, and
gamma_B = -2 on community B. This is the cleanest separation: on the
sparse "homophilic" side the prior couples neighbours (gamma>0), on the
dense "heterophilic" side Proposition 1 says high-degree nodes decouple
(gamma<0 towards/past -1).

We then fit six models and compare NLL/RMSE across 10 seeds:

  OUTSIDE THE FAMILY (stationary, no gamma):
    * Matern-1/2 spectral (via GeometricKernels)
    * Matern-3/2 spectral (via GeometricKernels)

  INSIDE THE FAMILY, fixed gamma:
    * stat    (gamma=0, uniform d_i = 1)
    * het     (gamma=-1)

  INSIDE THE FAMILY, learned gamma:
    * lgamma        (one scalar gamma)
    * lgamma_block  (two gamma_k on oracle block labels; K=2)

Expectation: stationary baselines have an irreducible NLL gap; scalar
lgamma takes a weighted-average gamma and closes part of the gap;
lgamma_block recovers (gamma_A, gamma_B) near the true (+1, -2) and
takes NLL/RMSE top-1.

Outputs: a JSON per seed to graph_gp/results/synth_two_regime/,
plus an aggregated table to stdout.

Usage:
    conda activate graphgp
    python -u graph_gp/synthetic_two_regime.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
torch.set_default_dtype(torch.float64)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')

from baselines import gp_posterior, learn_gk_matern
from run_all_methods import (
    fit_het_joint, fit_stationary, learn_gamma, learn_gamma_per_cluster,
    nll_test, compute_metrics,
)


SEEDS = list(range(1729, 1749))   # 20 seeds (override via SEEDS env)
N_PER_BLOCK = 100                  # n = 200
P_IN_SPARSE = 0.04                 # block A: expected d ~ 4
P_IN_DENSE  = 0.20                 # block B: expected d ~ 20
P_BETWEEN   = 0.005
GAMMA_SPARSE = +1.0                # "homophilic" side
GAMMA_DENSE  = -2.0                # "heterophilic" side
TAU_TRUE = 1.0
NOISE_VAR_TRUE = 0.05
OUTDIR = 'graph_gp/results/synth_two_regime'


def log(msg):
    print(msg, flush=True)


# ------------------------------------------------------------------
# Graph + signal generation
# ------------------------------------------------------------------

def build_two_block_sbm(n_per_block, p_in_sparse, p_in_dense, p_between, seed):
    """Two-block SBM. Block A (sparse) indexed 0..n_per_block; block B (dense)
    indexed n_per_block..2*n_per_block. Guarantees no isolated nodes by
    connecting any isolated node to its lowest-index peer.
    """
    rng = np.random.RandomState(seed)
    n = 2 * n_per_block
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if i < n_per_block and j < n_per_block:
                p = p_in_sparse
            elif i >= n_per_block and j >= n_per_block:
                p = p_in_dense
            else:
                p = p_between
            if rng.rand() < p:
                A[i, j] = A[j, i] = 1.0
    # repair any isolated node
    d = A.sum(1)
    for i in np.where(d == 0)[0]:
        block_start = 0 if i < n_per_block else n_per_block
        for j in range(block_start, block_start + n_per_block):
            if j != i and A[i, j] == 0:
                A[i, j] = A[j, i] = 1.0; break
    L = np.diag(A.sum(1)) - A
    return A, L


def build_kernel_piecewise_gamma(L, d_raw, tau, gamma_per_node):
    """K = (D^{-1} + 2*tau * D^{gamma/2} L D^{gamma/2})^{-1}, per-node gamma."""
    n = L.shape[0]
    d = np.maximum(d_raw, 1e-8)
    d_pow = np.clip(np.power(d, gamma_per_node / 2.0), 1e-4, 1e4)
    D_inv = np.diag(1.0 / d)
    D_pow = np.diag(d_pow)
    Q = D_inv + 2.0 * tau * D_pow @ L @ D_pow
    K = np.linalg.inv(Q + 1e-8 * np.eye(n))
    return 0.5 * (K + K.T)


def sample_y(K, noise_var, rng):
    n = K.shape[0]
    L_chol = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    z = rng.randn(n)
    f = L_chol @ z
    return f + np.sqrt(noise_var) * rng.randn(n)


# ------------------------------------------------------------------
# Per-seed fit pipeline
# ------------------------------------------------------------------

def run_one_seed(seed):
    t0 = time.time()
    log(f'\n========= seed={seed} =========')

    # Graph is shared across seeds via a dedicated RNG stream so that all
    # seeds see the same graph (noise varies, topology does not). This
    # matches the paper's protocol (fixed dataset, varying split).
    A_adj, L = build_two_block_sbm(N_PER_BLOCK, P_IN_SPARSE, P_IN_DENSE,
                                    P_BETWEEN, seed=0)
    n = L.shape[0]
    d_raw = A_adj.sum(1)
    labels = np.concatenate([np.zeros(N_PER_BLOCK, dtype=int),
                              np.ones(N_PER_BLOCK, dtype=int)])
    d_A = d_raw[:N_PER_BLOCK].mean(); d_B = d_raw[N_PER_BLOCK:].mean()
    log(f'  graph: n={n}  d(A)={d_A:.1f}  d(B)={d_B:.1f}')

    # Piecewise true gamma + data
    gamma_true = np.where(labels == 0, GAMMA_SPARSE, GAMMA_DENSE)
    K_true = build_kernel_piecewise_gamma(L, d_raw, TAU_TRUE, gamma_true)
    rng = np.random.RandomState(seed)
    y = sample_y(K_true, NOISE_VAR_TRUE, rng)

    # 70/30 split
    perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]; xi = perm[int(0.7 * n):]
    yt = y[ti]

    results = {'seed': seed, 'd_A': float(d_A), 'd_B': float(d_B)}

    # --- Matern-1/2 (outside family) ---
    t = time.time()
    try:
        K_m05, noise_m05, info_m05 = learn_gk_matern(A_adj, n, yt, ti, nu=0.5)
        mu, var, _ = gp_posterior(K_m05, yt, ti, noise_m05)
        results.update(compute_metrics('m05', mu, var, y, xi))
        log(f'  m05   NLL={results["nll_m05"]:7.2f}  RMSE={results["rmse_m05"]:.3f}  ({time.time()-t:.1f}s)')
    except Exception as e:
        log(f'  m05 FAILED: {e}')

    # --- Matern-3/2 (outside family) ---
    t = time.time()
    try:
        K_m15, noise_m15, info_m15 = learn_gk_matern(A_adj, n, yt, ti, nu=1.5)
        mu, var, _ = gp_posterior(K_m15, yt, ti, noise_m15)
        results.update(compute_metrics('m15', mu, var, y, xi))
        log(f'  m15   NLL={results["nll_m15"]:7.2f}  RMSE={results["rmse_m15"]:.3f}  ({time.time()-t:.1f}s)')
    except Exception as e:
        log(f'  m15 FAILED: {e}')

    # --- stat: gamma=0, uniform d_i (the "no-learn" null) ---
    t = time.time()
    tau_s, sc_s, no_s = fit_stationary(L, n, yt, ti)
    K_stat = sc_s * np.linalg.inv(np.eye(n) + 2 * tau_s * L)
    K_stat = 0.5 * (K_stat + K_stat.T)
    mu, var, _ = gp_posterior(K_stat, yt, ti, no_s)
    results.update(compute_metrics('stat', mu, var, y, xi))
    log(f'  stat  NLL={results["nll_stat"]:7.2f}  RMSE={results["rmse_stat"]:.3f}  ({time.time()-t:.1f}s)')

    # --- het: gamma = -1, real d_i ---
    t = time.time()
    het = fit_het_joint(L, n, d_raw, yt, ti)
    if het is None:
        torch.manual_seed(seed + 1); het = fit_het_joint(L, n, d_raw, yt, ti)
    tau_h, sc_h, no_h = het
    D_sqrt = np.sqrt(np.maximum(d_raw, 1e-8))
    K_het = sc_h * (D_sqrt[:, None] * np.linalg.inv(
        np.eye(n) + 2 * tau_h * L) * D_sqrt[None, :])
    K_het = 0.5 * (K_het + K_het.T)
    mu, var, _ = gp_posterior(K_het, yt, ti, no_h)
    results.update(compute_metrics('het', mu, var, y, xi))
    log(f'  het   NLL={results["nll_het"]:7.2f}  RMSE={results["rmse_het"]:.3f}  ({time.time()-t:.1f}s)')

    # --- lgamma: scalar learned gamma ---
    t = time.time()
    pg = learn_gamma(L, n, d_raw, yt, ti, init=(tau_h, sc_h, no_h), gamma_init=-1.0)
    if pg is None:
        log(f'  lgamma returned None; skipping'); return results
    tau_g, sc_g, no_g, gamma_scalar = pg
    d_pow_g = np.clip(np.power(np.maximum(d_raw, 1e-8), gamma_scalar / 2.0), 1e-4, 1e4)
    Q_g = np.diag(1.0 / np.maximum(d_raw, 1e-8)) + 2 * tau_g * (
        d_pow_g[:, None] * L * d_pow_g[None, :])
    K_g = sc_g * np.linalg.inv(Q_g + 1e-8 * np.eye(n))
    K_g = 0.5 * (K_g + K_g.T)
    mu, var, _ = gp_posterior(K_g, yt, ti, no_g)
    results.update(compute_metrics('lgamma', mu, var, y, xi))
    results['gamma_scalar'] = float(gamma_scalar)
    log(f'  lgamma NLL={results["nll_lgamma"]:7.2f}  RMSE={results["rmse_lgamma"]:.3f}  '
        f'gamma*={gamma_scalar:+.2f}  ({time.time()-t:.1f}s)')

    # --- lgamma_block: oracle (K=2) block labels ---
    t = time.time()
    pb = learn_gamma_per_cluster(L, n, d_raw, yt, ti, labels,
                                  init=(tau_h, sc_h, no_h))
    if pb is None:
        log(f'  lgamma_block returned None; skipping'); return results
    tau_b, sc_b, no_b, gk = pb
    gamma_per_node = gk[labels]
    d_pow_b = np.clip(np.power(np.maximum(d_raw, 1e-8), gamma_per_node / 2.0), 1e-4, 1e4)
    Q_b = np.diag(1.0 / np.maximum(d_raw, 1e-8)) + 2 * tau_b * (
        d_pow_b[:, None] * L * d_pow_b[None, :])
    K_b = sc_b * np.linalg.inv(Q_b + 1e-8 * np.eye(n))
    K_b = 0.5 * (K_b + K_b.T)
    mu, var, _ = gp_posterior(K_b, yt, ti, no_b)
    results.update(compute_metrics('lgamma_block', mu, var, y, xi))
    results['gamma_k'] = [float(x) for x in gk]
    log(f'  block  NLL={results["nll_lgamma_block"]:7.2f}  '
        f'RMSE={results["rmse_lgamma_block"]:.3f}  '
        f'(gamma_A={gk[0]:+.2f}, gamma_B={gk[1]:+.2f})  ({time.time()-t:.1f}s)')

    log(f'  seed {seed} total {time.time()-t0:.1f}s')
    return results


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------

def summarise(rows):
    methods = ['m05', 'm15', 'stat', 'het', 'lgamma', 'lgamma_block']
    log('\n' + '=' * 86)
    log(f'SYNTHETIC TWO-REGIME  (n=200; gamma_true_A=+1, gamma_true_B=-2)')
    log(f'{len(rows)} seeds')
    log('=' * 86)
    hdr = f'  {"method":<16s}  {"NLL":>18s}  {"RMSE":>18s}'
    log(hdr); log('  ' + '-' * 72)
    for m in methods:
        nlls = [r[f'nll_{m}'] for r in rows if f'nll_{m}' in r]
        rmses = [r[f'rmse_{m}'] for r in rows if f'rmse_{m}' in r]
        if not nlls: continue
        nll_mean, nll_se = float(np.mean(nlls)), float(np.std(nlls, ddof=1) / np.sqrt(len(nlls)))
        rmse_mean, rmse_se = float(np.mean(rmses)), float(np.std(rmses, ddof=1) / np.sqrt(len(rmses)))
        log(f'  {m:<16s}  {nll_mean:>8.2f} +/- {nll_se:<5.2f}   {rmse_mean:>.4f} +/- {rmse_se:.4f}')
    # Recovered gammas
    if any('gamma_scalar' in r for r in rows):
        gs = [r['gamma_scalar'] for r in rows if 'gamma_scalar' in r]
        log(f'\n  scalar gamma*:    mean {np.mean(gs):+.2f}  std {np.std(gs):.2f}  '
            f'(compromise between true +1 and -2)')
    if any('gamma_k' in r for r in rows):
        gA = [r['gamma_k'][0] for r in rows if 'gamma_k' in r]
        gB = [r['gamma_k'][1] for r in rows if 'gamma_k' in r]
        log(f'  block gamma_A:    mean {np.mean(gA):+.2f}  std {np.std(gA):.2f}  (true +1.00)')
        log(f'  block gamma_B:    mean {np.mean(gB):+.2f}  std {np.std(gB):.2f}  (true -2.00)')


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    log(f'Synthetic two-regime experiment; outdir={OUTDIR}')
    seeds = SEEDS
    env_seeds = os.environ.get('SEEDS')
    if env_seeds:
        seeds = [int(s) for s in env_seeds.split(',')]
    rows = []
    for seed in seeds:
        r = run_one_seed(seed)
        rows.append(r)
        with open(f'{OUTDIR}/seed{seed}.json', 'w') as f:
            json.dump(r, f, indent=2)
    summarise(rows)
    with open(f'{OUTDIR}/all_seeds.json', 'w') as f:
        json.dump(rows, f, indent=2)
    log(f'\nWrote per-seed JSONs and all_seeds.json to {OUTDIR}')


if __name__ == '__main__':
    main()

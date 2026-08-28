"""Identifiability experiment: can MLL recover known gamma from one snapshot?

Uses the locked paper-form precision
    Q_gamma = D^{-1} + 2*tau * D^{gamma/2} L D^{gamma/2}
with unnormalised Laplacian L = D_deg - A on a 20x20 grid. For each true
gamma and training fraction, we draw 10 synthetic samples y ~ GP(0, K_gamma),
refit gamma by joint MLL (warm-started at gamma = -1 as in run_all_methods
learn_gamma), and record recovered gamma_hat.

Saves docs/nonstationary_graph_gp/figs/identifiability.pdf and
graph_gp/results/identifiability_results.json.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')
torch.set_default_dtype(torch.float64)

from synthetic_nonstationary import build_laplacian, make_grid_graph


def paper_kernel(L, d_raw, tau, scale, gamma, n):
    d = np.maximum(d_raw, 1e-8)
    D_inv = np.diag(1.0 / d)
    d_pow = np.clip(np.power(d, -gamma / 2.0), 1e-4, 1e4)
    D_pow = np.diag(d_pow)
    Q = D_inv + 2.0 * tau * D_pow @ L @ D_pow
    K = scale * np.linalg.inv(Q + 1e-8 * np.eye(n))
    return 0.5 * (K + K.T)


def fit_gamma_mll(L, n, d_raw, yt, ti, n_iter=300, lr=0.03):
    """Joint MLL for (tau, scale, noise, gamma). Warm-start at gamma = -1."""
    Lt = torch.tensor(L)
    d_t = torch.tensor(np.maximum(d_raw, 1e-8))
    D_inv = torch.diag(1.0 / d_t)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long)
    m = len(ti)

    lt = torch.tensor(0.0, requires_grad=True)
    ls = torch.tensor(0.0, requires_grad=True)
    ln = torch.tensor(-4.0, requires_grad=True)
    gamma = torch.tensor(-1.0, requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln, gamma], lr=lr)

    best_nll = 1e20
    best_g = float('nan')
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        d_pow = torch.clamp(torch.pow(d_t, -gamma / 2.0), min=1e-4, max=1e4)
        D_pow = torch.diag(d_pow)
        Q = D_inv + 2.0 * t * D_pow @ Lt @ D_pow
        K = s * torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
        K = 0.5 * (K + K.T)
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(
                Kn + 1e-6 * torch.eye(m, dtype=torch.float64)
            )
        except Exception:
            continue
        a = torch.linalg.solve_triangular(
            Lc.T,
            torch.linalg.solve_triangular(Lc, yy.unsqueeze(-1), upper=False),
            upper=True,
        ).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        if not torch.isfinite(nll):
            continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_g = gamma.item()
        nll.backward()
        opt.step()
    return best_g


def generate_and_recover(gamma_true, train_frac, seed, side=20):
    n, edge_index, pos = make_grid_graph(side)
    L, _ = build_laplacian(n, edge_index, normalize=False)

    x_coord = (pos[:, 0] - pos[:, 0].min()) / (
        pos[:, 0].max() - pos[:, 0].min()
    )
    d_true = 0.2 + 1.8 * x_coord
    d_true = d_true / d_true.mean()
    d_true = np.clip(d_true, 0.1, 10)

    tau_true, scale_true, noise_true = 2.0, 1.0, 0.01
    K_true = paper_kernel(L, d_true, tau_true, scale_true, gamma_true, n)

    rng = np.random.RandomState(seed)
    L_chol = np.linalg.cholesky(K_true + 1e-6 * np.eye(n))
    f_true = L_chol @ rng.randn(n)
    y = f_true + np.sqrt(noise_true) * rng.randn(n)

    perm = rng.permutation(n)
    m = int(train_frac * n)
    ti = perm[:m]
    yt = y[ti]

    return fit_gamma_mll(L, n, d_true, yt, ti)


if __name__ == '__main__':
    SIGN = 1.0
    gammas_true_paper = [-1.0, 0.0, 1.0, 3.0]
    train_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    n_seeds = 10

    results = {}
    for gt_p in gammas_true_paper:
        gt_c = SIGN * gt_p  # convert to code convention for the fitter
        results[gt_p] = {}
        for tf in train_fracs:
            recovered = []
            for seed in range(n_seeds):
                g_hat_c = generate_and_recover(gt_c, tf, seed)
                recovered.append(SIGN * g_hat_c)  # flip back to paper conv
            results[gt_p][tf] = recovered
            print(
                f'gamma_true(paper)={gt_p:+.1f}, frac={tf:.1f}: '
                f'mean={np.mean(recovered):+.3f} +/- {np.std(recovered):.3f}',
                flush=True,
            )

    fig, axes = plt.subplots(
        1, len(gammas_true_paper), figsize=(4 * len(gammas_true_paper), 4),
        sharey=False,
    )
    for ax, gt in zip(axes, gammas_true_paper):
        means = [np.mean(results[gt][tf]) for tf in train_fracs]
        stds = [np.std(results[gt][tf]) for tf in train_fracs]
        ax.errorbar(train_fracs, means, yerr=stds, fmt='ko-',
                    capsize=3, markersize=5)
        ax.axhline(gt, color='red', linestyle='--', linewidth=1.5,
                   label=rf'True $\gamma={gt:+.1f}$')
        ax.set_xlabel(r'Training fraction $|S|/n$')
        ax.set_ylabel(r'Recovered $\hat{\gamma}$')
        ax.set_title(rf'$\gamma_{{\mathrm{{true}}}} = {gt:+.1f}$')
        ax.legend(fontsize=8)

    plt.suptitle(
        r'Identifiability: MLL recovers $\gamma$ from one spatial snapshot',
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = 'docs/nonstationary_graph_gp/neurips2026/figs/identifiability.pdf'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, bbox_inches='tight')
    print(f'Saved {out}')

    os.makedirs('graph_gp/results', exist_ok=True)
    with open('graph_gp/results/identifiability_results.json', 'w') as f:
        json.dump(
            {str(k): {str(tf): v for tf, v in d.items()}
             for k, d in results.items()},
            f,
        )

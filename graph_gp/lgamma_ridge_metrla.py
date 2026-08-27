"""Ridge-regularised γ on METR-LA: does pulling γ toward γ_ref=-1 close
the RMSE gap vs m15 while preserving the NLL win vs het?

Objective:
    -MLL(tau, scale, sigma2, gamma) + lam * (gamma - gamma_ref)**2

For lam=0 this recovers vanilla lgamma; for lam -> inf it pins gamma=gamma_ref,
i.e. lgamma collapses to het. The question is whether an intermediate lam
lands gamma in a regime where high-d nodes stay coupled to the graph (so
extreme-|y| outliers get shrunk) without giving up the NLL win.

Usage:
    python graph_gp/lgamma_ridge_metrla.py \
        --seeds 1729 1730 1731 1732 1733 1734 1735 1736 1737 1738 \
        --lambdas 0 0.5 1 2 5 10 30 \
        --gamma-ref -1.0

For each seed and each lam we refit (warm-started from het) and evaluate
NLL, RMSE, recal-NLL on the held-out 30%. Outputs a table per seed and a
summary table (mean/std across seeds) per lam.
"""
import argparse
import sys
import time
sys.path.insert(0, 'graph_gp')
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from run_all_methods import (
    load_metrla, fit_het_joint, compute_metrics,
)
from baselines import gp_posterior, learn_gk_matern


def _log(msg):
    print(msg, flush=True)


def learn_gamma_ridge(L, n, d_raw, yt, ti, lam, gamma_ref=1.0,
                      n_iter=300, lr=0.03, init=None, gamma_init=1.0,
                      nu=0.5):
    """learn_gamma variant with a ridge penalty lam*(gamma - gamma_ref)^2
    added to -MLL. lam=0 recovers the paper's vanilla lgamma.
    nu=0.5: K = s*Q^{-1}. nu=1.5: K = s*Q^{-2} (Matern-3/2 smoothness).
    """
    power = int(round(nu + 0.5))  # 0.5 -> 1, 1.5 -> 2
    Lt = torch.tensor(L)
    d_t = torch.tensor(np.maximum(d_raw, 1e-8))
    D_inv = torch.diag(1.0 / d_t)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    if init is None:
        lt0, ls0, ln0 = 0., 0., -4.
    else:
        t0, s0, no0 = init
        lt0 = float(np.log(max(t0, 1e-12)))
        ls0 = float(np.log(max(s0, 1e-12)))
        ln0 = float(np.log(max(no0, 1e-12)))
    lt = torch.tensor(lt0, requires_grad=True)
    ls = torch.tensor(ls0, requires_grad=True)
    ln = torch.tensor(ln0, requires_grad=True)
    gamma = torch.tensor(float(gamma_init), requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln, gamma], lr=lr)
    best_obj = 1e20; best_params = None
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        d_pow = torch.pow(d_t, -gamma / 2.0)
        d_pow = torch.clamp(d_pow, min=1e-4, max=1e4)
        D_pow = torch.diag(d_pow)
        Q = D_inv + 2 * t * D_pow @ Lt @ D_pow
        Qinv = torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
        K = Qinv
        for _ in range(power - 1):
            K = K @ Qinv
        K = s * K
        K = (K + K.T) / 2
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        penalty = lam * (gamma - gamma_ref) ** 2
        obj = nll + penalty
        if not torch.isfinite(obj):
            continue
        if obj.item() < best_obj:
            best_obj = obj.item()
            best_params = (t.item(), s.item(), no.item(), gamma.item(),
                           nll.item(), penalty.item())
        obj.backward(); opt.step()
    return best_params


def build_Q_gamma(L, d_raw, tau, gamma):
    d = np.maximum(d_raw, 1e-8)
    D_inv = np.diag(1.0 / d)
    dpow = np.clip(np.power(d, -gamma / 2.0), 1e-4, 1e4)
    D_pow = np.diag(dpow)
    return D_inv + 2.0 * tau * D_pow @ L @ D_pow


def kernel_from_Q(Q, scale, n, nu=0.5):
    """K = scale * Q^{-(nu+1/2)}. nu=0.5 -> inverse; nu=1.5 -> squared inverse."""
    power = int(round(nu + 0.5))
    Qinv = np.linalg.inv(Q + 1e-8 * np.eye(n))
    K = Qinv
    for _ in range(power - 1):
        K = K @ Qinv
    K = scale * K
    return 0.5 * (K + K.T)


def eval_one(A, L, n, y, d_raw, seed, lambdas, gamma_ref):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]
    xi = perm[int(0.7 * n):]
    yt = y[ti]

    t0 = time.time()
    th, sh, noh = fit_het_joint(L, n, d_raw, yt, ti, n_iter=200)
    _log(f'[seed {seed}] het: tau={th:.3f} scale={sh:.3f} noise={noh:.4f} '
         f'({time.time()-t0:.1f}s)')

    # m15 reference for RMSE comparison
    t1 = time.time()
    K_m15, noise_m15, info_m15 = learn_gk_matern(
        A, n, yt, ti, nu=1.5, n_iter=200)
    mu_m15, var_m15, _ = gp_posterior(K_m15, yt, ti, noise_m15)
    m_m15 = compute_metrics('m15', mu_m15, var_m15, y, xi)
    _log(f'[seed {seed}] m15: ls={info_m15["lengthscale"]:.3f} '
         f'nll={m_m15["nll_m15"]:+.2f} rmse={m_m15["rmse_m15"]:.3f} '
         f'({time.time()-t1:.1f}s)')

    rows = []
    for lam in lambdas:
        t2 = time.time()
        params = learn_gamma_ridge(L, n, d_raw, yt, ti, lam=lam,
                                   gamma_ref=gamma_ref, n_iter=300,
                                   init=(th, sh, noh), gamma_init=1.0)
        if params is None:
            _log(f'[seed {seed}] lam={lam}: FAIL')
            continue
        tau, scale, noise, gamma, mll_part, pen_part = params
        Q = build_Q_gamma(L, d_raw, tau, gamma)
        K_lg = kernel_from_Q(Q, scale, n)
        mu_lg, var_lg, _ = gp_posterior(K_lg, yt, ti, noise)
        m_lg = compute_metrics('lg', mu_lg, var_lg, y, xi)
        _log(f'[seed {seed}] lam={lam:>5.1f}: gamma={gamma:+.3f} '
             f'nll={m_lg["nll_lg"]:+.2f} rmse={m_lg["rmse_lg"]:.3f} '
             f'recal={m_lg["recal_lg"]:+.2f} '
             f'(MLL part {mll_part:+.2f}, pen {pen_part:.3f}, '
             f'{time.time()-t2:.1f}s)')
        rows.append({
            'seed': seed, 'lam': lam, 'gamma': gamma, 'tau': tau,
            'scale': scale, 'noise': noise,
            'nll': m_lg['nll_lg'], 'rmse': m_lg['rmse_lg'],
            'recal': m_lg['recal_lg'],
        })
    return rows, m_m15


def summarise(all_rows, m15_stats, lambdas):
    _log('\n=== summary across seeds ===')
    seeds = sorted({r['seed'] for r in all_rows})
    _log(f'N seeds = {len(seeds)}')
    _log(f'm15 baseline (held-out): '
         f'NLL mean={np.mean([m["nll_m15"] for m in m15_stats]):+.2f} '
         f'RMSE mean={np.mean([m["rmse_m15"] for m in m15_stats]):.3f} '
         f'recal mean={np.mean([m["recal_m15"] for m in m15_stats]):+.2f}')
    _log('')
    _log(f'{"lam":>6s} {"gamma_mean":>11s} {"gamma_std":>10s} '
         f'{"NLL mean":>9s} {"NLL std":>8s} '
         f'{"RMSE mean":>10s} {"RMSE std":>9s} '
         f'{"recal mean":>11s}')
    for lam in lambdas:
        rs = [r for r in all_rows if r['lam'] == lam]
        if not rs: continue
        g = np.array([r['gamma'] for r in rs])
        nll = np.array([r['nll'] for r in rs])
        rmse = np.array([r['rmse'] for r in rs])
        rec = np.array([r['recal'] for r in rs])
        _log(f'{lam:>6.2f} {g.mean():>+11.3f} {g.std(ddof=1):>10.3f} '
             f'{nll.mean():>+9.2f} {nll.std(ddof=1):>8.2f} '
             f'{rmse.mean():>10.3f} {rmse.std(ddof=1):>9.3f} '
             f'{rec.mean():>+11.2f}')

    # Paired diffs vs lam=0 (vanilla lgamma) and vs m15
    _log('\n=== paired differences ===')
    lam0_rows = {r['seed']: r for r in all_rows if r['lam'] == 0.0}
    if lam0_rows:
        for lam in lambdas:
            if lam == 0.0: continue
            rs = [r for r in all_rows if r['lam'] == lam]
            if not rs: continue
            d_nll = np.array([r['nll'] - lam0_rows[r['seed']]['nll'] for r in rs])
            d_rmse = np.array([r['rmse'] - lam0_rows[r['seed']]['rmse'] for r in rs])
            _log(f'  lam={lam:>5.2f} vs lam=0: '
                 f'dNLL mean={d_nll.mean():+.3f} (std {d_nll.std(ddof=1):.3f}), '
                 f'dRMSE mean={d_rmse.mean():+.4f} (std {d_rmse.std(ddof=1):.4f})')
    # vs m15
    seed_to_m15 = {s: m15_stats[i] for i, s in enumerate(seeds)}
    _log('')
    for lam in lambdas:
        rs = [r for r in all_rows if r['lam'] == lam]
        if not rs: continue
        d_nll = np.array([r['nll'] - seed_to_m15[r['seed']]['nll_m15'] for r in rs])
        d_rmse = np.array([r['rmse'] - seed_to_m15[r['seed']]['rmse_m15'] for r in rs])
        _log(f'  lam={lam:>5.2f} vs m15 : '
             f'dNLL mean={d_nll.mean():+.3f}, dRMSE mean={d_rmse.mean():+.4f}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds', type=int, nargs='+',
                   default=[1729, 1730, 1731, 1732, 1733, 1734, 1735, 1736, 1737, 1738])
    p.add_argument('--lambdas', type=float, nargs='+',
                   default=[0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0])
    p.add_argument('--gamma-ref', type=float, default=-1.0)
    args = p.parse_args()

    A, L, n, y, d_raw, coords_km, feat = load_metrla()
    _log(f'METR-LA: n={n}  '
         f'd_raw in [{d_raw.min():.2f}, {d_raw.max():.2f}]  '
         f'y std={y.std():.3f}')
    _log(f'seeds = {args.seeds}')
    _log(f'lambdas = {args.lambdas}  gamma_ref = {args.gamma_ref}')

    all_rows = []
    m15_stats = []
    for seed in args.seeds:
        rows, m15 = eval_one(A, L, n, y, d_raw, seed,
                             args.lambdas, args.gamma_ref)
        all_rows.extend(rows)
        m15_stats.append(m15)

    summarise(all_rows, m15_stats, args.lambdas)


if __name__ == '__main__':
    main()

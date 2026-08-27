"""Fisher-info-matched lgamma-ridge, single seed × dataset.

Protocol:
  1. 70/30 split via np.random.RandomState(seed).permutation.
  2. Fit vanilla lgamma (lam=0) via MLL on the 70% train.
  3. At the vanilla MLE, compute the 4x4 Hessian of -MLL and invert;
     Var(γ̂) = Hinv[3,3].  lam_fisher = 0.5 / Var(γ̂).
  4. Refit lgamma-ridge with penalty lam_fisher*(γ - γ_ref)² on the
     same train indices.
  5. Evaluate held-out NLL/RMSE/recal on the 30% test nodes.

Outputs a single JSON per (dataset, seed [,day]) with:
  gamma_vanilla, gamma_fisher, lam_fisher, var_gamma, se_gamma,
  nll_vanilla, rmse_vanilla, recal_vanilla,
  nll_fisher,  rmse_fisher,  recal_fisher.

Usage:
  python graph_gp/run_fisher_ridge.py --dataset metrla --seed 1729
  python graph_gp/run_fisher_ridge.py --dataset caweather --seed 1729 --day 195
"""
import argparse
import json
import os
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
    load_metrla, load_pemsbay, load_laqn, load_pm25, load_caweather,
    fit_het_joint, compute_metrics,
)
from baselines import gp_posterior
from lgamma_ridge_metrla import (
    learn_gamma_ridge, build_Q_gamma, kernel_from_Q,
)


def _log(msg):
    print(msg, flush=True)


LOADERS = {
    'metrla': load_metrla,
    'pemsbay': load_pemsbay,
    'laqn': load_laqn,
    'pm25': load_pm25,
}


def load_dataset(name, day=None):
    if name == 'caweather':
        return load_caweather(day=day)
    return LOADERS[name]()


def neg_mll(lt, ls, ln, gamma, L, d_raw, n, yt, ti, nu=0.5):
    power = int(round(nu + 0.5))
    Lt = torch.tensor(L)
    d_t = torch.tensor(np.maximum(d_raw, 1e-8))
    D_inv = torch.diag(1.0 / d_t)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
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
    # Escalating jitter: nu=3/2 (Q^{-2}) can be ill-conditioned at some
    # pathological parameter points; don't let a single Cholesky failure
    # abort the whole seed.
    Lc = None
    for jitter in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        try:
            Lc = torch.linalg.cholesky(Kn + jitter * torch.eye(m, dtype=torch.float64))
            break
        except Exception:
            continue
    if Lc is None:
        raise RuntimeError('neg_mll: Cholesky failed with jitter up to 1e-2')
    a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
        Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
    return 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))


def fisher_var_gamma(tau, scale, noise, gamma, L, d_raw, n, yt, ti, nu=0.5):
    lt = torch.tensor(float(np.log(max(tau, 1e-12))), requires_grad=True)
    ls = torch.tensor(float(np.log(max(scale, 1e-12))), requires_grad=True)
    ln = torch.tensor(float(np.log(max(noise, 1e-12))), requires_grad=True)
    g  = torch.tensor(float(gamma), requires_grad=True)
    params = [lt, ls, ln, g]
    nll = neg_mll(lt, ls, ln, g, L, d_raw, n, yt, ti, nu=nu)
    grad1 = torch.autograd.grad(nll, params, create_graph=True)
    H = torch.zeros(4, 4, dtype=torch.float64)
    for i in range(4):
        grad2 = torch.autograd.grad(grad1[i], params, retain_graph=True)
        for j in range(4):
            H[i, j] = grad2[j].detach()
    H = 0.5 * (H + H.T)
    H_reg = H + 1e-8 * torch.eye(4, dtype=torch.float64)
    Hinv = torch.linalg.inv(H_reg)
    return float(Hinv[3, 3]), float(H[3, 3])


def eval_on_holdout(L, n, d_raw, y, ti, xi, tau, scale, noise, gamma, tag, nu=0.5):
    Q = build_Q_gamma(L, d_raw, tau, gamma)
    K = kernel_from_Q(Q, scale, n, nu=nu)
    mu, var, _ = gp_posterior(K, y[ti], ti, noise)
    return compute_metrics(tag, mu, var, y, xi)


def run(dataset, seed, day, outdir, gamma_ref, n_iter, nu=0.5):
    t0 = time.time()
    A, L, n, y, d_raw, coords_km, feat = load_dataset(dataset, day=day)
    _log(f'[{dataset} seed={seed}{" day="+str(day) if day is not None else ""}] '
         f'loaded n={n} in {time.time()-t0:.1f}s')

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]
    xi = perm[int(0.7 * n):]
    yt = y[ti]

    # Warm-start from het; retry with perturbed seed, fall back to a neutral init
    t1 = time.time()
    het = fit_het_joint(L, n, d_raw, yt, ti, n_iter=200)
    if het is None:
        _log(f'  het warm-start returned None; retrying with alt torch seed')
        torch.manual_seed(seed + 1)
        het = fit_het_joint(L, n, d_raw, yt, ti, n_iter=200)
    if het is None:
        _log(f'  het warm-start still None; falling back to neutral init '
             f'(tau=0.1, scale=1.0, noise=0.1)')
        het = (0.1, 1.0, 0.1)
    th, sh, noh = het
    _log(f'  het warm-start in {time.time()-t1:.1f}s: '
         f'tau={th:.3f} scale={sh:.3f} noise={noh:.4f}')

    # Vanilla lgamma (lam=0)
    t2 = time.time()
    params0 = learn_gamma_ridge(L, n, d_raw, yt, ti, lam=0.0,
                                gamma_ref=gamma_ref, n_iter=n_iter,
                                init=(th, sh, noh), gamma_init=1.0, nu=nu)
    if params0 is None:
        _log(f'  vanilla lgamma FAILED')
        return
    tau0, scale0, noise0, gamma0, _, _ = params0
    m_v = eval_on_holdout(L, n, d_raw, y, ti, xi, tau0, scale0, noise0, gamma0, 'v', nu=nu)
    _log(f'  vanilla: gamma={gamma0:+.3f}  nll={m_v["nll_v"]:+.2f}  '
         f'rmse={m_v["rmse_v"]:.3f}  ({time.time()-t2:.1f}s)')

    # Fisher variance at vanilla MLE
    t3 = time.time()
    var_g, H_gg = fisher_var_gamma(
        tau0, scale0, noise0, gamma0, L, d_raw, n, yt, ti, nu=nu)
    se_g = float(np.sqrt(max(var_g, 1e-12)))
    lam_fisher = 0.5 / max(var_g, 1e-12)
    _log(f'  Fisher: Var(gamma)={var_g:.4f}  SE={se_g:.3f}  H_diag={H_gg:.3f}  '
         f'==> lam_fisher={lam_fisher:.3f}  ({time.time()-t3:.1f}s)')

    # Refit with Fisher-matched ridge
    t4 = time.time()
    params_f = learn_gamma_ridge(L, n, d_raw, yt, ti, lam=lam_fisher,
                                 gamma_ref=gamma_ref, n_iter=n_iter,
                                 init=(th, sh, noh), gamma_init=1.0, nu=nu)
    if params_f is None:
        _log(f'  ridge-fisher FAILED')
        return
    tauF, scaleF, noiseF, gammaF, _, _ = params_f
    m_f = eval_on_holdout(L, n, d_raw, y, ti, xi, tauF, scaleF, noiseF, gammaF, 'f', nu=nu)
    _log(f'  fisher : gamma={gammaF:+.3f}  nll={m_f["nll_f"]:+.2f}  '
         f'rmse={m_f["rmse_f"]:.3f}  recal={m_f["recal_f"]:+.2f}  '
         f'({time.time()-t4:.1f}s)')

    # Write JSON
    out = {
        'dataset': dataset, 'seed': int(seed), 'day': day, 'n': int(n),
        'nu': float(nu),
        'gamma_ref': float(gamma_ref),
        'tau_vanilla': float(tau0), 'scale_vanilla': float(scale0),
        'noise_vanilla': float(noise0), 'gamma_vanilla': float(gamma0),
        'nll_vanilla': float(m_v['nll_v']),
        'rmse_vanilla': float(m_v['rmse_v']),
        'recal_vanilla': float(m_v['recal_v']),
        'c_vanilla': float(m_v['c_v']),
        'var_gamma_fisher': float(var_g),
        'se_gamma_fisher': float(se_g),
        'H_gamma_gamma_diag': float(H_gg),
        'lam_fisher': float(lam_fisher),
        'tau_fisher': float(tauF), 'scale_fisher': float(scaleF),
        'noise_fisher': float(noiseF), 'gamma_fisher': float(gammaF),
        'nll_fisher': float(m_f['nll_f']),
        'rmse_fisher': float(m_f['rmse_f']),
        'recal_fisher': float(m_f['recal_f']),
        'c_fisher': float(m_f['c_f']),
        'elapsed_s': float(time.time() - t0),
    }
    os.makedirs(outdir, exist_ok=True)
    suffix = f'_day{day}' if day is not None else ''
    nu_tag = '' if abs(nu - 0.5) < 1e-8 else '_nu' + f'{nu:g}'.replace('.', '_')
    fname = f'fisher_ridge{nu_tag}_{dataset}{suffix}_seed{seed}.json'
    fpath = os.path.join(outdir, fname)
    with open(fpath, 'w') as fh:
        json.dump(out, fh, indent=2)
    _log(f'  => wrote {fpath}  (total {time.time()-t0:.1f}s)')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['metrla', 'pemsbay', 'laqn', 'pm25', 'caweather'])
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--day', type=int, default=None,
                   help='Day-of-year (caweather only).')
    p.add_argument('--outdir', default='graph_gp/results')
    p.add_argument('--gamma-ref', type=float, default=-1.0)
    p.add_argument('--n-iter', type=int, default=300)
    p.add_argument('--nu', type=float, default=0.5,
                   choices=[0.5, 1.5],
                   help='Matern smoothness. 0.5 -> K=s*Q^{-1}; 1.5 -> K=s*Q^{-2}.')
    args = p.parse_args()
    if args.dataset == 'caweather' and args.day is None:
        p.error('--day is required when --dataset=caweather')
    run(args.dataset, args.seed, args.day, args.outdir,
        args.gamma_ref, args.n_iter, nu=args.nu)


if __name__ == '__main__':
    main()

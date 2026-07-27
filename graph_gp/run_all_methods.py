"""Unified experiment runner: all methods, all metrics, all datasets.

For each (dataset, seed), computes NLL, RMSE, Recal for every method:
  stat, het, semisup, m05, m15, lgamma, fuglstad_unreg, fuglstad_reg (lambda sweep),
  DK, PS, BKTR, SpecPoly

Datasets: metrla, pemsbay, laqn (single-snapshot 70/30 split)
          caweather (per-day evaluation on day_of_year, d_i from 2022)

All methods use the unnormalized (combinatorial) Laplacian L = D_deg - A.
het and semisup are fit by joint MLL (their own tau, scale, noise).
learn_gamma uses the paper's Q_gamma = D^{-1} + 2*tau * D^{gamma/2} L D^{gamma/2},
so gamma=0 recovers semisup and gamma=-1 recovers het exactly (Proposition 1).

Output: graph_gp/results/results_{dataset}_seed{seed}.json
        graph_gp/results/results_caweather_day{day}_seed{seed}.json

Usage:
  python graph_gp/run_all_methods.py --dataset metrla --seed 1729
  python graph_gp/run_all_methods.py --dataset caweather --seed 1729 --day 195
"""
import argparse, os, sys, json, gzip, pickle, numpy as np, torch, h5py
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')

from baselines import gp_posterior, learn_gk_matern, learn_gk_matern_nu
from deep_kernel_baseline import train_deep_kernel
from paciorek_schervish_baseline import learn_ps_kernel

torch.set_default_dtype(torch.float64)


# =====================================================================
# Shared utilities
# =====================================================================

def compute_laplacian(A):
    """Combinatorial (unnormalized) graph Laplacian L = D_deg - A."""
    deg = A.sum(1)
    return np.diag(deg) - A


def apply_blocking(coords_km, n_clusters, k_nn=8, cluster_seed=0):
    """Partition nodes by K-means on coordinates, build k-NN within each cluster.

    Returns (A_block, L_block, cluster_labels). A_block is a block-diagonal
    adjacency: edges exist only between nodes in the same cluster. This is a
    dataset-agnostic helper; callers can pass any 2D (or 3D) coordinate array.
    """
    n = coords_km.shape[0]
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=n_clusters, random_state=cluster_seed, n_init=10)
    cluster_labels = km.fit_predict(coords_km).astype(np.int64)
    A = np.zeros((n, n))
    for c in range(n_clusters):
        idx_c = np.where(cluster_labels == c)[0]
        n_c = len(idx_c)
        if n_c < 2:
            continue
        k_c = min(k_nn, n_c - 1)
        sub = cdist(coords_km[idx_c], coords_km[idx_c])
        np.fill_diagonal(sub, np.inf)
        nn = np.argpartition(sub, k_c, axis=1)[:, :k_c]
        for li, i in enumerate(idx_c):
            for lj in nn[li]:
                j = idx_c[lj]
                A[i, j] = 1.0; A[j, i] = 1.0
    L = compute_laplacian(A)
    return A, L, cluster_labels


# Module-level handle for cluster labels (set by the CLI when --n_clusters>1;
# consumed by the pm25 per-block lgamma code path).
_PM25_CLUSTER_LABELS = None


def nll_test(mu, var, y, idx):
    yt, mt, vt = y[idx], mu[idx], var[idx]
    return float(np.sum(0.5 * np.log(2 * np.pi * vt) + 0.5 * (yt - mt)**2 / vt))


def recal_nll(c, mu, var, y, idx):
    v = c * var[idx]
    return float(np.sum(0.5 * np.log(2 * np.pi * v) + 0.5 * (y[idx] - mu[idx])**2 / v))


def compute_metrics(name, mu, var, y, xi):
    """Compute NLL, RMSE, Recal for one method. Returns dict of results."""
    nll = nll_test(mu, var, y, xi)
    rmse = float(np.sqrt(np.mean((y[xi] - mu[xi])**2)))
    res = minimize_scalar(lambda c: recal_nll(c, mu, var, y, xi),
                          bounds=(0.01, 100), method='bounded')
    recal = recal_nll(res.x, mu, var, y, xi)
    return {f'nll_{name}': nll, f'rmse_{name}': rmse,
            f'recal_{name}': recal, f'c_{name}': res.x}


# =====================================================================
# Model fitting functions
# =====================================================================

def fit_stationary(L, n, yt, ti, n_iter=200, lr=0.05):
    """Learn tau, scale, noise for stationary resolvent (I + 2*tau*L)^{-1}."""
    Lt = torch.tensor(L)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    lt = torch.tensor(0., requires_grad=True)
    ls = torch.tensor(0., requires_grad=True)
    ln = torch.tensor(-4., requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln], lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        Q = torch.eye(n, dtype=torch.float64) + 2 * t * Lt
        K = s * torch.linalg.inv(Q); K = (K + K.T) / 2
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        nll.backward(); opt.step()
    return torch.exp(lt).item(), torch.exp(ls).item(), torch.exp(ln).item()


def build_kernel(L, n, tau, scale, kappa=None):
    """Build kernel from precision Q = D_kappa^{-1} + 2*tau * D_kappa^{1/2} L D_kappa^{1/2}.
    If kappa is None, uses stationary (kappa=1 for all nodes).
    """
    if kappa is None:
        Q = np.eye(n) + 2 * tau * L
    else:
        kappa_safe = np.maximum(kappa, 1e-8)
        Q = np.diag(1.0 / kappa_safe) + \
            2 * tau * np.diag(np.sqrt(kappa_safe)) @ L @ np.diag(np.sqrt(kappa_safe))
    K = scale * np.linalg.inv(Q + 1e-8 * np.eye(n))
    return (K + K.T) / 2


def fit_het_joint(L, n, d_raw, yt, ti, n_iter=200, lr=0.05):
    """Joint MLL: learn (tau, scale, noise) for K_h = s * D^{1/2} (I + 2*tau*L)^{-1} D^{1/2}."""
    Lt = torch.tensor(L)
    D_sqrt = torch.tensor(np.sqrt(np.maximum(d_raw, 1e-8)))
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    lt = torch.tensor(0., requires_grad=True)
    ls = torch.tensor(0., requires_grad=True)
    ln = torch.tensor(-4., requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln], lr=lr)
    best_nll = 1e20; best = None
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        Q = torch.eye(n, dtype=torch.float64) + 2 * t * Lt
        Kinv = torch.linalg.inv(Q)
        K = s * (D_sqrt[:, None] * Kinv * D_sqrt[None, :])
        K = (K + K.T) / 2
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        if not torch.isfinite(nll): continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best = (t.item(), s.item(), no.item())
        nll.backward(); opt.step()
    return best


def fit_semisup_joint(L, n, d_raw, yt, ti, n_iter=200, lr=0.05):
    """Joint MLL: learn (tau, scale, noise) for K_ss = s * (D^{-1} + 2*tau*L)^{-1}."""
    Lt = torch.tensor(L)
    d_inv = torch.tensor(1.0 / np.maximum(d_raw, 1e-8))
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    lt = torch.tensor(0., requires_grad=True)
    ls = torch.tensor(0., requires_grad=True)
    ln = torch.tensor(-4., requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln], lr=lr)
    best_nll = 1e20; best = None
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        Q = torch.diag(d_inv) + 2 * t * Lt
        K = s * torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
        K = (K + K.T) / 2
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        if not torch.isfinite(nll): continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best = (t.item(), s.item(), no.item())
        nll.backward(); opt.step()
    return best


def learn_gamma(L, n, d_raw, yt, ti, n_iter=300, lr=0.03, init=None,
                gamma_init=-1.0):
    """Learn (tau, scale, noise, gamma) jointly via MLL for the paper's Q_gamma:

        Q_gamma = D^{-1} + 2*tau * D^{gamma/2} L D^{gamma/2}

    SIGN CONVENTION: the code uses D^{+gamma/2}; the paper writes D^{-gamma/2},
    so gamma_paper = -gamma_code. Reported gamma-hat values are in the paper's
    convention. (Same convention throughout this file and synthetic_two_regime.py.)

    where D = diag(d_i) is fixed from the auxiliary data. The diagonal is
    D^{-1} independent of gamma; only the edge wrapping D^{gamma/2} scales
    with gamma. With this form:
        gamma = 0   -> Q = D^{-1} + 2*tau*L           (=== semisup)
        gamma = -1  -> Q = D^{-1} + 2*tau*D^{-1/2} L D^{-1/2}  (=== het)
    so lgamma is a one-parameter family containing semisup and het exactly.

    Warm-start: if `init=(tau, scale, noise)` is provided, initialises those
    three at the supplied values. `gamma_init` is the initial value of gamma
    (default -1.0, the Proposition 1 factorisation point). Passing the het
    fit as `init` with `gamma_init=-1.0` makes lgamma start exactly at the
    het optimum, guaranteeing the returned NLL is no worse than het's NLL.
    Setting `gamma_init=0.0` instead (with or without `init`) starts from
    the semisup factorisation point and can be used to probe whether the
    learned gamma is robust to the choice of warm start.
    """
    Lt = torch.tensor(L)
    d_t = torch.tensor(np.maximum(d_raw, 1e-8))
    D_inv = torch.diag(1.0 / d_t)                     # fixed: D^{-1}
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
    # Initialise at gamma = -1 (het, the Proposition 1 factorisation point)
    # by default; overridable via gamma_init.
    gamma = torch.tensor(float(gamma_init), requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln, gamma], lr=lr)
    best_nll = 1e20; best_params = None
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        d_pow = torch.pow(d_t, gamma / 2.0)            # D^{gamma/2}
        d_pow = torch.clamp(d_pow, min=1e-4, max=1e4)
        D_pow = torch.diag(d_pow)
        Q = D_inv + 2 * t * D_pow @ Lt @ D_pow
        K = s * torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
        K = (K + K.T) / 2
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        if not torch.isfinite(nll):
            continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_params = (t.item(), s.item(), no.item(), gamma.item())
        nll.backward(); opt.step()
    return best_params


def learn_gamma_per_cluster(L, n, d_raw, yt, ti, cluster_labels,
                            n_iter=300, lr=0.03, init=None):
    """Piecewise-constant gamma(x): per-node gamma is indexed by cluster label.

    Precision Q = D^{-1} + 2*tau * diag(d^{gamma(x)/2}) L diag(d^{gamma(x)/2})
    where gamma(x) = gamma_k if node x is in cluster k.

    Returns (tau, scale, noise, gamma_k_array).
    """
    K = int(cluster_labels.max()) + 1
    Lt = torch.tensor(L)
    d_t = torch.tensor(np.maximum(d_raw, 1e-8))
    D_inv = torch.diag(1.0 / d_t)
    c_t = torch.tensor(cluster_labels, dtype=torch.long)
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
    gamma_k = torch.full((K,), -1., dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln, gamma_k], lr=lr)
    best_nll = 1e20; best_params = None
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        gamma_per_node = gamma_k[c_t]
        d_pow = torch.pow(d_t, gamma_per_node / 2.0)
        d_pow = torch.clamp(d_pow, min=1e-4, max=1e4)
        Q = D_inv + 2 * t * (d_pow[:, None] * Lt * d_pow[None, :])
        K_ker = s * torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
        K_ker = (K_ker + K_ker.T) / 2
        Kn = K_ker[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        if not torch.isfinite(nll):
            continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_params = (t.item(), s.item(), no.item(),
                           gamma_k.detach().cpu().numpy().copy())
        nll.backward(); opt.step()
    return best_params


def learn_fuglstad_independent(L, n, yt, ti, n_iter=200, lr=0.03):
    """Learn n per-node kappa_i independently via MLL."""
    Lt = torch.tensor(L)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    lt = torch.tensor(0., requires_grad=True)
    ls = torch.tensor(0., requires_grad=True)
    ln = torch.tensor(-4., requires_grad=True)
    lk = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln, lk], lr=lr)
    best_nll = 1e20; best_params = None
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        kappa = torch.exp(lk)
        D_inv = torch.diag(1.0 / kappa); D_sqrt = torch.diag(torch.sqrt(kappa))
        Q = D_inv + 2 * t * D_sqrt @ Lt @ D_sqrt
        K = s * torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
        K = (K + K.T) / 2
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        if not torch.isfinite(nll):
            continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_params = (t.item(), s.item(), no.item(), kappa.detach().numpy().copy())
        nll.backward(); opt.step()
    return best_params


def learn_fuglstad_regularized(L, n, d_raw, yt, ti, lam, n_iter=200, lr=0.03):
    """Learn n per-node kappa_i with smoothness penalty lam * sum_{ij} (log k_i - log k_j)^2."""
    A_sparse = (np.abs(L) > 1e-10).astype(float)
    np.fill_diagonal(A_sparse, 0)
    Lt = torch.tensor(L)
    At = torch.tensor(A_sparse)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    # Initialise log-kappa from gamma=-1 (hetero)
    lk_init = -np.log(np.maximum(d_raw, 1e-8))
    lt = torch.tensor(0., requires_grad=True)
    ls = torch.tensor(0., requires_grad=True)
    ln = torch.tensor(-4., requires_grad=True)
    lk = torch.tensor(lk_init, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([lt, ls, ln, lk], lr=lr)
    best_nll = 1e20; best_params = None
    for _ in range(n_iter):
        opt.zero_grad()
        t, s, no = torch.exp(lt), torch.exp(ls), torch.exp(ln)
        kappa = torch.exp(lk)
        D_inv = torch.diag(1.0 / kappa); D_sqrt = torch.diag(torch.sqrt(kappa))
        Q = D_inv + 2 * t * D_sqrt @ Lt @ D_sqrt
        K = s * torch.linalg.inv(Q + 1e-8 * torch.eye(n, dtype=torch.float64))
        K = (K + K.T) / 2
        Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
        data_nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        # Smoothness penalty on log-kappa
        diff = lk.unsqueeze(0) - lk.unsqueeze(1)  # [n, n]
        penalty = 0.5 * lam * torch.sum(At * diff**2)
        loss = data_nll + penalty
        if not torch.isfinite(loss):
            continue
        if data_nll.item() < best_nll:
            best_nll = data_nll.item()
            best_params = (t.item(), s.item(), no.item(), kappa.detach().numpy().copy())
        loss.backward(); opt.step()
    return best_params


def learn_spectral_poly(L, n, yt, ti, degree=3, n_iter=300, lr=0.03):
    """Learn polynomial spectral filter g(lambda) = sum_p softplus(beta_p) * lambda^p."""
    evals, evecs = np.linalg.eigh(L)
    evals = np.maximum(evals, 0)
    evals_t = torch.tensor(evals)
    V = torch.tensor(evecs)
    yy = torch.tensor(yt)
    ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
    betas = torch.zeros(degree + 1, dtype=torch.float64, requires_grad=True)
    ln = torch.tensor(-4., dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([betas, ln], lr=lr)
    best_nll = 1e20; best_params = None
    lam_powers = torch.stack([evals_t ** p for p in range(degree + 1)], dim=1)
    for _ in range(n_iter):
        opt.zero_grad()
        noise = torch.exp(ln)
        coeffs = torch.nn.functional.softplus(betas)
        g_vals = lam_powers @ coeffs
        K = V @ torch.diag(g_vals) @ V.T; K = (K + K.T) / 2
        Kn = K[ot][:, ot] + noise * torch.eye(m, dtype=torch.float64)
        try:
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
            Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze()
        nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
        if not torch.isfinite(nll):
            continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_params = (coeffs.detach().numpy().copy(), noise.item(),
                           K.detach().numpy())
        nll.backward(); opt.step()
    return best_params


def run_bktr_manual(coords, L, d_raw, y_norm, ti, xi, n_samples=300):
    """BKTR-style spatial GP: SE kernel on coordinates, grid search over hyperparams."""
    n = len(y_norm)
    n_train = len(ti)
    dist_sq = cdist(coords, coords, 'sqeuclidean')
    median_dist = np.median(dist_sq[dist_sq > 0])
    y_train = y_norm[ti]
    best_nll = 1e20; best_params = None
    for log_ell in np.linspace(np.log(median_dist * 0.1), np.log(median_dist * 10), 10):
        ell = np.exp(log_ell)
        K_spatial = np.exp(-dist_sq / (2 * ell))
        K_spatial = (K_spatial + K_spatial.T) / 2 + 1e-6 * np.eye(n)
        K_train = K_spatial[np.ix_(ti, ti)]
        for log_noise in np.linspace(-6, 2, 15):
            noise = np.exp(log_noise)
            for log_scale in np.linspace(-2, 4, 12):
                scale = np.exp(log_scale)
                K_y = scale * K_train + noise * np.eye(n_train)
                try:
                    Lc = np.linalg.cholesky(K_y)
                    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y_train))
                    nll = 0.5 * y_train @ alpha + np.sum(np.log(np.diag(Lc)))
                    if nll < best_nll:
                        best_nll = nll
                        K_full = scale * K_spatial
                        best_params = (K_full, noise)
                except np.linalg.LinAlgError:
                    continue
    if best_params is None:
        return None, None
    K_full, noise = best_params
    K_train_full = K_full[np.ix_(ti, ti)] + noise * np.eye(n_train)
    Lc = np.linalg.cholesky(K_train_full)
    alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, y_train))
    mu_pred = K_full[:, ti] @ alpha
    v = np.linalg.solve(Lc, K_full[ti, :])
    var_pred = np.diag(K_full) - np.sum(v ** 2, axis=0) + noise
    return mu_pred, var_pred


# =====================================================================
# Dataset loaders — each returns (A, L, n, y_norm, d_raw, coords_km, feat)
# =====================================================================

def load_metrla():
    DATA_DIR = 'graph_gp/data_cache/metr_la'
    with open(f'{DATA_DIR}/adj_mx.pkl', 'rb') as fp:
        _, _, adj = pickle.load(fp, encoding='latin1')
    with h5py.File(f'{DATA_DIR}/metr-la.h5', 'r') as fp:
        speeds = fp['df']['block0_values'][:]
    n = speeds.shape[1]
    A = (adj > 0).astype(float); A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    deg = A.sum(1)
    L = compute_laplacian(A)
    # Auxiliary d_i (temporal coefficient of variation) is computed from the
    # readings AFTER the target day, so it shares no observation with the
    # target (the mean over the first 288 readings). This removes the
    # target/auxiliary leakage flagged in review; CA Weather is already clean
    # by construction (2022 auxiliary vs 2023 target).
    aux = speeds[288:, :]
    d_raw = aux.std(axis=0) / (aux.mean(axis=0) + 1e-8)
    d_raw = d_raw / d_raw.mean(); d_raw = np.clip(d_raw, 0.1, 10)
    y = speeds[:288, :].mean(axis=0); y_norm = (y - y.mean()) / y.std()
    # Coordinates
    import pandas as pd
    try:
        loc = pd.read_csv(f'{DATA_DIR}/graph_sensor_locations.csv')
        coords_km = loc[['latitude', 'longitude']].values.astype(np.float64).copy()
        coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    except Exception:
        coords_km = np.zeros((n, 2))
    # Features for DK
    feat = np.stack([(d_raw - d_raw.mean()) / (d_raw.std() + 1e-8),
                     np.log(deg.clip(1))], axis=1)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    return A, L, n, y_norm, d_raw, coords_km, feat


def load_pemsbay():
    DATA_DIR = 'graph_gp/data_cache/pems_bay'
    with open(f'{DATA_DIR}/adj_mx_bay.pkl', 'rb') as fp:
        sensor_ids, _, adj = pickle.load(fp, encoding='latin1')
    with h5py.File(f'{DATA_DIR}/pems-bay.h5', 'r') as fp:
        speeds = fp['speed']['block0_values'][:]
    n = speeds.shape[1]
    A = (adj > 0).astype(float); A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    deg = A.sum(1)
    L = compute_laplacian(A)
    # Auxiliary d_i from the readings AFTER the target day (disjoint from the
    # target, which is the mean over the first 288 readings) -- same
    # leakage-free construction as METR-LA.
    aux = speeds[288:, :]
    d_raw = aux.std(axis=0) / (aux.mean(axis=0) + 1e-8)
    d_raw = d_raw / d_raw.mean(); d_raw = np.clip(d_raw, 0.1, 10)
    y = speeds[:288, :].mean(axis=0); y_norm = (y - y.mean()) / y.std()
    # Coordinates
    coords_km = np.zeros((n, 2))
    loc_file = f'{DATA_DIR}/graph_sensor_locations_bay.csv'
    if os.path.exists(loc_file):
        with open(loc_file) as fp:
            lines = fp.readlines()
        sensor_coords = {}
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                sensor_coords[parts[0]] = (float(parts[1]), float(parts[2]))
        coords = np.array([sensor_coords.get(str(sid), (37.5, -122.0))
                           for sid in sensor_ids])
        coords_km = coords.copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    else:
        evals, evecs = np.linalg.eigh(L)
        coords_km = evecs[:, 1:3] * 100
    feat = np.stack([(d_raw - d_raw.mean()) / (d_raw.std() + 1e-8),
                     np.log(deg.clip(1))], axis=1)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    return A, L, n, y_norm, d_raw, coords_km, feat


def load_laqn():
    """LAQN NO2 (2024-06-15), read from the shipped cache -- no live API call
    and no geometric_kernels dependency. Run graph_gp/fetch_laqn_cache.py once
    on a machine with internet, then copy data_cache/ into graph_gp/data_cache/."""
    cand = ['graph_gp/data_cache', 'data_cache',
            os.path.join(os.path.dirname(__file__), 'data_cache')]
    cdir = next((c for c in cand
                 if os.path.exists(os.path.join(c, 'laqn_sites.json'))), None)
    if cdir is None:
        raise FileNotFoundError(
            'LAQN cache not found. Run graph_gp/fetch_laqn_cache.py on a machine '
            'with internet, then copy data_cache/ to graph_gp/data_cache/.')
    with open(os.path.join(cdir, 'laqn_sites.json')) as f:
        sites = json.load(f)
    with open(os.path.join(cdir, 'laqn_NO2_2024-06-15.json')) as f:
        values = json.load(f)
    sites = [s for s in sites if s['code'] in values]
    ys = np.array([values[s['code']] for s in sites])
    type_nums = np.array([{'Rural': 0, 'Suburban': 1, 'Urban Background': 2,
                            'Industrial': 3, 'Roadside': 4, 'Kerbside': 5
                            }.get(s['type'], 2) for s in sites], dtype=float)
    coords = np.array([[s['lat'], s['lon']] for s in sites])
    ys_n = (ys - ys.mean()) / ys.std()
    # build_graph(coords, threshold_km=5.0): lat/lon -> km (x111, x70 at London
    # latitude), then connect pairs closer than 5 km.
    ckm = coords.copy(); ckm[:, 0] *= 111.0; ckm[:, 1] *= 70.0
    A = (cdist(ckm, ckm) < 5.0).astype(float); np.fill_diagonal(A, 0.0)
    deg = A.sum(1); conn = deg > 0
    ys_n = ys_n[conn]; type_nums = type_nums[conn]
    A = A[np.ix_(conn, conn)]; n = int(conn.sum())
    L = compute_laplacian(A)
    d_raw = type_nums / type_nums.mean(); d_raw = np.clip(d_raw + 0.1, 0.1, 10)
    coords_km = coords[conn].copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 70
    feat = np.stack([(d_raw - d_raw.mean()) / (d_raw.std() + 1e-8),
                     np.log(A.sum(1).clip(1))], axis=1)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    return A, L, n, ys_n, d_raw, coords_km, feat


def load_caweather(day=195):
    """CA Weather: d_i from 2022 temporal variance, signal from 2023 day-of-year."""
    def load_ghcn_year(year_file, id_to_info):
        import pandas as pd
        rows = []
        with gzip.open(year_file, 'rt') as fh:
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    sid = parts[0]
                    if sid in id_to_info and parts[2] == 'TMAX':
                        rows.append({'id': sid, 'date': parts[1],
                                     'tmax': float(parts[3]) / 10})
        return __import__('pandas').DataFrame(rows)

    stations = []
    with open('graph_gp/data_cache/weather/ghcnd-stations.txt') as fh:
        for line in fh:
            sid = line[:11].strip()
            lat = float(line[12:20]); lon = float(line[21:30])
            if sid.startswith('US') and 32 < lat < 42 and -125 < lon < -114:
                stations.append({'id': sid, 'lat': lat, 'lon': lon})
    id_to_info = {s['id']: s for s in stations}

    df_2022 = load_ghcn_year('graph_gp/data_cache/weather/2022.csv.gz', id_to_info)
    pivot_2022 = df_2022.pivot_table(index='date', columns='id', values='tmax')
    df_2023 = load_ghcn_year('graph_gp/data_cache/weather/2023.csv.gz', id_to_info)
    pivot_2023 = df_2023.pivot_table(index='date', columns='id', values='tmax')

    good_2022 = set(pivot_2022.columns[pivot_2022.notna().sum() > 300])
    good_2023 = set(pivot_2023.columns[pivot_2023.notna().sum() > 300])
    common = sorted(good_2022 & good_2023)
    pivot_2022 = pivot_2022[common].fillna(pivot_2022[common].mean())
    pivot_2023 = pivot_2023[common].fillna(pivot_2023[common].mean())

    coords = np.array([[id_to_info[sid]['lat'], id_to_info[sid]['lon']]
                        for sid in common])
    coords_km = coords.copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    Dd = cdist(coords_km, coords_km)
    A = ((Dd < 50) & (Dd > 0)).astype(float)
    deg = A.sum(1); conn = deg > 0
    A = A[np.ix_(conn, conn)]
    vals_2022 = pivot_2022.values[:, np.where(conn)[0]]
    vals_2023 = pivot_2023.values[:, np.where(conn)[0]]
    coords_km = coords_km[conn]
    n = int(conn.sum()); deg = A.sum(1)
    L = compute_laplacian(A)

    d_raw = vals_2022.var(axis=0)
    d_raw = d_raw / d_raw.mean(); d_raw = np.clip(d_raw, 0.1, 10)

    y = vals_2023[day, :]
    valid = ~np.isnan(y)
    if not valid.all():
        y = np.where(valid, y, np.nanmean(y))
    y_norm = (y - y.mean()) / (y.std() + 1e-8)

    feat = np.stack([(d_raw - d_raw.mean()) / (d_raw.std() + 1e-8),
                     np.log(deg.clip(1))], axis=1)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    return A, L, n, y_norm, d_raw, coords_km, feat


# =====================================================================
# Main: run all methods on one (dataset, seed)
# =====================================================================

def load_pm25(k=8):
    """EPA daily PM2.5 (parameter code 88101) for 2023. Returns the single-
    connected k-NN graph over ~988 CONUS monitoring sites. Block-diagonal
    variants are produced downstream by apply_blocking() when --n_clusters>1.
    """
    import pandas as pd, glob as _glob
    csv_path = 'graph_gp/data_cache/pm25/daily_88101_2023/daily_88101_2023.csv'
    if not os.path.exists(csv_path):
        # The EPA zip may unpack with the CSV at the pm25/ root rather than in
        # the daily_88101_2023/ subdir; find it wherever it landed.
        hits = _glob.glob('graph_gp/data_cache/pm25/**/daily_88101_2023.csv',
                          recursive=True)
        if hits:
            csv_path = hits[0]
    df = pd.read_csv(csv_path, usecols=['State Code', 'County Code', 'Site Num',
                                        'POC', 'Latitude', 'Longitude',
                                        'Arithmetic Mean', 'Date Local'])
    df['site'] = (df['State Code'].astype(str) + '_' +
                  df['County Code'].astype(str) + '_' +
                  df['Site Num'].astype(str))
    daily = df.groupby(['site', 'Date Local'])['Arithmetic Mean'].mean().reset_index()
    # Disjoint target/auxiliary split. Interleave the sorted daily series per
    # site: even-ranked days define the target level (annual mean); odd-ranked
    # days define the auxiliary variability (annual standard deviation -- per
    # the paper, NOT the coefficient of variation). Interleaving keeps both
    # statistics representative of the full year while sharing no observation,
    # removing the same-year leakage flagged in review.
    daily = daily.sort_values(['site', 'Date Local'])
    daily['day_rank'] = daily.groupby('site').cumcount()
    tgt = (daily[daily['day_rank'] % 2 == 0]
           .groupby('site')['Arithmetic Mean'].mean())
    aux = (daily[daily['day_rank'] % 2 == 1]
           .groupby('site')['Arithmetic Mean'].std())
    stats = pd.DataFrame({'mean': tgt, 'std': aux}).reset_index()
    stats = stats.dropna()
    locs = df.drop_duplicates('site')[['site', 'Latitude', 'Longitude']]
    stats = stats.merge(locs, on='site')
    stats = stats[(stats['Latitude'].between(24, 50)) & (stats['Longitude'].between(-125, -66))]
    stats = stats.reset_index(drop=True)
    n = len(stats)

    y = stats['mean'].values.astype(np.float64)
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    d_raw = stats['std'].values.astype(np.float64)   # annual std (disjoint days)
    d_raw = d_raw / d_raw.mean()
    d_raw = np.clip(d_raw, 0.1, 10)

    coords_km = np.stack([stats['Latitude'].values * 111.0,
                          stats['Longitude'].values * 85.0], axis=1).astype(np.float64)

    dists = cdist(coords_km, coords_km)
    np.fill_diagonal(dists, np.inf)
    nn = np.argpartition(dists, k, axis=1)[:, :k]
    A = np.zeros((n, n))
    for i in range(n):
        for j in nn[i]:
            A[i, j] = 1.0; A[j, i] = 1.0
    deg = A.sum(1)
    L = compute_laplacian(A)

    feat = np.stack([(d_raw - d_raw.mean()) / (d_raw.std() + 1e-8),
                     np.log(deg.clip(1))], axis=1)
    feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-8)
    return A, L, n, y_norm, d_raw, coords_km, feat


LOADERS = {'metrla': load_metrla, 'pemsbay': load_pemsbay, 'laqn': load_laqn,
           'pm25': load_pm25}
REG_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def run_all(A, L, n, y_norm, d_raw, coords_km, feat, seed, dataset, skip=None):
    """Run every method and return result dict. skip: list of method names to skip."""
    skip = set(skip or [])
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]; xi = perm[int(0.7 * n):]
    yt = y_norm[ti]

    result = {'seed': seed, 'dataset': dataset, 'n': n}
    ntest = max(1, len(xi))   # held-out count; NLL is a total, report per-point

    def _pp(key):
        """Per-point NLL for the progress prints (total NLL / n_test)."""
        v = result.get(key)
        return f"{v / ntest:.3f}" if isinstance(v, (int, float)) else "nan"

    print(f"  [{dataset} seed={seed} n={n} skip={skip or 'none'}]", flush=True)

    # --- 1. Stationary ---
    tau, scale, noise = fit_stationary(L, n, yt, ti)
    result['tau'] = tau
    K_s = build_kernel(L, n, tau, scale)
    mu_s, var_s, _ = gp_posterior(K_s, yt, ti, noise)
    result.update(compute_metrics('stat', mu_s, var_s, y_norm, xi))
    print(f"    stat: NLL/pt={_pp('nll_stat')}", flush=True)

    # --- 2. Het (gamma = -1 factorisation): K = s * D^{1/2} (I + 2*tau*L)^{-1} D^{1/2}
    # Fit jointly via MLL for (tau, scale, noise).
    D_sqrt = np.diag(np.sqrt(np.maximum(d_raw, 1e-8)))
    t_h, s_h, no_h = fit_het_joint(L, n, d_raw, yt, ti)
    K_h = s_h * D_sqrt @ np.linalg.inv(np.eye(n) + 2 * t_h * L) @ D_sqrt
    K_h = (K_h + K_h.T) / 2
    mu_h, var_h, _ = gp_posterior(K_h, yt, ti, no_h)
    result['tau_het'] = t_h; result['scale_het'] = s_h; result['noise_het'] = no_h
    result.update(compute_metrics('het', mu_h, var_h, y_norm, xi))
    print(f"    het:  NLL/pt={_pp('nll_het')}", flush=True)

    # --- 2b. Semisup (gamma = 0): K = s * (D^{-1} + 2*tau*L)^{-1}
    # Fit jointly via MLL for (tau, scale, noise).
    t_s, s_s, no_s = fit_semisup_joint(L, n, d_raw, yt, ti)
    Q_ss = np.diag(1.0 / np.maximum(d_raw, 1e-8)) + 2 * t_s * L
    K_ss = s_s * np.linalg.inv(Q_ss + 1e-8 * np.eye(n))
    K_ss = (K_ss + K_ss.T) / 2
    mu_ss, var_ss, _ = gp_posterior(K_ss, yt, ti, no_s)
    result['tau_ss'] = t_s; result['scale_ss'] = s_s; result['noise_ss'] = no_s
    result.update(compute_metrics('semisup', mu_ss, var_ss, y_norm, xi))
    print(f"    fem:  NLL/pt={_pp('nll_semisup')}", flush=True)

    # --- 3. Matern-0.5 ---
    try:
        K_m05, noise_m05, _ = learn_gk_matern(A, n, yt, ti, nu=0.5)
        mu_m05, var_m05, _ = gp_posterior(K_m05, yt, ti, noise_m05)
        result.update(compute_metrics('m05', mu_m05, var_m05, y_norm, xi))
    except Exception as e:
        print(f"    m05 FAILED: {e}")
        result.update({f'{k}_m05': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
    print(f"    m05:  NLL/pt={_pp('nll_m05')}", flush=True)

    # --- 4. Matern-1.5 ---
    try:
        K_m15, noise_m15, _ = learn_gk_matern(A, n, yt, ti, nu=1.5)
        mu_m15, var_m15, _ = gp_posterior(K_m15, yt, ti, noise_m15)
        result.update(compute_metrics('m15', mu_m15, var_m15, y_norm, xi))
    except Exception as e:
        print(f"    m15 FAILED: {e}")
        result.update({f'{k}_m15': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
    print(f"    m15:  NLL/pt={_pp('nll_m15')}", flush=True)

    # --- 4b. Learned nu (Matern with nu optimised via MLL) ---
    if 'lnu' in skip:
        print("    lnu:  SKIPPED", flush=True)
    else:
        try:
            K_lnu, noise_lnu, info_lnu = learn_gk_matern_nu(A, n, yt, ti)
            if K_lnu is not None:
                mu_lnu, var_lnu, _ = gp_posterior(K_lnu, yt, ti, noise_lnu)
                result.update(compute_metrics('lnu', mu_lnu, var_lnu, y_norm, xi))
                result['nu_learned'] = info_lnu['nu']
            else:
                result.update({f'{k}_lnu': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
                result['nu_learned'] = float('nan')
        except Exception as e:
            print(f"    lnu FAILED: {e}")
            result.update({f'{k}_lnu': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
            result['nu_learned'] = float('nan')
        print(f"    lnu:  NLL/pt={_pp('nll_lnu')} nu={result.get('nu_learned', 'nan')}", flush=True)

    # --- 5. Learned gamma ---
    # Warm-start from the het fit (which is Q_gamma at gamma=-1). Guarantees
    # lgamma NLL <= het NLL because Adam starts at the het optimum and can
    # only move to lower-NLL regions.
    params = learn_gamma(L, n, d_raw, yt, ti, init=(t_h, s_h, no_h))
    if params is not None:
        tau_g, scale_g, noise_g, gamma_opt = params
        # Build Q_gamma using the paper's form (matches learn_gamma):
        #   Q = D^{-1} + 2*tau * D^{gamma/2} L D^{gamma/2}
        d_safe = np.maximum(d_raw, 1e-8)
        d_pow = np.power(d_safe, gamma_opt / 2.0)
        d_pow = np.clip(d_pow, 1e-4, 1e4)
        Q_lg = np.diag(1.0 / d_safe) + 2 * tau_g * np.diag(d_pow) @ L @ np.diag(d_pow)
        K_lg = scale_g * np.linalg.inv(Q_lg + 1e-8 * np.eye(n))
        K_lg = (K_lg + K_lg.T) / 2
        mu_lg, var_lg, _ = gp_posterior(K_lg, yt, ti, noise_g)
        result.update(compute_metrics('lgamma', mu_lg, var_lg, y_norm, xi))
        result['gamma_learned'] = gamma_opt
    else:
        result.update({f'{k}_lgamma': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})

        result['gamma_learned'] = float('nan')
    print(f"    lgam: NLL/pt={_pp('nll_lgamma')} gamma={result['gamma_learned']:.3f}", flush=True)

    # --- 6. Fuglstad (unregularised) ---
    try:
        fugl_params = learn_fuglstad_independent(L, n, yt, ti)
        if fugl_params is not None:
            tau_f, scale_f, noise_f, kappa_f = fugl_params
            K_f = build_kernel(L, n, tau_f, scale_f, kappa_f)
            mu_f, var_f, _ = gp_posterior(K_f, yt, ti, noise_f)
            result.update(compute_metrics('fuglstad', mu_f, var_f, y_norm, xi))
            result['kappa_std_fuglstad'] = float(kappa_f.std())
        else:
            result.update({f'{k}_fuglstad': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
    except Exception as e:
        print(f"    fuglstad FAILED: {e}")
        result.update({f'{k}_fuglstad': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
    print(f"    fugl: NLL/pt={_pp('nll_fuglstad')}", flush=True)

    # --- 7. Fuglstad regularised (lambda sweep) ---
    for lam in REG_LAMBDAS:
        tag = f'fuglreg_lam{lam}'
        try:
            freg = learn_fuglstad_regularized(L, n, d_raw, yt, ti, lam)
            if freg is not None:
                tau_fr, scale_fr, noise_fr, kappa_fr = freg
                K_fr = build_kernel(L, n, tau_fr, scale_fr, kappa_fr)
                mu_fr, var_fr, _ = gp_posterior(K_fr, yt, ti, noise_fr)
                result.update(compute_metrics(tag, mu_fr, var_fr, y_norm, xi))
                result[f'kappa_std_{tag}'] = float(kappa_fr.std())
            else:
                result.update({f'{k}_{tag}': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
        except Exception as e:
            print(f"    {tag} FAILED: {e}")
            result.update({f'{k}_{tag}': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
    print(f"    fuglreg: done (5 lambdas)", flush=True)

    # --- 8. Deep Kernel ---
    if 'dk' in skip:
        print("    dk:   SKIPPED", flush=True)
    else:
        try:
            torch.manual_seed(seed)
            K_dk, noise_dk = train_deep_kernel(feat, yt, ti, n)
            mu_dk, var_dk, _ = gp_posterior(K_dk, yt, ti, noise_dk)
            result.update(compute_metrics('dk', mu_dk, var_dk, y_norm, xi))
        except Exception as e:
            print(f"    dk FAILED: {e}")
            result.update({f'{k}_dk': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
        print(f"    dk:   NLL/pt={_pp('nll_dk')}", flush=True)

    # --- 9. Paciorek-Schervish ---
    if 'ps' in skip:
        print("    ps:   SKIPPED", flush=True)
    else:
        try:
            K_ps, noise_ps, _ = learn_ps_kernel(coords_km, d_raw, yt, ti, n)
            mu_ps, var_ps, _ = gp_posterior(K_ps, yt, ti, noise_ps)
            result.update(compute_metrics('ps', mu_ps, var_ps, y_norm, xi))
        except Exception as e:
            print(f"    ps FAILED: {e}")
            result.update({f'{k}_ps': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
        print(f"    ps:   NLL/pt={_pp('nll_ps')}", flush=True)

    # --- 10. BKTR (spatial GP) ---
    if 'bktr' in skip:
        print("    bktr: SKIPPED", flush=True)
    else:
        try:
            mu_bk, var_bk = run_bktr_manual(coords_km, L, d_raw, y_norm, ti, xi)
            if mu_bk is not None:
                result.update(compute_metrics('bktr', mu_bk, var_bk, y_norm, xi))
            else:
                result.update({f'{k}_bktr': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
        except Exception as e:
            print(f"    bktr FAILED: {e}")
            result.update({f'{k}_bktr': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
        print(f"    se_kriging: NLL/pt={_pp('nll_bktr')}", flush=True)

    # --- 11. Spectral Polynomial GP ---
    if 'specpoly' in skip:
        print("    spec: SKIPPED", flush=True)
    else:
        try:
            sp = learn_spectral_poly(L, n, yt, ti, degree=3)
            if sp:
                coeffs_sp, noise_sp, K_sp = sp
                mu_sp, var_sp, _ = gp_posterior(K_sp, yt, ti, noise_sp)
                result.update(compute_metrics('specpoly', mu_sp, var_sp, y_norm, xi))
            else:
                result.update({f'{k}_specpoly': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
        except Exception as e:
            print(f"    specpoly FAILED: {e}")
            result.update({f'{k}_specpoly': float('nan') for k in ['nll', 'rmse', 'recal', 'c']})
        print(f"    spec: NLL/pt={_pp('nll_specpoly')}", flush=True)

    return result


# =====================================================================
# Entry point
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all methods on a dataset')
    parser.add_argument('--dataset', required=True,
                        choices=['metrla', 'pemsbay', 'laqn', 'caweather', 'pm25'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--day', type=int, default=195,
                        help='Day of year for caweather (default: 195 = Jul 14)')
    parser.add_argument('--skip', nargs='*', default=[],
                        help='Methods to skip (e.g. --skip bktr dk)')
    parser.add_argument('--outdir', default='graph_gp/results')
    parser.add_argument('--n_clusters', type=int, default=1,
                        help='If > 1, partition nodes by K-means on coords and '
                             'build block-diagonal Laplacian. Only per-block '
                             'lgamma is fit (runs in a few minutes, not hours). '
                             'K=1 (default) runs the full pipeline on the single '
                             'connected k-NN graph.')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- Load the dataset (single-graph form) ---
    if args.dataset == 'caweather':
        A, L, n, y_norm, d_raw, coords_km, feat = load_caweather(day=args.day)
        ds_tag = f'caweather_day{args.day}'
    else:
        A, L, n, y_norm, d_raw, coords_km, feat = LOADERS[args.dataset]()
        ds_tag = args.dataset

    # --- Dispatch on --n_clusters ---
    if args.n_clusters > 1:
        # Block-diagonal variant: only per-block lgamma is computed.
        A, L, cluster_labels = apply_blocking(coords_km, args.n_clusters)
        print(f"[{ds_tag} seed={args.seed} K={args.n_clusters}  "
              f"min-deg={int(A.sum(1).min())}]", flush=True)
        rng = np.random.RandomState(args.seed)
        perm = rng.permutation(n)
        ti = np.sort(perm[:int(0.7 * n)])
        xi = np.sort(perm[int(0.7 * n):])
        yt = y_norm[ti]
        t_h, s_h, no_h = fit_het_joint(L, n, d_raw, yt, ti)
        params_k = learn_gamma_per_cluster(L, n, d_raw, yt, ti, cluster_labels,
                                           init=(t_h, s_h, no_h))
        tau_k, scale_k, noise_k, gamma_k_arr = params_k
        d_safe = np.maximum(d_raw, 1e-8)
        gamma_per_node = gamma_k_arr[cluster_labels]
        d_pow = np.clip(np.power(d_safe, gamma_per_node / 2.0), 1e-4, 1e4)
        Q_lgk = np.diag(1.0 / d_safe) + 2 * tau_k * np.outer(d_pow, d_pow) * L
        K_lgk = scale_k * np.linalg.inv(Q_lgk + 1e-8 * np.eye(n))
        K_lgk = (K_lgk + K_lgk.T) / 2
        mu_lgk, var_lgk, _ = gp_posterior(K_lgk, yt, ti, noise_k)
        result = {'seed': args.seed, 'dataset': args.dataset,
                  'n_clusters': args.n_clusters, 'n': n,
                  'cluster_sizes': np.bincount(cluster_labels).tolist()}
        if args.dataset == 'caweather':
            result['day'] = args.day
        result.update(compute_metrics('lgamma_block', mu_lgk, var_lgk, y_norm, xi))
        result['gamma_per_cluster'] = gamma_k_arr.tolist()
        result['tau_block'] = tau_k
        result['scale_block'] = scale_k
        result['noise_block'] = noise_k
        print(f"  lgamma_block NLL/pt={result['nll_lgamma_block']/max(1, n-int(0.7*n)):.3f}  "
              f"RMSE={result['rmse_lgamma_block']:.3f}  "
              f"gammas=[{', '.join(f'{g:+.2f}' for g in gamma_k_arr)}]", flush=True)
        day_suffix = f"_day{args.day}" if args.dataset == 'caweather' else ""
        outfile = f"{args.outdir}/results_{args.dataset}{day_suffix}_k{args.n_clusters}_seed{args.seed}.json"
    else:
        # K=1: full pipeline (all methods).
        result = run_all(A, L, n, y_norm, d_raw, coords_km, feat,
                         args.seed, ds_tag, skip=args.skip)
        if args.dataset == 'caweather':
            result['day'] = args.day
            outfile = f"{args.outdir}/results_caweather_day{args.day}_seed{args.seed}.json"
        else:
            outfile = f"{args.outdir}/results_{args.dataset}_seed{args.seed}.json"

    with open(outfile, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten to {outfile}", flush=True)

    # Print summary. NLL/recal are stored as totals over the held-out nodes;
    # report them per-point (NLL/pt) to match the paper's tables. Display names
    # follow the paper: semisup -> fem; the bktr column is an SE-kernel kriging
    # baseline (the cited BKTR method is run_bktr_baseline.py) -> se_kriging.
    methods = ['stat', 'het', 'semisup', 'm05', 'm15', 'lnu', 'lgamma', 'fuglstad',
               'dk', 'ps', 'bktr', 'specpoly']
    DISPLAY = {'semisup': 'fem', 'bktr': 'se_kriging'}
    ntest = max(1, n - int(0.7 * n))
    print(f"\n{'Method':>15s} {'NLL/pt':>10s} {'RMSE':>8s} {'recNLL/pt':>10s}")
    print('-' * 48)
    for m in methods:
        nll = result.get(f'nll_{m}', float('nan')) / ntest
        rmse = result.get(f'rmse_{m}', float('nan'))
        recal = result.get(f'recal_{m}', float('nan')) / ntest
        print(f"{DISPLAY.get(m, m):>15s} {nll:10.3f} {rmse:8.3f} {recal:10.3f}")

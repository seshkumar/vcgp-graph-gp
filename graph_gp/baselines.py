"""Baseline kernels for graph GP regression.

All baselines use the same GP inference (exact posterior mean + variance)
and learn hyperparameters via marginal likelihood. They differ only in
the kernel function.

Baselines:
  1. Diffusion kernel (heat): K = scale * exp(-tau * L)
  2. Graph Matern-nu: K = scale * (2nu/kappa^2 I + L)^{-nu}  (via GeometricKernels)
  3. Proximal kernel (ours): K = scale * (A^{-1} + 2*tau*L)^{-1}

Usage:
    from baselines import DiffusionKernel, learn_kernel_gp, proximal_gp_two_stage
"""

import numpy as np
import torch
from scipy.linalg import expm
from scipy import sparse
from scipy.sparse.linalg import splu

# geometric_kernels is only used by learn_gk_matern / learn_gk_matern_nu.
# Deferred import so that consumers who don't call those functions (e.g. the
# Fisher-ridge runner) avoid the geometric_kernels import cost / warnings.


# -------------------------------------------------------------------
# Proximal kernel implementations (multiple backends)
# -------------------------------------------------------------------

def proximal_kernel_dense(L, a, tau, scale=1.0):
    """Dense O(n^3): K = scale * (A^{-1} + 2*tau*L)^{-1}. For n < 500."""
    n = L.shape[0]
    A_inv = np.diag(1.0 / (a + 1e-10))
    Q = A_inv + 2 * tau * L
    K = scale * np.linalg.inv(Q)
    return (K + K.T) / 2


def proximal_kernel_cholesky(L_sparse, a, tau, scale=1.0):
    """Sparse Cholesky O(n^{3/2}): factor Q = A^{-1} + 2*tau*L. For n > 500."""
    n = L_sparse.shape[0]
    A_inv_diag = 1.0 / (a + 1e-10)
    Q_sparse = sparse.diags(A_inv_diag) + 2 * tau * L_sparse
    # SuperLU factorisation of Q (sparse LU, not Cholesky, but scipy doesn't
    # have sparse Cholesky natively — use scikit-sparse if available)
    lu = splu(Q_sparse.tocsc())
    # Solve Q K = scale * I column by column (or use selected inversion)
    # For GP inference we only need K @ alpha and diag(K), not full K.
    return lu, Q_sparse


def proximal_gp_sparse(L_sparse, a, tau, scale, noise, y_obs, obs_idx, n):
    """GP inference via sparse precision Q without forming dense K.

    Uses the GMRF structure: Q = A^{-1} + 2*tau*L is sparse.
    Posterior mean: mu = K_*n (K_nn + sigma^2 I)^{-1} y
    But K_nn = Q_obs^{-1} where Q_obs is the observed subblock + noise.

    For exact GP this still requires forming K_nn (m x m dense).
    The GMRF advantage is for marginal likelihood: log det Q is cheap.
    """
    m = len(obs_idx)
    A_inv_diag = 1.0 / (a + 1e-10)
    Q = sparse.diags(A_inv_diag) + 2 * tau * L_sparse

    # For small m (training set), form K_nn = Q^{-1}[obs, obs] explicitly
    lu = splu(Q.tocsc())
    K_cols = np.zeros((n, m))
    for j, idx in enumerate(obs_idx):
        e_j = np.zeros(n)
        e_j[idx] = scale
        K_cols[:, j] = lu.solve(e_j)

    K_nn = K_cols[obs_idx, :] + noise * np.eye(m)
    K_star_n = K_cols  # [n, m]

    L_chol = np.linalg.cholesky(K_nn + 1e-8 * np.eye(m))
    alpha = np.linalg.solve(L_chol.T, np.linalg.solve(L_chol, y_obs))
    mu = K_star_n @ alpha

    # Variance: diag(K) - diag(K_*n K_nn^{-1} K_n*)
    diag_K = np.array([scale * lu.solve(np.eye(n)[:, i])[i] for i in range(n)])
    v = np.linalg.solve(L_chol, K_star_n.T)
    var = diag_K + noise - np.sum(v ** 2, axis=0)
    var = np.maximum(var, 1e-10)

    nll = 0.5 * y_obs @ alpha + np.sum(np.log(np.diag(L_chol))) + 0.5 * m * np.log(2 * np.pi)
    return mu, var, nll


# -------------------------------------------------------------------
# GP inference (shared across all methods)
# -------------------------------------------------------------------

def gp_posterior(K, y_obs, obs_idx, noise_var):
    """Exact GP posterior. Returns mean, variance, NLL at all nodes.

    Cholesky uses escalating jitter 1e-8 -> 1e-2 to survive ill-conditioned
    kernels (e.g. nu=3/2 Q^{-2} at pathological MLE points).
    """
    n = K.shape[0]
    m = len(obs_idx)
    K_nn = K[np.ix_(obs_idx, obs_idx)] + noise_var * np.eye(m)
    K_star_n = K[:, obs_idx]

    L = None
    for jitter in [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        try:
            L = np.linalg.cholesky(K_nn + jitter * np.eye(m))
            break
        except np.linalg.LinAlgError:
            continue
    if L is None:
        raise np.linalg.LinAlgError(
            'gp_posterior: Cholesky failed even with 1e-2 jitter; K_nn is too ill-conditioned.')

    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))
    mu = K_star_n @ alpha
    v = np.linalg.solve(L, K_star_n.T)
    var = np.diag(K) + noise_var - np.sum(v ** 2, axis=0)
    var = np.maximum(var, 1e-10)

    nll = 0.5 * y_obs @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * m * np.log(2 * np.pi)
    return mu, var, nll


def evaluate_gp(y_true, mu, var, test_idx):
    """RMSE, 95% coverage, mean interval width."""
    y_t = y_true[test_idx]
    mu_t = mu[test_idx]
    std_t = np.sqrt(var[test_idx])

    rmse = np.sqrt(np.mean((y_t - mu_t) ** 2))
    z = 1.96
    coverage = np.mean((y_t >= mu_t - z * std_t) & (y_t <= mu_t + z * std_t))
    width = np.mean(2 * z * std_t)
    return rmse, coverage, width


# -------------------------------------------------------------------
# 1. Diffusion kernel (heat kernel)
# -------------------------------------------------------------------

def diffusion_kernel_matrix(L, tau):
    """K = exp(-tau * L). Uses matrix exponential."""
    return expm(-tau * L)


def learn_diffusion_kernel(L, n, y_obs, obs_idx, n_iter=200, lr=0.05):
    """Learn tau, scale, noise for diffusion kernel via marginal likelihood."""
    L_t = torch.tensor(L, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64)
    obs_t = torch.tensor(obs_idx, dtype=torch.long)
    m = len(obs_idx)

    log_tau = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    log_noise = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_tau, log_scale, log_noise], lr=lr)

    # Eigendecompose L once (heat kernel = V diag(exp(-tau*lambda)) V^T)
    evals, evecs = torch.linalg.eigh(L_t)
    evals = evals.clamp(min=0)

    for it in range(n_iter):
        opt.zero_grad()
        tau = torch.exp(log_tau)
        scale = torch.exp(log_scale)
        noise = torch.exp(log_noise)

        # K = scale * V diag(exp(-tau*lambda)) V^T
        spectral = torch.exp(-tau * evals)
        K = scale * (evecs * spectral.unsqueeze(0)) @ evecs.T

        K_nn = K[obs_t][:, obs_t] + noise * torch.eye(m, dtype=torch.float64)
        L_chol = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(m, dtype=torch.float64))
        alpha = torch.linalg.solve_triangular(
            L_chol.T,
            torch.linalg.solve_triangular(L_chol, y_t.unsqueeze(-1), upper=False),
            upper=True
        ).squeeze(-1)
        nll = 0.5 * y_t @ alpha + torch.sum(torch.log(torch.diag(L_chol)))
        nll.backward()
        opt.step()

    with torch.no_grad():
        tau = torch.exp(log_tau)
        scale = torch.exp(log_scale)
        noise = torch.exp(log_noise)
        spectral = torch.exp(-tau * evals)
        K = scale * (evecs * spectral.unsqueeze(0)) @ evecs.T
        K = ((K + K.T) / 2).numpy()

    return K, noise.item(), {'tau': tau.item(), 'scale': scale.item()}


# -------------------------------------------------------------------
# 2. Graph Matern (via GeometricKernels)
# -------------------------------------------------------------------

def learn_gk_matern(A_adj, n, y_obs, obs_idx, nu=0.5, n_iter=200, lr=0.05):
    """Learn Matern GP hyperparameters via marginal likelihood."""
    import geometric_kernels.torch  # noqa: F401  (backend registration)
    from geometric_kernels.spaces import Graph as GKGraph
    from geometric_kernels.kernels import MaternKarhunenLoeveKernel
    # GeometricKernels needs the adjacency as a scipy sparse matrix
    A_sparse = sparse.csr_matrix(A_adj)
    space = GKGraph(A_sparse)
    kernel = MaternKarhunenLoeveKernel(space, int(n))

    log_ls = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    log_noise = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_ls, log_scale, log_noise], lr=lr)

    y_t = torch.tensor(y_obs, dtype=torch.float64)
    obs_t = torch.tensor(obs_idx, dtype=torch.long)
    xs = torch.arange(n).unsqueeze(-1)
    m = len(obs_idx)

    for it in range(n_iter):
        opt.zero_grad()
        ls = torch.exp(log_ls)
        scale = torch.exp(log_scale)
        noise = torch.exp(log_noise)
        params = {'nu': torch.tensor([nu]), 'lengthscale': ls}
        K = scale * kernel.K(params, xs, xs)
        K = (K + K.T) / 2

        K_nn = K[obs_t][:, obs_t] + noise * torch.eye(m, dtype=torch.float64)
        L_chol = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(m, dtype=torch.float64))
        alpha = torch.linalg.solve_triangular(
            L_chol.T,
            torch.linalg.solve_triangular(L_chol, y_t.unsqueeze(-1), upper=False),
            upper=True
        ).squeeze(-1)
        nll = 0.5 * y_t @ alpha + torch.sum(torch.log(torch.diag(L_chol)))
        nll.backward()
        opt.step()

    with torch.no_grad():
        ls = torch.exp(log_ls)
        scale = torch.exp(log_scale)
        noise = torch.exp(log_noise)
        params = {'nu': torch.tensor([nu]), 'lengthscale': ls}
        K = scale * kernel.K(params, xs, xs)
        K = ((K + K.T) / 2).numpy()

    return K, noise.item(), {'lengthscale': ls.item(), 'scale': scale.item()}


def learn_gk_matern_nu(A_adj, n, y_obs, obs_idx, n_iter=300, lr=0.03):
    """Learn Matern GP with nu as a learnable parameter via MLL.

    Initialises lengthscale using the Mercer-coefficient trick: set ls so
    that the ratio of the first to second spectral coefficients equals 1/2.
    This keeps nu in an interpretable range during optimisation.

    Returns: (K, noise, info_dict) where info_dict contains learned nu, ls, scale.
    """
    import geometric_kernels.torch  # noqa: F401  (backend registration)
    from geometric_kernels.spaces import Graph as GKGraph
    from geometric_kernels.kernels import MaternKarhunenLoeveKernel
    A_sparse = sparse.csr_matrix(A_adj)
    space = GKGraph(A_sparse)
    kernel = MaternKarhunenLoeveKernel(space, int(n))

    # Mercer-coefficient initialisation for lengthscale
    eigvals = kernel._eigenvalues_laplacian.flatten()
    if hasattr(eigvals, 'numpy'):
        eigvals_np = eigvals.numpy()
    else:
        eigvals_np = np.array(eigvals)
    lam1 = float(eigvals_np[eigvals_np > 1e-10].min())
    nu_init = 1.0
    r = 2 ** (-1.0 / nu_init)
    ls_init = np.sqrt(2 * nu_init / (lam1 * r / (1 - r)))

    log_nu = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)  # nu_init ~ 1
    log_ls = torch.tensor([np.log(ls_init)], dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    log_noise = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_nu, log_ls, log_scale, log_noise], lr=lr)

    y_t = torch.tensor(y_obs, dtype=torch.float64)
    obs_t = torch.tensor(obs_idx, dtype=torch.long)
    xs = torch.arange(n).unsqueeze(-1)
    m = len(obs_idx)

    best_nll = float('inf')
    best_state = None

    for it in range(n_iter):
        opt.zero_grad()
        nu = torch.exp(log_nu) + 0.01
        ls = torch.exp(log_ls)
        scale = torch.exp(log_scale)
        noise = torch.exp(log_noise)
        params = {'nu': nu, 'lengthscale': ls}
        K = scale * kernel.K(params, xs, xs)
        K = (K + K.T) / 2
        K_nn = K[obs_t][:, obs_t] + noise * torch.eye(m, dtype=torch.float64)
        try:
            L_chol = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        alpha = torch.linalg.solve_triangular(
            L_chol.T,
            torch.linalg.solve_triangular(L_chol, y_t.unsqueeze(-1), upper=False),
            upper=True
        ).squeeze(-1)
        nll = 0.5 * y_t @ alpha + torch.sum(torch.log(torch.diag(L_chol)))
        if not torch.isfinite(nll):
            continue
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_state = (nu.item(), ls.item(), scale.item(), noise.item())
        nll.backward()
        opt.step()

    if best_state is None:
        return None, None, None

    nu_opt, ls_opt, scale_opt, noise_opt = best_state
    with torch.no_grad():
        params = {'nu': torch.tensor([nu_opt]), 'lengthscale': torch.tensor([ls_opt])}
        K = scale_opt * kernel.K(params, xs, xs)
        K = ((K + K.T) / 2).numpy()

    return K, noise_opt, {'nu': nu_opt, 'lengthscale': ls_opt, 'scale': scale_opt}


# -------------------------------------------------------------------
# 3. Proximal kernel (ours) — two-stage
# -------------------------------------------------------------------

def learn_proximal_two_stage(L, n, y_obs, obs_idx, features,
                              stage1_iters=200, stage2_iters=300, lr=0.05,
                              lambda_z=0.05):
    """Two-stage: learn (tau, scale, noise) stationary, then a_i from features."""
    m = len(obs_idx)
    L_t = torch.tensor(L, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64)
    obs_t = torch.tensor(obs_idx, dtype=torch.long)
    feat_t = torch.tensor(features, dtype=torch.float64)
    if feat_t.dim() == 1:
        feat_t = feat_t.unsqueeze(-1)
    d_feat = feat_t.shape[1]

    # Stage 1
    log_tau = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    log_noise = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    opt1 = torch.optim.Adam([log_tau, log_scale, log_noise], lr=lr)

    for it in range(stage1_iters):
        opt1.zero_grad()
        tau = torch.exp(log_tau)
        scale = torch.exp(log_scale)
        noise = torch.exp(log_noise)
        Q = torch.eye(n, dtype=torch.float64) + 2 * tau * L_t
        K = scale * torch.linalg.inv(Q)
        K = (K + K.T) / 2
        K_nn = K[obs_t][:, obs_t] + noise * torch.eye(m, dtype=torch.float64)
        L_chol = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(m, dtype=torch.float64))
        alpha = torch.linalg.solve_triangular(
            L_chol.T,
            torch.linalg.solve_triangular(L_chol, y_t.unsqueeze(-1), upper=False),
            upper=True
        ).squeeze(-1)
        nll = 0.5 * y_t @ alpha + torch.sum(torch.log(torch.diag(L_chol)))
        nll.backward()
        opt1.step()

    tau_f = torch.exp(log_tau).detach().item()
    scale_f = torch.exp(log_scale).detach().item()
    noise_f = torch.exp(log_noise).detach().item()

    # Stage 2
    w = torch.zeros(d_feat, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt2 = torch.optim.Adam([w, b], lr=lr * 0.5)
    tau_t = torch.tensor(tau_f, dtype=torch.float64)
    scale_t = torch.tensor(scale_f, dtype=torch.float64)
    noise_t = torch.tensor(noise_f, dtype=torch.float64)

    best_nll = float('inf')
    best_w = w.detach().clone()
    best_b = b.detach().clone()

    for it in range(stage2_iters):
        opt2.zero_grad()
        a = torch.sigmoid(feat_t @ w + b)
        A_inv = torch.diag(1.0 / (a + 1e-8))
        Q = A_inv + 2 * tau_t * L_t
        K = scale_t * torch.linalg.inv(Q)
        K = (K + K.T) / 2
        K_nn = K[obs_t][:, obs_t] + noise_t * torch.eye(m, dtype=torch.float64)
        L_chol = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(m, dtype=torch.float64))
        alpha = torch.linalg.solve_triangular(
            L_chol.T,
            torch.linalg.solve_triangular(L_chol, y_t.unsqueeze(-1), upper=False),
            upper=True
        ).squeeze(-1)
        nll = 0.5 * y_t @ alpha + torch.sum(torch.log(torch.diag(L_chol)))
        loss = nll + lambda_z * (torch.sigmoid(feat_t @ w + b) @ L_t @ torch.sigmoid(feat_t @ w + b))
        loss.backward()
        opt2.step()
        if nll.item() < best_nll:
            best_nll = nll.item()
            best_w = w.detach().clone()
            best_b = b.detach().clone()

    a_final = torch.sigmoid(feat_t @ best_w + best_b)

    with torch.no_grad():
        A_inv = torch.diag(1.0 / (a_final + 1e-8))
        Q = A_inv + 2 * tau_t * L_t
        K = scale_t * torch.linalg.inv(Q)
        K = ((K + K.T) / 2).numpy()

    return K, noise_f, a_final.numpy(), {'tau': tau_f, 'scale': scale_f}


# -------------------------------------------------------------------
# Run all baselines on a dataset
# -------------------------------------------------------------------

def run_all_baselines(L, A_adj, n, y_all, train_idx, test_idx, features,
                       matern_nus=[0.5, 1.5, 2.5]):
    """Run all baselines and return results dict."""
    y_train = y_all[train_idx]
    results = {}

    # 1. Diffusion kernel
    print("  Diffusion kernel...")
    K_diff, noise_diff, params_diff = learn_diffusion_kernel(L, n, y_train, train_idx)
    mu, var, nll = gp_posterior(K_diff, y_train, train_idx, noise_diff)
    rmse, cov, width = evaluate_gp(y_all, mu, var, test_idx)
    results['Diffusion'] = {'rmse': rmse, 'coverage': cov, 'nll': nll, 'width': width}
    print(f"    RMSE={rmse:.4f}, Cov={cov:.3f}, NLL={nll:.2f}")

    # 2. Graph Matern (multiple nu)
    for nu in matern_nus:
        name = f'Matern-{nu}'
        print(f"  {name}...")
        K_gk, noise_gk, params_gk = learn_gk_matern(A_adj, n, y_train, train_idx, nu=nu)
        mu, var, nll = gp_posterior(K_gk, y_train, train_idx, noise_gk)
        rmse, cov, width = evaluate_gp(y_all, mu, var, test_idx)
        results[name] = {'rmse': rmse, 'coverage': cov, 'nll': nll, 'width': width}
        print(f"    RMSE={rmse:.4f}, Cov={cov:.3f}, NLL={nll:.2f}")

    # 3. Proximal (ours)
    print("  Proximal (ours)...")
    K_prox, noise_prox, a_learned, params_prox = learn_proximal_two_stage(
        L, n, y_train, train_idx, features
    )
    mu, var, nll = gp_posterior(K_prox, y_train, train_idx, noise_prox)
    rmse, cov, width = evaluate_gp(y_all, mu, var, test_idx)
    results['Proximal (ours)'] = {'rmse': rmse, 'coverage': cov, 'nll': nll,
                                   'width': width, 'a_learned': a_learned}
    print(f"    RMSE={rmse:.4f}, Cov={cov:.3f}, NLL={nll:.2f}")

    return results

"""Experiment: Non-stationary GP regression on synthetic graph signals.

Generates a graph with spatially varying smoothness, observes a noisy
subset of nodes, and compares:
  1. Stationary Matern GP (global kappa) — GeometricKernels baseline
  2. Non-stationary proximal GP (per-node a_i) — our method
  3. Oracle non-stationary GP (true a_i known)

Metrics: RMSE, NLL, calibration (coverage of 95% credible intervals).

Usage:
    python experiments/graph_gp/synthetic_nonstationary.py
    python experiments/graph_gp/synthetic_nonstationary.py --graph_type grid --n 400
"""

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# GeometricKernels baseline
try:
    import geometric_kernels.torch
    from geometric_kernels.spaces import Graph as GKGraph
    from geometric_kernels.kernels import MaternKarhunenLoeveKernel
    HAS_GK = True
except ImportError:
    HAS_GK = False


# -------------------------------------------------------------------
# Graph construction
# -------------------------------------------------------------------

def make_grid_graph(side=20):
    """2D grid graph with n = side^2 nodes."""
    n = side * side
    edges = []
    for i in range(side):
        for j in range(side):
            node = i * side + j
            if j < side - 1:
                edges.append((node, node + 1))
            if i < side - 1:
                edges.append((node, node + side))
    edges = np.array(edges)
    # Symmetric
    edge_index = np.concatenate([edges, edges[:, ::-1]], axis=0).T
    pos = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
    return n, edge_index, pos


def make_barabasi_albert_graph(n=200, m=3):
    """Barabasi-Albert preferential attachment graph."""
    # Simple BA construction
    adj = np.zeros((n, n))
    degrees = np.zeros(n)
    # Start with a complete graph on m+1 nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1

    for new_node in range(m + 1, n):
        # Attach to m existing nodes with probability proportional to degree
        probs = degrees[:new_node] / degrees[:new_node].sum()
        targets = np.random.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1
            degrees[new_node] += 1
            degrees[t] += 1

    rows, cols = np.where(adj > 0)
    edge_index = np.stack([rows, cols])
    # Spring layout (simplified: use eigenvectors of Laplacian)
    L = np.diag(degrees) - adj
    evals, evecs = np.linalg.eigh(L)
    pos = evecs[:, 1:3]  # Fiedler vector + next
    return n, edge_index, pos


def build_laplacian(n, edge_index, normalize=True):
    """Build normalized or unnormalized Laplacian."""
    A = np.zeros((n, n))
    for k in range(edge_index.shape[1]):
        i, j = edge_index[0, k], edge_index[1, k]
        A[i, j] = 1.0
    D = A.sum(axis=1)
    L = np.diag(D) - A
    if normalize:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D, 1e-10)))
        L = D_inv_sqrt @ L @ D_inv_sqrt
    return L, D


# -------------------------------------------------------------------
# Non-stationary signal generation
# -------------------------------------------------------------------

def generate_nonstationary_signal(n, L, pos, seed=42):
    """Generate a signal with spatially varying smoothness.

    Left half of the graph: smooth (large effective tau).
    Right half: rough (small effective tau).

    Returns:
        f_true: true signal [n]
        a_true: true per-node smoothness weights [n]
    """
    rng = np.random.RandomState(seed)

    # Per-node smoothness: smooth on the left, rough on the right
    x_coord = (pos[:, 0] - pos[:, 0].min()) / (pos[:, 0].max() - pos[:, 0].min())
    a_true = 0.1 + 0.9 * (1 - x_coord)  # Left: a~1 (smooth), Right: a~0.1 (rough)

    # Generate signal: sample from the non-stationary GP prior
    # K = (I + 2*tau*A*L)^{-1}
    tau = 5.0
    A = np.diag(a_true)
    K = np.linalg.inv(np.eye(n) + 2 * tau * A @ L)

    # Make K symmetric (numerical)
    K = (K + K.T) / 2

    # Sample from GP prior
    L_chol = np.linalg.cholesky(K + 1e-6 * np.eye(n))
    f_true = L_chol @ rng.randn(n)

    return f_true, a_true, tau


# -------------------------------------------------------------------
# GP inference
# -------------------------------------------------------------------

def gp_inference(K_prior, y_obs, obs_idx, noise_var=0.01):
    """Standard GP posterior inference.

    Args:
        K_prior: [n, n] prior covariance matrix.
        y_obs: [m] observed values.
        obs_idx: [m] indices of observed nodes.
        noise_var: observation noise variance.

    Returns:
        mu_post: [n] posterior mean.
        var_post: [n] posterior marginal variances.
        nll: negative log-likelihood of observations.
    """
    n = K_prior.shape[0]
    m = len(obs_idx)

    K_nn = K_prior[np.ix_(obs_idx, obs_idx)] + noise_var * np.eye(m)
    K_star_n = K_prior[:, obs_idx]  # [n, m]

    # Solve K_nn^{-1} y
    L = np.linalg.cholesky(K_nn + 1e-8 * np.eye(m))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_obs))

    # Posterior mean
    mu_post = K_star_n @ alpha

    # Posterior variance
    v = np.linalg.solve(L, K_star_n.T)  # [m, n]
    var_post = np.diag(K_prior) - np.sum(v ** 2, axis=0)
    var_post = np.maximum(var_post, 1e-10)

    # NLL
    nll = 0.5 * y_obs @ alpha + np.sum(np.log(np.diag(L))) + 0.5 * m * np.log(2 * np.pi)

    return mu_post, var_post, nll


def geometric_kernels_matern(A_adj, n, nu, lengthscale):
    """Compute Matern kernel matrix using GeometricKernels.

    Args:
        A_adj: [n, n] adjacency matrix (binary, symmetric).
        n: number of nodes.
        nu: smoothness parameter (0.5, 1.5, 2.5, or inf).
        lengthscale: kernel lengthscale.

    Returns:
        K: [n, n] kernel matrix (numpy).
    """
    if not HAS_GK:
        raise ImportError("geometric_kernels not installed")

    space = GKGraph(A_adj)
    kernel = MaternKarhunenLoeveKernel(space, n)
    params = {
        'nu': torch.tensor([nu]),
        'lengthscale': torch.tensor([lengthscale]),
    }
    xs = torch.arange(n).unsqueeze(-1)
    K = kernel.K(params, xs, xs).detach().numpy()
    return (K + K.T) / 2


def learn_gk_matern(A_adj, n, y_obs, obs_idx, noise_var, nu=0.5,
                     n_iter=200, lr=0.05):
    """Learn lengthscale for GeometricKernels Matern GP via marginal likelihood.

    Args:
        A_adj: adjacency matrix.
        n: number of nodes.
        y_obs: observed values.
        obs_idx: indices of observed nodes.
        noise_var: observation noise variance.
        nu: smoothness (fixed).

    Returns:
        K: learned kernel matrix [n, n].
        lengthscale: learned lengthscale.
    """
    if not HAS_GK:
        raise ImportError("geometric_kernels not installed")

    space = GKGraph(A_adj)
    kernel = MaternKarhunenLoeveKernel(space, n)

    log_ls = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor([0.0], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_ls, log_scale], lr=lr)

    y_t = torch.tensor(y_obs, dtype=torch.float64)
    obs_t = torch.tensor(obs_idx, dtype=torch.long)
    xs = torch.arange(n).unsqueeze(-1)
    m = len(obs_idx)

    for it in range(n_iter):
        opt.zero_grad()
        ls = torch.exp(log_ls)
        scale = torch.exp(log_scale)
        params = {'nu': torch.tensor([nu]), 'lengthscale': ls}
        K = scale * kernel.K(params, xs, xs)
        K = (K + K.T) / 2

        K_nn = K[obs_t][:, obs_t] + noise_var * torch.eye(m, dtype=torch.float64)
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
        params = {'nu': torch.tensor([nu]), 'lengthscale': ls}
        K = scale * kernel.K(params, xs, xs)
        K = (K + K.T) / 2

    return K.numpy(), ls.item(), scale.item()


def proximal_kernel(L, a, tau):
    """Compute K = (I + 2*tau*A*L)^{-1}."""
    n = L.shape[0]
    A = np.diag(a)
    M = np.eye(n) + 2 * tau * A @ L
    K = np.linalg.inv(M)
    # Symmetrise in A^{-1} inner product
    A_sqrt = np.diag(np.sqrt(a))
    A_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(a, 1e-10)))
    K_sym = A_inv_sqrt @ np.linalg.inv(
        np.eye(n) + 2 * tau * A_sqrt @ L @ A_sqrt
    ) @ A_inv_sqrt
    return (K_sym + K_sym.T) / 2


def stationary_kernel(L, tau):
    """Compute stationary Matern-1/2: K = (I + 2*tau*L)^{-1}."""
    n = L.shape[0]
    K = np.linalg.inv(np.eye(n) + 2 * tau * L)
    return (K + K.T) / 2


# -------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------

def compute_metrics(f_true, mu_post, var_post, test_idx):
    """Compute RMSE, calibration at test nodes."""
    f_test = f_true[test_idx]
    mu_test = mu_post[test_idx]
    var_test = var_post[test_idx]

    rmse = np.sqrt(np.mean((f_test - mu_test) ** 2))

    # 95% coverage
    z = 1.96
    lower = mu_test - z * np.sqrt(var_test)
    upper = mu_test + z * np.sqrt(var_test)
    coverage = np.mean((f_test >= lower) & (f_test <= upper))

    # Average interval width
    width = np.mean(2 * z * np.sqrt(var_test))

    return rmse, coverage, width


# -------------------------------------------------------------------
# Main experiment
# -------------------------------------------------------------------

def run_experiment(graph_type='grid', n_side=20, obs_frac=0.3,
                   noise_var=0.01, seed=42, output_dir=None):
    """Run the full comparison."""
    rng = np.random.RandomState(seed)

    # Build graph
    if graph_type == 'grid':
        n, edge_index, pos = make_grid_graph(side=n_side)
    else:
        n, edge_index, pos = make_barabasi_albert_graph(n=n_side * n_side)

    L, D = build_laplacian(n, edge_index, normalize=True)

    # Generate non-stationary signal
    f_true, a_true, tau_true = generate_nonstationary_signal(n, L, pos, seed=seed)

    # Observe a random subset
    n_obs = int(obs_frac * n)
    obs_idx = rng.choice(n, size=n_obs, replace=False)
    test_idx = np.setdiff1d(np.arange(n), obs_idx)
    y_obs = f_true[obs_idx] + np.sqrt(noise_var) * rng.randn(n_obs)

    print(f"Graph: {graph_type}, n={n}, observed={n_obs}, test={len(test_idx)}")
    print(f"True tau={tau_true}, a_i range=[{a_true.min():.2f}, {a_true.max():.2f}]")
    print()

    # --- Method 1: Stationary GP ---
    # Use true tau but uniform a_i = mean(a_true) to be fair
    K_stat = stationary_kernel(L, tau_true)
    mu_stat, var_stat, nll_stat = gp_inference(K_stat, y_obs, obs_idx, noise_var)
    rmse_stat, cov_stat, width_stat = compute_metrics(f_true, mu_stat, var_stat, test_idx)

    # --- Method 2: Oracle non-stationary GP (true a_i known) ---
    K_oracle = proximal_kernel(L, a_true, tau_true)
    mu_oracle, var_oracle, nll_oracle = gp_inference(K_oracle, y_obs, obs_idx, noise_var)
    rmse_oracle, cov_oracle, width_oracle = compute_metrics(f_true, mu_oracle, var_oracle, test_idx)

    # --- Method 3: Learned non-stationary GP ---
    # Learn a_i by minimising NLL. Fix tau at the stationary MLE first,
    # then learn a_i with tau fixed — avoids the tau*a_i identifiability issue.
    a_learned, tau_learned = learn_hyperparameters(
        L, y_obs, obs_idx, n, noise_var, seed=seed
    )
    K_learned = proximal_kernel(L, a_learned, tau_learned)
    mu_learned, var_learned, nll_learned = gp_inference(
        K_learned, y_obs, obs_idx, noise_var
    )
    rmse_learned, cov_learned, width_learned = compute_metrics(
        f_true, mu_learned, var_learned, test_idx
    )

    # --- GeometricKernels baselines ---
    gk_results = {}
    if HAS_GK:
        # Build adjacency for GeometricKernels
        A_adj = np.zeros((n, n))
        for k in range(edge_index.shape[1]):
            i, j = edge_index[0, k], edge_index[1, k]
            A_adj[i, j] = 1.0

        for nu in [0.5, 1.5, 2.5]:
            name = f'GK Matern-{nu}'
            try:
                K_gk, ls, sc = learn_gk_matern(
                    A_adj, n, y_obs, obs_idx, noise_var, nu=nu
                )
                mu_gk, var_gk, nll_gk = gp_inference(K_gk, y_obs, obs_idx, noise_var)
                rmse_gk, cov_gk, width_gk = compute_metrics(f_true, mu_gk, var_gk, test_idx)
                gk_results[name] = {
                    'rmse': rmse_gk, 'coverage': cov_gk, 'width': width_gk,
                    'nll': nll_gk, 'lengthscale': ls, 'scale': sc
                }
            except Exception as e:
                print(f"  {name}: FAILED — {e}")

    # Print results
    print(f"{'Method':<25} {'RMSE':>8} {'Coverage':>10} {'Width':>8} {'NLL':>10}")
    print("-" * 65)
    print(f"{'Stationary GP (ours)':<25} {rmse_stat:>8.4f} {cov_stat:>10.3f} {width_stat:>8.4f} {nll_stat:>10.2f}")
    for name, r in gk_results.items():
        print(f"{name:<25} {r['rmse']:>8.4f} {r['coverage']:>10.3f} {r['width']:>8.4f} {r['nll']:>10.2f}")
    print(f"{'Learned non-stat (ours)':<25} {rmse_learned:>8.4f} {cov_learned:>10.3f} {width_learned:>8.4f} {nll_learned:>10.2f}")
    print(f"{'Oracle non-stat (ours)':<25} {rmse_oracle:>8.4f} {cov_oracle:>10.3f} {width_oracle:>8.4f} {nll_oracle:>10.2f}")

    # Correlation between true and learned a_i
    corr = np.corrcoef(a_true, a_learned)[0, 1]
    print(f"\nCorrelation(a_true, a_learned) = {corr:.4f}")
    print(f"Learned tau = {tau_learned:.4f} (true = {tau_true:.4f})")

    # --- Plots ---
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if graph_type == 'grid':
            side = n_side
            plot_grid_results(
                side, f_true, a_true, a_learned,
                mu_stat, mu_oracle, mu_learned,
                var_stat, var_oracle, var_learned,
                obs_idx, output_dir
            )

    return {
        'stationary': {'rmse': rmse_stat, 'coverage': cov_stat, 'width': width_stat, 'nll': nll_stat},
        'learned': {'rmse': rmse_learned, 'coverage': cov_learned, 'width': width_learned, 'nll': nll_learned},
        'oracle': {'rmse': rmse_oracle, 'coverage': cov_oracle, 'width': width_oracle, 'nll': nll_oracle},
        'a_corr': corr,
    }


def learn_hyperparameters(L, y_obs, obs_idx, n, noise_var, seed=42,
                          n_iter_tau=100, n_iter_a=300, lr=0.05):
    """Learn per-node a_i and tau by minimising marginal NLL.

    Two-stage optimisation to avoid the tau*a_i identifiability problem:
      Stage 1: Learn tau with uniform a_i = 1 (stationary MLE).
      Stage 2: Fix tau, learn per-node a_i = sigmoid(z_i).

    This ensures tau captures the global lengthscale and a_i captures
    the spatially varying deviation from it.
    """
    torch.manual_seed(seed)

    L_t = torch.tensor(L, dtype=torch.float64)
    y_t = torch.tensor(y_obs, dtype=torch.float64)
    obs_t = torch.tensor(obs_idx, dtype=torch.long)
    m = len(obs_idx)

    def compute_nll(K, y, idx, nv):
        mm = len(idx)
        K_nn = K[idx][:, idx] + nv * torch.eye(mm, dtype=torch.float64)
        L_chol = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(mm, dtype=torch.float64))
        alpha = torch.linalg.solve_triangular(
            L_chol.T,
            torch.linalg.solve_triangular(L_chol, y.unsqueeze(-1), upper=False),
            upper=True
        ).squeeze(-1)
        return 0.5 * y @ alpha + torch.sum(torch.log(torch.diag(L_chol))) + 0.5 * mm * np.log(2 * np.pi)

    # Stage 1: learn tau with uniform a_i = 1
    log_tau = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    opt1 = torch.optim.Adam([log_tau], lr=lr)

    for it in range(n_iter_tau):
        opt1.zero_grad()
        tau = torch.exp(log_tau)
        K = torch.linalg.inv(torch.eye(n, dtype=torch.float64) + 2 * tau * L_t)
        K = (K + K.T) / 2
        nll = compute_nll(K, y_t, obs_t, noise_var)
        nll.backward()
        opt1.step()

    tau_fixed = torch.exp(log_tau).detach().item()
    print(f"  Stage 1: tau_MLE = {tau_fixed:.4f}")

    # Stage 2: fix tau, learn per-node a_i
    z = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    opt2 = torch.optim.Adam([z], lr=lr * 0.5)

    best_nll = float('inf')
    best_z = z.detach().clone()
    tau_t = torch.tensor(tau_fixed, dtype=torch.float64)

    for it in range(n_iter_a):
        opt2.zero_grad()

        a = torch.sigmoid(z)
        A_sqrt = torch.diag(torch.sqrt(a))
        A_inv_sqrt = torch.diag(1.0 / torch.sqrt(a + 1e-10))
        M = torch.eye(n, dtype=torch.float64) + 2 * tau_t * A_sqrt @ L_t @ A_sqrt
        K = A_inv_sqrt @ torch.linalg.inv(M) @ A_inv_sqrt
        K = (K + K.T) / 2

        nll = compute_nll(K, y_t, obs_t, noise_var)

        # Smoothness prior: encourage a_i to vary smoothly on the graph
        z_smooth = 0.05 * (z @ L_t @ z)
        loss = nll + z_smooth

        loss.backward()
        opt2.step()

        if nll.item() < best_nll:
            best_nll = nll.item()
            best_z = z.detach().clone()

        if (it + 1) % 100 == 0:
            print(f"  Stage 2 iter {it+1}: NLL={nll.item():.2f}, "
                  f"a=[{a.min().item():.3f}, {a.max().item():.3f}]")

    a_final = torch.sigmoid(best_z).numpy()
    return a_final, tau_fixed


# -------------------------------------------------------------------
# Plotting (grid graphs only)
# -------------------------------------------------------------------

def plot_grid_results(side, f_true, a_true, a_learned,
                      mu_stat, mu_oracle, mu_learned,
                      var_stat, var_oracle, var_learned,
                      obs_idx, output_dir):
    """Plot 2D grid results."""
    n = side * side

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    def imshow(ax, vals, title, cmap='viridis'):
        im = ax.imshow(vals.reshape(side, side), cmap=cmap, interpolation='nearest')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Row 1: signals
    imshow(axes[0, 0], f_true, 'True signal')
    imshow(axes[0, 1], mu_stat, 'Stationary GP mean')
    imshow(axes[0, 2], mu_learned, 'Learned non-stat GP mean')
    imshow(axes[0, 3], mu_oracle, 'Oracle non-stat GP mean')

    # Row 2: variances and a_i
    imshow(axes[1, 0], a_true, 'True a_i', cmap='RdYlBu')
    imshow(axes[1, 1], np.sqrt(var_stat), 'Stationary GP std')
    imshow(axes[1, 2], a_learned, 'Learned a_i', cmap='RdYlBu')
    imshow(axes[1, 3], np.sqrt(var_learned), 'Learned non-stat GP std')

    plt.tight_layout()
    plt.savefig(output_dir / 'grid_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'grid_comparison.png'}")

    # Error maps
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (mu, name) in zip(axes, [
        (mu_stat, 'Stationary'), (mu_learned, 'Learned non-stat'), (mu_oracle, 'Oracle non-stat')
    ]):
        err = np.abs(f_true - mu)
        imshow(ax, err, f'{name} |error|', cmap='Reds')
    plt.tight_layout()
    plt.savefig(output_dir / 'grid_errors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'grid_errors.png'}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_type', type=str, default='grid',
                        choices=['grid', 'ba'])
    parser.add_argument('--n', type=int, default=20,
                        help='Grid side length or sqrt(n) for BA')
    parser.add_argument('--obs_frac', type=float, default=0.3)
    parser.add_argument('--noise_var', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path('graph_gp/results')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Non-stationary GP Regression on Synthetic Graph Signals")
    print("=" * 65)

    results = run_experiment(
        graph_type=args.graph_type,
        n_side=args.n,
        obs_frac=args.obs_frac,
        noise_var=args.noise_var,
        seed=args.seed,
        output_dir=output_dir,
    )

    # Multiple seeds
    print("\n" + "=" * 65)
    print("Multi-seed evaluation (seeds 42-46)")
    print("=" * 65)

    all_results = {k: [] for k in ['stationary', 'learned', 'oracle']}
    all_corrs = []

    for s in range(42, 47):
        r = run_experiment(
            graph_type=args.graph_type,
            n_side=args.n,
            obs_frac=args.obs_frac,
            noise_var=args.noise_var,
            seed=s,
            output_dir=None,
        )
        for k in all_results:
            all_results[k].append(r[k])
        all_corrs.append(r['a_corr'])

    print(f"\n{'Method':<25} {'RMSE':>12} {'Coverage':>12} {'NLL':>12}")
    print("-" * 65)
    for method in ['stationary', 'learned', 'oracle']:
        rmses = [r['rmse'] for r in all_results[method]]
        covs = [r['coverage'] for r in all_results[method]]
        nlls = [r['nll'] for r in all_results[method]]
        print(f"{method:<25} {np.mean(rmses):.4f}+/-{np.std(rmses):.4f} "
              f"{np.mean(covs):.3f}+/-{np.std(covs):.3f} "
              f"{np.mean(nlls):.2f}+/-{np.std(nlls):.2f}")
    print(f"\na_i recovery correlation: {np.mean(all_corrs):.4f}+/-{np.std(all_corrs):.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()

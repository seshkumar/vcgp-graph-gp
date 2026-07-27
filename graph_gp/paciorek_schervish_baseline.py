"""Paciorek-Schervish non-stationary kernel baseline.

Spatially varying lengthscale from the same auxiliary d_i data.
Uses Euclidean coordinates (lat/lon) as input space.
High d_i → short lengthscale (rough, local).
"""

import os; os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, pickle, gzip, numpy as np, pandas as pd, h5py, torch
from scipy.spatial.distance import cdist
from scipy.stats import wilcoxon
sys.path.insert(0, 'graph_gp')
from baselines import gp_posterior, evaluate_gp, learn_gk_matern

SEEDS = list(range(42, 62))

def nll_test(mu, var, y, idx):
    yt = y[idx]; mt = mu[idx]; vt = var[idx]
    return np.sum(0.5 * np.log(2 * np.pi * vt) + 0.5 * (yt - mt)**2 / vt)


def paciorek_schervish_kernel(coords, ell_i, scale=1.0):
    """Paciorek-Schervish non-stationary kernel.

    K(i,j) = scale * sqrt(ell_i * ell_j) / sqrt((ell_i^2 + ell_j^2)/2)
             * exp(-r^2 / (ell_i^2 + ell_j^2))

    where r = ||x_i - x_j|| in coordinate space (km).
    """
    n = len(ell_i)
    # Pairwise Euclidean distances
    r = cdist(coords, coords)
    # PS kernel
    ell_prod = np.outer(ell_i, ell_i)  # ell_i * ell_j
    ell_sum_sq = np.add.outer(ell_i**2, ell_i**2)  # ell_i^2 + ell_j^2
    # Normalisation: sqrt(ell_i * ell_j) / sqrt(mean(ell_i^2, ell_j^2))
    norm = np.sqrt(ell_prod) / np.sqrt(ell_sum_sq / 2 + 1e-10)
    # Exponential
    K = scale * norm * np.exp(-r**2 / (ell_sum_sq + 1e-10))
    K = (K + K.T) / 2
    # Ensure PD: project out negative eigenvalues and add jitter
    eigvals, eigvecs = np.linalg.eigh(K)
    eigvals = np.maximum(eigvals, 1e-4)
    K = eigvecs @ np.diag(eigvals) @ eigvecs.T
    K = (K + K.T) / 2 + 1e-6 * np.eye(len(ell_i))
    return K


def learn_ps_kernel(coords, d_i, y_train, train_idx, n, n_iter=200, lr=0.05):
    """Learn PS kernel hyperparameters: base lengthscale, scale, noise.

    Lengthscale per node: ell_i = ell_base / sqrt(d_i).
    High d_i → short lengthscale (rough).
    """
    coords_t = torch.tensor(coords, dtype=torch.float64)
    d_t = torch.tensor(d_i, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    obs_t = torch.tensor(train_idx, dtype=torch.long)
    m = len(train_idx)

    log_ell = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    log_noise = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_ell, log_scale, log_noise], lr=lr)

    # Precompute distances
    r = torch.cdist(coords_t, coords_t)

    best_nll = float('inf')
    best_params = None

    for it in range(n_iter):
        opt.zero_grad()
        # Clamp log_ell so exp() cannot overflow even when gradients drift
        # on flat landscapes (continental-scale PM2.5 was the failure case).
        log_ell_c = torch.clamp(log_ell, min=-5.0, max=10.0)
        log_scale_c = torch.clamp(log_scale, min=-10.0, max=10.0)
        log_noise_c = torch.clamp(log_noise, min=-10.0, max=2.0)
        ell_base = torch.exp(log_ell_c)
        scale = torch.exp(log_scale_c)
        noise = torch.exp(log_noise_c)

        ell_i = ell_base / torch.sqrt(d_t + 1e-8)
        ell_prod = ell_i.unsqueeze(1) * ell_i.unsqueeze(0)
        ell_sum_sq = ell_i.unsqueeze(1)**2 + ell_i.unsqueeze(0)**2
        norm = torch.sqrt(ell_prod) / torch.sqrt(ell_sum_sq / 2 + 1e-10)
        K = scale * norm * torch.exp(-r**2 / (ell_sum_sq + 1e-10))
        K = (K + K.T) / 2

        # Ensure PD: eigenvalue clipping (restored; was essential for
        # small-n datasets to converge in practice).
        try:
            eigvals, eigvecs = torch.linalg.eigh(K)
        except Exception:
            continue
        eigvals = torch.clamp(eigvals, min=1e-4)
        K = eigvecs @ torch.diag(eigvals) @ eigvecs.T

        K_nn = K[obs_t][:, obs_t] + noise * torch.eye(m, dtype=torch.float64)
        try:
            L = torch.linalg.cholesky(K_nn + 1e-6 * torch.eye(m, dtype=torch.float64))
        except Exception:
            continue
        alpha = torch.linalg.solve_triangular(
            L.T, torch.linalg.solve_triangular(L, y_t.unsqueeze(-1), upper=False),
            upper=True).squeeze(-1)
        nll = 0.5 * y_t @ alpha + torch.sum(torch.log(torch.diag(L)))
        if not torch.isfinite(nll):
            continue

        nll.backward()
        opt.step()
        # Save best params AFTER the step (matches the legacy behaviour on
        # small-n datasets, which relied on this to produce a good fit),
        # but reject non-finite values so runaway drift (the PM2.5 failure
        # mode) does not poison best_params.
        if nll.item() < best_nll:
            le = log_ell.item()
            ls = log_scale.item()
            ln = log_noise.item()
            if all(map(np.isfinite, (le, ls, ln))):
                best_nll = nll.item()
                best_params = (le, ls, ln)

    if best_params is None:
        best_params = (2.0, 0.0, -4.0)

    ell_base = np.exp(best_params[0])
    scale = np.exp(best_params[1])
    noise = np.exp(best_params[2])
    ell_i = ell_base / np.sqrt(d_i + 1e-8)
    K = paciorek_schervish_kernel(coords, ell_i, scale)
    return K, noise, ell_i


def run_dataset(name, coords_km, A, n, y_norm, d_i, correct_constraint):
    log(f'\n{"="*60}')
    log(f'{name} (n={n})')
    log(f'{"="*60}')

    nll_ps = []; nll_stat = []; nll_con = []; nll_m15 = []
    rmse_ps = []; rmse_con = []

    from new_kernel import constrained_proximal_kernel
    from two_param_kernel import two_param_kernel
    import torch

    def fit_s1(L, n, yt, ti):
        Lt = torch.tensor(L, dtype=torch.float64); yy = torch.tensor(yt, dtype=torch.float64)
        ot = torch.tensor(ti, dtype=torch.long); m = len(ti)
        lt = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        ls = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        ln = torch.tensor(-4., dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([lt, ls, ln], lr=0.05)
        for i in range(200):
            opt.zero_grad(); t = torch.exp(lt); s = torch.exp(ls); no = torch.exp(ln)
            Q = torch.eye(n, dtype=torch.float64) + 2 * t * Lt
            K = s * torch.linalg.inv(Q); K = (K + K.T) / 2
            Kn = K[ot][:, ot] + no * torch.eye(m, dtype=torch.float64)
            Lc = torch.linalg.cholesky(Kn + 1e-6 * torch.eye(m, dtype=torch.float64))
            a = torch.linalg.solve_triangular(Lc.T, torch.linalg.solve_triangular(
                Lc, yy.unsqueeze(-1), upper=False), upper=True).squeeze(-1)
            nll = 0.5 * yy @ a + torch.sum(torch.log(torch.diag(Lc)))
            nll.backward(); opt.step()
        return torch.exp(lt).item(), torch.exp(ls).item(), torch.exp(ln).item()

    def constrained_Dinv(L, d, tau, scale=1.0):
        n2 = L.shape[0]; d_inv = 1. / np.maximum(d, 1e-8)
        D_inv = np.diag(d_inv); D_inv_sqrt = np.diag(np.sqrt(d_inv))
        Q = D_inv + 2 * tau * D_inv_sqrt @ L @ D_inv_sqrt
        K = scale * np.linalg.inv(Q + 1e-8 * np.eye(n2)); return (K + K.T) / 2

    def constrained_AD_log(L, d, tau, scale=1.0):
        d_log = np.log(d + 0.1); d_log_s = (d_log - d_log.mean()) / (d_log.std() + 1e-8)
        d_scaled = np.exp(d_log_s * 0.3)
        n2 = L.shape[0]; D_inv = np.diag(1. / np.maximum(d_scaled, 1e-8))
        D_sqrt = np.diag(np.sqrt(np.maximum(d_scaled, 1e-8)))
        Q = D_inv + 2 * tau * D_sqrt @ L @ D_sqrt
        K = scale * np.linalg.inv(Q + 1e-8 * np.eye(n2)); return (K + K.T) / 2

    # Build Laplacian
    deg = A.sum(1); Dis = np.diag(1. / np.sqrt(np.maximum(deg, 1e-10)))
    L = np.eye(n) - Dis @ A @ Dis

    for seed in SEEDS:
        rng = np.random.RandomState(seed); perm = rng.permutation(n)
        ti = perm[:int(0.7 * n)]; xi = perm[int(0.7 * n):]; yt = y_norm[ti]

        # Paciorek-Schervish
        K_ps, noise_ps, ell_learned = learn_ps_kernel(coords_km, d_i, yt, ti, n)
        mu_ps, var_ps, _ = gp_posterior(K_ps, yt, ti, noise_ps)
        nll_ps.append(nll_test(mu_ps, var_ps, y_norm, xi))
        rmse_ps.append(np.sqrt(np.mean((y_norm[xi] - mu_ps[xi])**2)))

        # Constrained kernel
        tau, scale, noise = fit_s1(L, n, yt, ti)
        if correct_constraint == 'Dinv':
            K_c = constrained_Dinv(L, d_i, tau, scale)
        else:
            K_c = constrained_AD_log(L, d_i, tau, scale)
        mu_c, var_c, _ = gp_posterior(K_c, yt, ti, noise)
        nll_con.append(nll_test(mu_c, var_c, y_norm, xi))
        rmse_con.append(np.sqrt(np.mean((y_norm[xi] - mu_c[xi])**2)))

        # Stationary
        K_s = two_param_kernel(L, np.ones(n), np.ones(n), tau, scale)
        mu_s, var_s, _ = gp_posterior(K_s, yt, ti, noise)
        nll_stat.append(nll_test(mu_s, var_s, y_norm, xi))

        # Matern-1.5
        K_m, nm2, _ = learn_gk_matern(A, int(n), yt, ti, nu=1.5)
        mu_m, var_m, _ = gp_posterior(K_m, yt, ti, nm2)
        nll_m15.append(nll_test(mu_m, var_m, y_norm, xi))

        log(f'  seed {seed}: PS={nll_ps[-1]:.1f}, con={nll_con[-1]:.1f}, stat={nll_stat[-1]:.1f}, m15={nll_m15[-1]:.1f}')

    nps = np.array(nll_ps); nc = np.array(nll_con); ns = np.array(nll_stat); nm = np.array(nll_m15)
    rps = np.array(rmse_ps); rc = np.array(rmse_con)

    log(f'\n{"Method":<25} {"NLL":>12} {"RMSE":>8}')
    log('-' * 47)
    log(f'{"Stationary":<25} {ns.mean():.2f}+/-{ns.std():.2f}')
    log(f'{"Matern-1.5":<25} {nm.mean():.2f}+/-{nm.std():.2f}')
    log(f'{"Paciorek-Schervish":<25} {nps.mean():.2f}+/-{nps.std():.2f} {rps.mean():.3f}')
    log(f'{"Constrained (ours)":<25} {nc.mean():.2f}+/-{nc.std():.2f} {rc.mean():.3f}')

    # Wilcoxon: constrained vs PS
    d_cp = nps - nc; _, p_cp = wilcoxon(d_cp, alternative='greater')
    d_sp = ns - nps; _, p_sp = wilcoxon(d_sp, alternative='greater')
    log(f'\nConstrained vs PS: diff={d_cp.mean():+.2f}, p={p_cp:.4f} {"*" if p_cp < 0.05 else ""}')
    log(f'PS vs Stationary:  diff={d_sp.mean():+.2f}, p={p_sp:.4f} {"*" if p_sp < 0.05 else ""}')
    log(f'Constrained wins {(d_cp > 0).sum()}/20 seeds vs PS')


if __name__ == '__main__':
    OUTFILE = 'graph_gp/paciorek_schervish_results.txt'
    f = open(OUTFILE, 'w')
    def log(msg): f.write(msg + '\n'); f.flush()

    # ==================== METR-LA ====================
    log('Paciorek-Schervish Baseline')
    log('=' * 60)

    DATA_DIR = 'graph_gp/data_cache/metr_la'
    with open(f'{DATA_DIR}/adj_mx.pkl', 'rb') as fp:
        sensor_ids, _, adj = pickle.load(fp, encoding='latin1')
    with h5py.File(f'{DATA_DIR}/metr-la.h5', 'r') as fp:
        speeds = fp['df']['block0_values'][:]
    import csv
    locs = {}
    with open(f'{DATA_DIR}/graph_sensor_locations.csv') as fh:
        for row in csv.reader(fh):
            if len(row) >= 4:
                try: locs[row[1].strip()] = (float(row[2]), float(row[3]))
                except: pass
    n_m = speeds.shape[1]
    coords_m = np.array([locs.get(str(sid), (34.05, -118.25)) for sid in sensor_ids[:n_m]])
    coords_m_km = coords_m.copy(); coords_m_km[:, 0] *= 111; coords_m_km[:, 1] *= 85
    A_m = (adj > 0).astype(float); A_m = np.maximum(A_m, A_m.T); np.fill_diagonal(A_m, 0)
    y_m = speeds[:288, :].mean(axis=0); y_m = (y_m - y_m.mean()) / y_m.std()
    d_cv = speeds.std(axis=0) / (speeds.mean(axis=0) + 1e-8)
    d_cv = d_cv / d_cv.mean(); d_cv = np.clip(d_cv, 0.1, 10)

    run_dataset('METR-LA', coords_m_km, A_m, n_m, y_m, d_cv, 'Dinv')

    # ==================== LAQN ====================
    from laqn_regression import fetch_laqn_sites, fetch_laqn_data, build_graph
    sites = fetch_laqn_sites(); values = fetch_laqn_data(sites, 'NO2', '2024-06-15')
    sites = [s for s in sites if s['code'] in values]
    ys_l = np.array([values[s['code']] for s in sites])
    type_nums = np.array([{'Rural':0,'Suburban':1,'Urban Background':2,'Industrial':3,'Roadside':4,'Kerbside':5}.get(s['type'],2) for s in sites], dtype=float)
    coords_l = np.array([[s['lat'], s['lon']] for s in sites])
    ys_ln = (ys_l - ys_l.mean()) / ys_l.std()
    A_l, _ = build_graph(coords_l, 5.0)
    deg_l = A_l.sum(1); conn = deg_l > 0
    ys_ln = ys_ln[conn]; type_nums = type_nums[conn]; coords_l = coords_l[conn]
    A_l = A_l[np.ix_(conn, conn)]; n_l = int(conn.sum())
    coords_l_km = coords_l.copy(); coords_l_km[:, 0] *= 111; coords_l_km[:, 1] *= 85
    d_site = type_nums / type_nums.mean(); d_site = np.clip(d_site + 0.1, 0.1, 10)

    run_dataset('LAQN', coords_l_km, A_l, n_l, ys_ln, d_site, 'Dinv')

    # ==================== CA Weather ====================
    stations = []
    with open('graph_gp/data_cache/weather/ghcnd-stations.txt') as fh:
        for line in fh:
            sid = line[:11].strip(); lat = float(line[12:20]); lon = float(line[21:30])
            if sid.startswith('US') and 32 < lat < 42 and -125 < lon < -114:
                stations.append({'id': sid, 'lat': lat, 'lon': lon})
    id_to_info = {s['id']: s for s in stations}
    rows = []
    with gzip.open('graph_gp/data_cache/weather/2023.csv.gz', 'rt') as fh:
        for line in fh:
            parts = line.strip().split(',')
            if len(parts) >= 4 and parts[0] in id_to_info and parts[2] == 'TMAX':
                rows.append({'id': parts[0], 'date': parts[1], 'tmax': float(parts[3]) / 10})
    df = pd.DataFrame(rows); pivot = df.pivot_table(index='date', columns='id', values='tmax')
    good = pivot.columns[pivot.notna().sum() > 300]; pivot = pivot[good].fillna(pivot[good].mean())
    coords_w = np.array([[id_to_info[sid]['lat'], id_to_info[sid]['lon']] for sid in good])
    coords_w_km = coords_w.copy(); coords_w_km[:, 0] *= 111; coords_w_km[:, 1] *= 85
    Dd = cdist(coords_w_km, coords_w_km); A_w = ((Dd < 50) & (Dd > 0)).astype(float)
    deg_w = A_w.sum(1); conn_w = deg_w > 0
    A_w = A_w[np.ix_(conn_w, conn_w)]; vals_w = pivot.values[:, np.where(conn_w)[0]]
    coords_w_km = coords_w_km[np.where(conn_w)[0]]
    n_w = int(conn_w.sum()); deg_w = A_w.sum(1)
    if n_w > 250:
        rng_sub = np.random.RandomState(0); sub = rng_sub.choice(n_w, 200, replace=False); sub.sort()
        A_w = A_w[np.ix_(sub, sub)]; vals_w = vals_w[:, sub]; coords_w_km = coords_w_km[sub]
        deg_w = A_w.sum(1); c2 = deg_w > 0
        A_w = A_w[np.ix_(c2, c2)]; vals_w = vals_w[:, np.where(c2)[0]]
        coords_w_km = coords_w_km[np.where(c2)[0]]
        n_w = int(c2.sum())
    d_var_w = vals_w.var(axis=0); d_raw_w = d_var_w / d_var_w.mean(); d_raw_w = np.clip(d_raw_w, 0.1, 10)
    y_w = vals_w[180, :]; y_w = (y_w - y_w.mean()) / (y_w.std() + 1e-8)

    run_dataset('CA Weather', coords_w_km, A_w, n_w, y_w, d_raw_w, 'AD')

    log('\nDone.')
    f.close()

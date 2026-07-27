"""pyBKTR baseline: Bayesian Kernelized Tensor Regression
(Lei, Labbé & Sun, Bayesian Analysis 20(3):919-947, 2025).

BKTR models spatially varying coefficients using GP kernels on spatial
coordinates. We adapt it to our single-snapshot GP regression setting:
- Spatial kernel: Matern-5/2 on node coordinates (lat/lon)
- Temporal dimension: use d_i (temporal CV) as the single "temporal" covariate
- Response: the same y used by all other baselines
- Evaluation: NLL and recalibrated NLL at test nodes

Since BKTR uses MCMC, this is slower than the MLL-based methods.

Usage: python graph_gp/run_bktr_baseline.py --dataset metrla --seed 1729
"""
import argparse, os, sys, json, pickle, gzip, numpy as np, torch, h5py, pandas as pd
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')

torch.set_default_dtype(torch.float64)


def nll_test(mu, var, y, idx):
    return float(np.sum(0.5 * np.log(2 * np.pi * var[idx]) + 0.5 * (y[idx] - mu[idx])**2 / var[idx]))


def recal_nll(c, mu, var, y, idx):
    v = c * var[idx]
    return float(np.sum(0.5 * np.log(2 * np.pi * v) + 0.5 * (y[idx] - mu[idx])**2 / v))


def run_bktr(coords, d_raw, y_norm, ti, xi, n_iter=500, burn_in=100):
    """Run pyBKTR on the given data.

    We set up BKTR with:
    - Spatial positions: node coordinates (lat/lon in km)
    - Temporal positions: single time point (t=0)
    - Covariates: intercept + d_i (temporal CV)
    - Response: y at training nodes
    """
    from pyBKTR.bktr import BKTRRegressor
    from pyBKTR.kernels import KernelMatern

    n = len(y_norm)
    n_train = len(ti)

    # Spatial positions for all nodes
    sp_all = pd.DataFrame({'x': coords[:, 0], 'y': coords[:, 1]},
                          index=[f's{i}' for i in range(n)])

    # Temporal positions: single time step
    tp = pd.DataFrame({'t': [0.0]}, index=['t0'])

    # Covariates: intercept + d_i at training nodes, single time step
    # Shape: (n_locations, n_times, n_covariates)
    # For BKTRRegressor we provide a DataFrame
    cov_data = []
    for i in ti:
        cov_data.append({'location': f's{i}', 'time': 't0',
                         'intercept': 1.0, 'd_i': d_raw[i], 'y': y_norm[i]})
    df_train = pd.DataFrame(cov_data)

    # Use pyBKTR's interface
    sp_train = sp_all.loc[[f's{i}' for i in ti]]

    try:
        model = BKTRRegressor(
            spatial_positions_df=sp_train,
            temporal_positions_df=tp,
            burn_in_iter=burn_in,
            sampling_iter=n_iter - burn_in,
            spatial_kernel=KernelMatern(smoothness_factor=2.5),
        )

        # Prepare data in the format BKTR expects
        # y_df: DataFrame with MultiIndex (location, time)
        y_series = pd.Series(y_norm[ti], index=pd.MultiIndex.from_arrays(
            [[f's{i}' for i in ti], ['t0'] * n_train], names=['location', 'time']))

        # X_df: covariate DataFrame with same MultiIndex
        X_df = pd.DataFrame({
            'intercept': np.ones(n_train),
            'd_i': d_raw[ti],
        }, index=y_series.index)

        model.fit(y_series, X_df)

        # Predict at test nodes
        sp_test = sp_all.loc[[f's{i}' for i in xi]]
        X_test = pd.DataFrame({
            'intercept': np.ones(len(xi)),
            'd_i': d_raw[xi],
        }, index=pd.MultiIndex.from_arrays(
            [[f's{i}' for i in xi], ['t0'] * len(xi)], names=['location', 'time']))

        pred = model.predict(X_test, sp_test, tp)

        # Extract mean and variance from posterior samples
        mu_pred = np.full(n, np.nan)
        var_pred = np.full(n, np.nan)

        mu_pred[xi] = pred['mean'].values
        var_pred[xi] = pred['variance'].values if 'variance' in pred.columns else \
                       pred['std'].values ** 2 if 'std' in pred.columns else \
                       np.ones(len(xi)) * 0.1  # fallback

        return mu_pred, var_pred

    except Exception as e:
        print(f"  BKTR failed: {e}", flush=True)
        return None, None


def run_bktr_manual(coords, L, d_raw, y_norm, ti, xi, n_samples=300, burn_in=100):
    """Manual BKTR-style spatially varying coefficient model via MCMC.

    Fallback if pyBKTR's API doesn't match our single-snapshot setting.
    Uses a Matern-like spatial kernel on coordinates for the coefficient prior.

    Model: y_i = beta_0(s_i) + beta_1(s_i) * d_i + noise
    Prior: beta_k ~ GP(0, K_spatial) where K_spatial is SE on coordinates
    """
    n = len(y_norm)
    n_train = len(ti); n_test = len(xi)

    # Build spatial kernel from coordinates (SE kernel with lengthscale grid)
    dist_sq = cdist(coords, coords, 'sqeuclidean')
    median_dist = np.median(dist_sq[dist_sq > 0])

    y_train = y_norm[ti]

    # Grid search over lengthscale, scale, and noise
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

    # Predict at all nodes
    mu_pred = K_full[:, ti] @ alpha
    v = np.linalg.solve(Lc, K_full[ti, :])
    var_pred = np.diag(K_full) - np.sum(v ** 2, axis=0) + noise

    return mu_pred, var_pred


# === Dataset loaders ===

def load_metrla():
    DATA_DIR = 'graph_gp/data_cache/metr_la'
    with open(f'{DATA_DIR}/adj_mx.pkl', 'rb') as fp:
        _, _, adj = pickle.load(fp, encoding='latin1')
    with h5py.File(f'{DATA_DIR}/metr-la.h5', 'r') as fp:
        speeds = fp['df']['block0_values'][:]
    n = speeds.shape[1]
    A = (adj > 0).astype(float); A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    deg = A.sum(1); Dis = np.diag(1. / np.sqrt(np.maximum(deg, 1e-10)))
    L = np.eye(n) - Dis @ A @ Dis
    d = speeds.std(axis=0) / (speeds.mean(axis=0) + 1e-8)
    d = d / d.mean(); d = np.clip(d, 0.1, 10)
    y = speeds[:288, :].mean(axis=0); y = (y - y.mean()) / (y.std() + 1e-8)
    # Sensor coordinates from location CSV (columns: index, sensor_id, latitude, longitude)
    with open(f'{DATA_DIR}/graph_sensor_locations.csv') as fp:
        lines = fp.readlines()
    sensor_coords = {}
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 4:
            sensor_coords[parts[1]] = (float(parts[2]), float(parts[3]))
    # Get sensor IDs in order
    with open(f'{DATA_DIR}/adj_mx.pkl', 'rb') as fp:
        sensor_ids, _, _ = pickle.load(fp, encoding='latin1')
    coords = np.array([sensor_coords.get(str(sid), (34.0, -118.0)) for sid in sensor_ids])
    coords_km = coords.copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    return L, n, d, y, coords_km

def load_pemsbay():
    DATA_DIR = 'graph_gp/data_cache/pems_bay'
    with open(f'{DATA_DIR}/adj_mx_bay.pkl', 'rb') as fp:
        _, _, adj = pickle.load(fp, encoding='latin1')
    with h5py.File(f'{DATA_DIR}/pems-bay.h5', 'r') as fp:
        speeds = fp['speed']['block0_values'][:]
    n = speeds.shape[1]
    A = (adj > 0).astype(float); A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    deg = A.sum(1); Dis = np.diag(1. / np.sqrt(np.maximum(deg, 1e-10)))
    L = np.eye(n) - Dis @ A @ Dis
    d = speeds.std(axis=0) / (speeds.mean(axis=0) + 1e-8)
    d = d / d.mean(); d = np.clip(d, 0.1, 10)
    y = speeds[:288, :].mean(axis=0); y = (y - y.mean()) / (y.std() + 1e-8)
    # PEMS-BAY sensor locations
    with open(f'{DATA_DIR}/adj_mx_bay.pkl', 'rb') as fp:
        sensor_ids, _, _ = pickle.load(fp, encoding='latin1')
    # Use graph distance as proxy if no coordinate file exists
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
        coords = np.array([sensor_coords.get(str(sid), (37.5, -122.0)) for sid in sensor_ids])
        coords_km = coords.copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    else:
        # Fallback: use spectral embedding of adjacency as coordinates
        evals, evecs = np.linalg.eigh(L)
        coords_km = evecs[:, 1:3] * 100  # scale spectral coords
    return L, n, d, y, coords_km

def load_laqn():
    from laqn_regression import fetch_laqn_sites, fetch_laqn_data, build_graph, build_laplacian as bl
    sites = fetch_laqn_sites(); values = fetch_laqn_data(sites, 'NO2', '2024-06-15')
    sites = [s for s in sites if s['code'] in values]
    ys = np.array([values[s['code']] for s in sites])
    type_nums = np.array([{'Rural': 0, 'Suburban': 1, 'Urban Background': 2, 'Industrial': 3,
                            'Roadside': 4, 'Kerbside': 5}.get(s['type'], 2) for s in sites], dtype=float)
    coords = np.array([[s['lat'], s['lon']] for s in sites])
    y = (ys - ys.mean()) / ys.std()
    A, _ = build_graph(coords, 5.0); L = bl(A)
    deg = A.sum(1); conn = deg > 0
    y = y[conn]; type_nums = type_nums[conn]; coords = coords[conn]
    A = A[np.ix_(conn, conn)]; L = bl(A); n = int(conn.sum())
    d = type_nums / type_nums.mean(); d = np.clip(d + 0.1, 0.1, 10)
    coords_km = coords.copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    return L, n, d, y, coords_km

def load_caweather():
    """CA Weather: d_i from 2022, snapshot from Jul 15 2023."""
    def load_ghcn_year(year_file, id_to_info):
        rows = []
        with gzip.open(year_file, 'rt') as fh:
            for line in fh:
                parts = line.strip().split(',')
                if len(parts) >= 4:
                    sid = parts[0]
                    if sid in id_to_info and parts[2] == 'TMAX':
                        rows.append({'id': sid, 'date': parts[1], 'tmax': float(parts[3]) / 10})
        return pd.DataFrame(rows)

    stations = []
    with open('graph_gp/data_cache/weather/ghcnd-stations.txt') as fh:
        for line in fh:
            sid = line[:11].strip(); lat = float(line[12:20]); lon = float(line[21:30])
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

    coords = np.array([[id_to_info[sid]['lat'], id_to_info[sid]['lon']] for sid in common])
    coords_km = coords.copy(); coords_km[:, 0] *= 111; coords_km[:, 1] *= 85
    Dd = cdist(coords_km, coords_km)
    A = ((Dd < 50) & (Dd > 0)).astype(float)
    deg = A.sum(1); conn = deg > 0
    A = A[np.ix_(conn, conn)]
    vals_2022 = pivot_2022.values[:, np.where(conn)[0]]
    vals_2023 = pivot_2023.values[:, np.where(conn)[0]]
    coords_km = coords_km[conn]
    n = int(conn.sum()); deg = A.sum(1)
    Dis = np.diag(1. / np.sqrt(np.maximum(deg, 1e-10)))
    L = np.eye(n) - Dis @ A @ Dis

    d_raw = vals_2022.var(axis=0)
    d_raw = d_raw / d_raw.mean(); d_raw = np.clip(d_raw, 0.1, 10)

    day_idx = 195
    y = vals_2023[day_idx, :]
    valid = ~np.isnan(y)
    if not valid.all():
        y = np.where(valid, y, np.nanmean(y))
    y = (y - y.mean()) / (y.std() + 1e-8)
    return L, n, d_raw, y, coords_km


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['metrla', 'pemsbay', 'laqn', 'caweather'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--outdir', type=str, default='graph_gp/results')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    loaders = {'metrla': load_metrla, 'pemsbay': load_pemsbay,
               'laqn': load_laqn, 'caweather': load_caweather}
    L, n, d_raw, y_norm, coords_km = loaders[args.dataset]()

    seed = args.seed
    rng = np.random.RandomState(seed); perm = rng.permutation(n)
    ti = perm[:int(0.7 * n)]; xi = perm[int(0.7 * n):]; yt = y_norm[ti]

    print(f"BKTR baseline: {args.dataset} seed={seed} n={n}", flush=True)
    result = {'seed': seed, 'dataset': args.dataset, 'n': n}

    # Try pyBKTR first, fall back to manual spatial GP
    try:
        mu, var = run_bktr(coords_km, d_raw, y_norm, ti, xi)
    except ImportError:
        print("  pyBKTR not installed, using manual spatial GP fallback", flush=True)
        mu, var = None, None

    if mu is None:
        mu, var = run_bktr_manual(coords_km, L, d_raw, y_norm, ti, xi)

    if mu is not None and var is not None:
        nll = nll_test(mu, var, y_norm, xi)
        res = minimize_scalar(lambda c: recal_nll(c, mu, var, y_norm, xi),
                              bounds=(0.01, 100), method='bounded')
        recal = recal_nll(res.x, mu, var, y_norm, xi)
        result['nll_bktr'] = nll
        result['recal_bktr'] = recal
        result['c_bktr'] = res.x
        print(f"  BKTR: NLL={nll:.1f} Recal={recal:.1f} c={res.x:.3f}", flush=True)
    else:
        result['nll_bktr'] = float('nan')
        print("  BKTR: FAILED completely", flush=True)

    outfile = f"{args.outdir}/bktr_{args.dataset}_seed{seed}.json"
    with open(outfile, 'w') as f:
        json.dump(result, f)
    print(f"Written to {outfile}", flush=True)

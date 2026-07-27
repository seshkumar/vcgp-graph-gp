"""Multi-dataset d_i auxiliary-field sweep.

One (dataset, d_field, K) per job; loops all 20 seeds internally.

Usage:
    python graph_gp/run_dataset_di_sweep.py \
        --dataset metrla --d_field std --K 2 --w 1.0

For CA Weather the dataset has 12 monthly snapshots; we evaluate every
seed-month pair and write one JSON per (seed, month).

Supported (dataset, d_field) pairs (log_mean dropped: it leaks y on PM2.5):
    metrla, pemsbay : cv, std, spatialy_k4, spatialy_k8, spatialy_k12
    laqn            : default (site-type ordinal), spatialy_k4, spatialy_k8, spatialy_k12
    pm25            : cv, std, daily_p95, spatialy_k4, spatialy_k8, spatialy_k12
    caweather       : cv, std, spatialy_k4, spatialy_k8, spatialy_k12

Unsupported pairs no-op exit with status='not_applicable'.

Output JSON per (seed, [month]):
    graph_gp/results/{dataset}_disweep_d-{field}_K{K}_w{w}_seed{seed}[_day{day}].json
fields: nll, rmse, recal, c_recal, gamma_*, tau, scale, noise, status, ...
"""
import argparse, os, sys, json, time
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')
torch.set_default_dtype(torch.float64)

from baselines import gp_posterior
from run_all_methods import (
    compute_laplacian, fit_het_joint, fit_semisup_joint,
    learn_gamma, learn_gamma_per_cluster,
    load_metrla, load_pemsbay, load_laqn, load_pm25,
)

SEEDS = list(range(1729, 1749))
CAWEATHER_DAYS = [14, 45, 73, 104, 134, 165, 195, 226, 257, 287, 318, 348]

DATASET_DFIELDS = {
    # log_mean removed: it leaks the prediction target on PM2.5 (annual
    # mean = y). Kept out of every dataset for symmetry and to avoid
    # accidental leakage on the others as well.
    'metrla':    {'cv', 'std', 'spatialy_k4', 'spatialy_k8', 'spatialy_k12'},
    'pemsbay':   {'cv', 'std', 'spatialy_k4', 'spatialy_k8', 'spatialy_k12'},
    'laqn':      {'default', 'spatialy_k4', 'spatialy_k8', 'spatialy_k12'},
    'pm25':      {'cv', 'std', 'daily_p95', 'spatialy_k4', 'spatialy_k8', 'spatialy_k12'},
    'caweather': {'cv', 'std', 'daily_p95', 'spatialy_k4', 'spatialy_k8', 'spatialy_k12'},
}


def normalise(x):
    x = np.asarray(x, dtype=np.float64)
    x = x / x.mean()
    return np.clip(x, 0.1, 10)


def spatial_std_of_y(yt, ti, all_coords, n, k):
    tc = all_coords[ti]
    out = np.zeros(n)
    nn = NearestNeighbors(n_neighbors=min(k, len(ti))).fit(tc)
    _, nbrs = nn.kneighbors(all_coords)
    for i in range(n):
        out[i] = float(np.std(yt[nbrs[i]]) + 1e-6)
    return out


def aug_labels(d_field, coords_km, K, w):
    cs = (coords_km - coords_km.mean(0)) / coords_km.std(0)
    ds = ((d_field - d_field.mean()) / (d_field.std() + 1e-12)).reshape(-1, 1)
    feats = np.column_stack([cs, w * ds])
    return KMeans(n_clusters=K, random_state=0, n_init=10).fit(feats).labels_


def block_lap(coords, lab, k=8):
    n = len(lab)
    A = np.zeros((n, n))
    for kk in range(int(lab.max()) + 1):
        idx = np.where(lab == kk)[0]
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx))).fit(coords[idx])
        _, nb = nn.kneighbors(coords[idx])
        for li, ne in enumerate(nb):
            for lj in ne[1:]:
                A[idx[li], idx[lj]] = 1.0
    A = np.maximum(A, A.T)
    return A, np.diag(A.sum(1)) - A


def lgamma_kernel(L, d, tau, scale, gammas, labels=None):
    d = np.maximum(d, 1e-8)
    if labels is None:
        gp_per_node = np.full(len(L), gammas)
    else:
        gp_per_node = gammas[labels]
    dp = np.clip(np.power(d, gp_per_node / 2.0), 1e-4, 1e4)
    Q = np.diag(1.0 / d) + 2.0 * tau * (dp[:, None] * L * dp[None, :])
    K = scale * np.linalg.inv(Q + 1e-8 * np.eye(len(L)))
    return 0.5 * (K + K.T)


def recal_nll(mu, var, y, idx):
    from scipy.optimize import minimize_scalar
    r = (y[idx] - mu[idx]) ** 2
    v = var[idx]
    def loss(log_c):
        c = np.exp(log_c)
        return 0.5 * np.sum(np.log(2 * np.pi * c * v) + r / (c * v))
    res = minimize_scalar(loss, bounds=(-6, 6), method='bounded')
    return float(res.fun), float(np.exp(res.x))


# ---------------------------------------------------------------------
# Per-dataset payload loaders.
# Returns dict with keys: A, L, n, y_norm, coords_km, base_d, [extra...].
# `base_d` is the dataset's stored auxiliary (CV-style for sensors,
# site-type ordinal for LAQN, 2022-variance for CA Weather). For LAQN we
# also keep that as the only natively-supported numeric d_field.
# ---------------------------------------------------------------------
def load_with_extras(dataset):
    if dataset == 'metrla':
        A, L, n, y_norm, d_raw, coords_km, _ = load_metrla()
        import h5py
        with h5py.File('graph_gp/data_cache/metr_la/metr-la.h5', 'r') as h:
            speeds = h['df']['block0_values'][:]
        std = speeds.std(axis=0)
        mean = speeds.mean(axis=0) + 1e-8
        return {'A': A, 'L': L, 'n': n, 'y_norm': y_norm, 'coords_km': coords_km,
                'base_d': d_raw, 'std_t': std, 'mean_t': mean, 'p95_t': None}
    if dataset == 'pemsbay':
        A, L, n, y_norm, d_raw, coords_km, _ = load_pemsbay()
        import h5py
        with h5py.File('graph_gp/data_cache/pems_bay/pems-bay.h5', 'r') as h:
            speeds = h['speed']['block0_values'][:]
        std = speeds.std(axis=0)
        mean = speeds.mean(axis=0) + 1e-8
        return {'A': A, 'L': L, 'n': n, 'y_norm': y_norm, 'coords_km': coords_km,
                'base_d': d_raw, 'std_t': std, 'mean_t': mean, 'p95_t': None}
    if dataset == 'laqn':
        A, L, n, y_norm, d_raw, coords_km, _ = load_laqn()
        return {'A': A, 'L': L, 'n': n, 'y_norm': y_norm, 'coords_km': coords_km,
                'base_d': d_raw, 'std_t': None, 'mean_t': None, 'p95_t': None}
    if dataset == 'pm25':
        # Re-read raw to recover temporal std/mean/p95 unnormalised.
        csv_path = 'graph_gp/data_cache/pm25/daily_88101_2023/daily_88101_2023.csv'
        df = pd.read_csv(csv_path, usecols=['State Code','County Code','Site Num','POC',
                                            'Latitude','Longitude','Arithmetic Mean','Date Local'])
        df['site'] = df['State Code'].astype(str)+'_'+df['County Code'].astype(str)+'_'+df['Site Num'].astype(str)
        daily = df.groupby(['site','Date Local'])['Arithmetic Mean'].mean().reset_index()
        agg = daily.groupby('site')['Arithmetic Mean']
        feats = agg.agg(['mean','std']).reset_index().dropna()
        p95 = agg.quantile(0.95).reset_index().rename(columns={'Arithmetic Mean':'p95'})
        feats = feats.merge(p95, on='site')
        locs = df.drop_duplicates('site')[['site','Latitude','Longitude']]
        feats = feats.merge(locs, on='site')
        feats = feats[(feats['Latitude'].between(24,50)) & (feats['Longitude'].between(-125,-66))].reset_index(drop=True)
        n = len(feats)
        y = feats['mean'].values.astype(np.float64)
        y_norm = (y - y.mean()) / (y.std() + 1e-8)
        coords_km = np.stack([feats['Latitude'].values*111.0,
                              feats['Longitude'].values*85.0], 1).astype(np.float64)
        # Build 8-NN graph
        dists = cdist(coords_km, coords_km); np.fill_diagonal(dists, np.inf)
        nn8 = np.argpartition(dists, 8, axis=1)[:, :8]
        A = np.zeros((n, n))
        for i in range(n):
            for j in nn8[i]: A[i,j]=1.0; A[j,i]=1.0
        L = compute_laplacian(A)
        std = feats['std'].values.astype(np.float64)
        mean = feats['mean'].values.astype(np.float64)
        p95 = feats['p95'].values.astype(np.float64)
        base_d = std / (mean + 1e-8)
        return {'A': A, 'L': L, 'n': n, 'y_norm': y_norm, 'coords_km': coords_km,
                'base_d': base_d, 'std_t': std, 'mean_t': mean, 'p95_t': p95}
    if dataset == 'caweather':
        # caweather is per-month; we keep the dataset-loader output.
        from run_all_methods import load_caweather
        # load_caweather signature: returns full graph + full per-day y
        # We need: A, L, n, coords, full daily data to compute std/mean.
        # The simplest path: reuse load_caweather to get base d, and recover
        # daily series by re-reading the cached CSV.
        out = load_caweather(day=14)  # base call, returns one month
        A, L, n, y_norm, d_raw, coords_km, _ = out
        # daily TMAX from 2022 cache for std/mean
        cache = 'graph_gp/data_cache/weather/2022.csv.gz'
        if os.path.exists(cache):
            tmax = pd.read_csv(cache)
            # very dataset-specific reconstruction; if cache layout differs,
            # std_t/mean_t fall back to the CV that load_caweather produced.
            std_t = mean_t = None
        else:
            std_t = mean_t = None
        return {'A': A, 'L': L, 'n': n, 'y_norm': y_norm, 'coords_km': coords_km,
                'base_d': d_raw, 'std_t': std_t, 'mean_t': mean_t, 'p95_t': None,
                '_caweather_per_day': True}
    raise ValueError(dataset)


def build_d_field(name, payload, yt, ti):
    coords_km = payload['coords_km']
    n = payload['n']
    if name == 'default':
        return normalise(payload['base_d'])
    if name == 'cv':
        if payload.get('std_t') is None or payload.get('mean_t') is None:
            return None
        return normalise(payload['std_t'] / (payload['mean_t'] + 1e-8))
    if name == 'std':
        if payload.get('std_t') is None:
            return None
        return normalise(payload['std_t'])
    if name == 'daily_p95':
        if payload.get('p95_t') is None:
            return None
        return normalise(payload['p95_t'])
    if name.startswith('spatialy_k'):
        k = int(name.split('k')[1])
        return normalise(spatial_std_of_y(yt, ti, coords_km, n, k))
    raise ValueError(name)


def het_kernel(L, n, d_raw, tau, scale):
    """K_h = s * D^{1/2} (I + 2 tau L)^{-1} D^{1/2}."""
    d = np.maximum(d_raw, 1e-8)
    D_sqrt = np.sqrt(d)
    Q = np.eye(n) + 2.0 * tau * L
    Kinv = np.linalg.inv(Q + 1e-8 * np.eye(n))
    K = scale * (D_sqrt[:, None] * Kinv * D_sqrt[None, :])
    return 0.5 * (K + K.T)


def semisup_kernel(L, n, d_raw, tau, scale):
    """K_ss = s * (D^{-1} + 2 tau L)^{-1}."""
    d = np.maximum(d_raw, 1e-8)
    Q = np.diag(1.0 / d) + 2.0 * tau * L
    K = scale * np.linalg.inv(Q + 1e-8 * np.eye(n))
    return 0.5 * (K + K.T)


def fit_one(L, n, d_field, y_norm, ti, xi, K_clusters, coords_km, w, method='lgamma'):
    yt = y_norm[ti]
    if method == 'het':
        res = fit_het_joint(L, n, d_field, yt, ti)
        if res is None: return None
        tau, scale, noise = res
        K_ker = het_kernel(L, n, d_field, tau, scale)
        labels_info, gamma_info = None, -1.0
    elif method == 'semisup':
        res = fit_semisup_joint(L, n, d_field, yt, ti)
        if res is None: return None
        tau, scale, noise = res
        K_ker = semisup_kernel(L, n, d_field, tau, scale)
        labels_info, gamma_info = None, 0.0
    elif K_clusters == 1:
        tau, scale, noise = fit_het_joint(L, n, d_field, yt, ti)
        res = learn_gamma(L, n, d_field, yt, ti, init=(tau, scale, noise))
        if res is None:
            return None
        tau, scale, noise, gamma = res
        K_ker = lgamma_kernel(L, d_field, tau, scale, gamma, labels=None)
        labels_info = None
        gamma_info = float(gamma)
    else:
        labels = aug_labels(d_field, coords_km, K_clusters, w)
        sizes = np.bincount(labels, minlength=K_clusters).tolist()
        A_bl, L_bl = block_lap(coords_km, labels)
        tau, scale, noise = fit_het_joint(L_bl, n, d_field, yt, ti)
        res = learn_gamma_per_cluster(L_bl, n, d_field, yt, ti, labels, init=(tau, scale, noise))
        if res is None:
            return None
        tau, scale, noise, gammas = res
        L_used = L_bl
        K_ker = lgamma_kernel(L_used, d_field, tau, scale, gammas, labels=labels)
        labels_info = sizes
        gamma_info = gammas.tolist()
    mu, var, _ = gp_posterior(K_ker, yt, ti, noise)
    nll = float(np.sum(0.5*np.log(2*np.pi*var[xi]) + 0.5*(y_norm[xi]-mu[xi])**2/var[xi]))
    rmse = float(np.sqrt(np.mean((y_norm[xi]-mu[xi])**2)))
    rec, c_star = recal_nll(mu, var, y_norm, xi)
    return {
        'tau': tau, 'scale': scale, 'noise': noise,
        'gamma': gamma_info, 'cluster_sizes': labels_info,
        'nll': nll, 'rmse': rmse, 'recal': rec, 'c_recal': c_star,
    }


def run_seed_loop(dataset, d_field_name, K, w, payload, outdir, method='lgamma', day=None):
    """Run all 20 seeds for one cell.
    method='lgamma' uses K (K=1 scalar lgamma, K>=2 lgamma_block_aug);
    method='het' fits at gamma=-1; method='semisup' fits at gamma=0.
    For het/semisup the K argument is ignored on the model side but still
    threaded into the file name (use K=0 in PBS for het, K=-1 for semisup,
    or any single value to namespace the output)."""
    n = payload['n']
    coords_km = payload['coords_km']
    L = payload['L']
    y_norm = payload['y_norm']

    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        ti = perm[:int(0.7 * n)]
        xi = perm[int(0.7 * n):]
        yt = y_norm[ti]
        d_field = build_d_field(d_field_name, payload, yt, ti)
        if d_field is None:
            payload_out = {'status': 'not_applicable',
                           'reason': f'd_field {d_field_name} not buildable for {dataset}'}
        else:
            t0 = time.time()
            r = fit_one(L, n, d_field, y_norm, ti, xi, K, coords_km, w, method=method)
            dt = time.time() - t0
            if r is None:
                payload_out = {'status': 'fit_failed', 'elapsed_s': dt}
            else:
                payload_out = {'status': 'ok', 'elapsed_s': dt, **r}
        out = {
            'dataset': dataset, 'd_field': d_field_name, 'K': K, 'w': w,
            'method': method, 'seed': seed, 'n': int(n), **payload_out,
        }
        if day is not None:
            out['day'] = day
            day_tag = f'_day{day}'
        else:
            day_tag = ''
        # File name: method-tagged for het/semisup so they don't collide
        # with the lgamma sweep's existing files.
        method_tag = '' if method == 'lgamma' else f'_{method}'
        fn = (f'{outdir}/{dataset}_disweep_d-{d_field_name}_K{K}_w{w:.1f}'
              f'{method_tag}{day_tag}_seed{seed}.json')
        with open(fn, 'w') as f:
            json.dump(out, f)
        if payload_out.get('status') == 'ok':
            n_test = len(xi)
            print(f'  seed={seed}: NLL/pt={r["nll"]/n_test:.4f}  RMSE={r["rmse"]:.4f}  '
                  f'recNLL/pt={r["recal"]/n_test:.4f}  c*={r["c_recal"]:.3f}  '
                  f'({dt:.1f}s)', flush=True)
        else:
            print(f'  seed={seed}: status={payload_out.get("status")}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['metrla', 'pemsbay', 'laqn', 'pm25', 'caweather'])
    p.add_argument('--d_field', required=True,
                   choices=['default', 'cv', 'std', 'daily_p95',
                            'spatialy_k4', 'spatialy_k8', 'spatialy_k12'])
    p.add_argument('--K', type=int, required=True,
                   choices=[-1, 0, 1, 2, 3, 5, 10, 20],
                   help='K=1 scalar lgamma; K>=2 lgamma_block_aug; '
                        'when --method=het use K=0, --method=semisup use K=-1 '
                        '(K is just a tag in those cases).')
    p.add_argument('--method', default='lgamma',
                   choices=['lgamma', 'het', 'semisup'],
                   help='Model variant. lgamma uses --K; het fixes gamma=-1; '
                        'semisup fixes gamma=0.')
    p.add_argument('--w', type=float, default=1.0)
    p.add_argument('--outdir', default='graph_gp/results')
    args = p.parse_args()

    if args.d_field not in DATASET_DFIELDS[args.dataset]:
        print(f'd_field={args.d_field} not supported for dataset={args.dataset}; exit 0.')
        return

    os.makedirs(args.outdir, exist_ok=True)
    print(f'=== {args.dataset} | method={args.method} | d_field={args.d_field} | '
          f'K={args.K} | w={args.w} ===', flush=True)

    if args.dataset == 'caweather':
        # Load 2022 daily TMAX once to derive std/mean/p95 per station,
        # so cv/std/daily_p95 d_field variants are buildable. Mirrors
        # load_caweather but returns extras and reuses the same station
        # filtering to keep indices aligned with load_caweather(day=...).
        import gzip
        stations = []
        with open('graph_gp/data_cache/weather/ghcnd-stations.txt') as fh:
            for line in fh:
                sid = line[:11].strip()
                lat = float(line[12:20]); lon = float(line[21:30])
                if sid.startswith('US') and 32 < lat < 42 and -125 < lon < -114:
                    stations.append({'id': sid, 'lat': lat, 'lon': lon})
        id_to_info = {s['id']: s for s in stations}

        def _load_year(path):
            rows = []
            with gzip.open(path, 'rt') as fh:
                for line in fh:
                    parts = line.strip().split(',')
                    if len(parts) >= 4 and parts[0] in id_to_info and parts[2] == 'TMAX':
                        rows.append({'id': parts[0], 'date': parts[1],
                                     'tmax': float(parts[3]) / 10})
            return pd.DataFrame(rows)

        df22 = _load_year('graph_gp/data_cache/weather/2022.csv.gz')
        df23 = _load_year('graph_gp/data_cache/weather/2023.csv.gz')
        pv22 = df22.pivot_table(index='date', columns='id', values='tmax')
        pv23 = df23.pivot_table(index='date', columns='id', values='tmax')
        good22 = set(pv22.columns[pv22.notna().sum() > 300])
        good23 = set(pv23.columns[pv23.notna().sum() > 300])
        common = sorted(good22 & good23)
        pv22 = pv22[common].fillna(pv22[common].mean())
        vals_2022 = pv22.values  # shape (n_days_2022, n_stations_common)
        # connected-component filter (50 km threshold) -- replicate exactly.
        coords_full = np.array([[id_to_info[sid]['lat'], id_to_info[sid]['lon']]
                                for sid in common])
        coords_km_full = coords_full.copy()
        coords_km_full[:, 0] *= 111; coords_km_full[:, 1] *= 85
        Dd = cdist(coords_km_full, coords_km_full)
        A_full = ((Dd < 50) & (Dd > 0)).astype(float)
        deg_full = A_full.sum(1); conn = deg_full > 0
        vals_2022 = vals_2022[:, conn]
        std_t = vals_2022.std(axis=0)
        mean_t = vals_2022.mean(axis=0) + 1e-8
        p95_t = np.quantile(vals_2022, 0.95, axis=0)

        for day in CAWEATHER_DAYS:
            print(f'-- day {day} --', flush=True)
            from run_all_methods import load_caweather
            A, L, n, y_norm, d_raw, coords_km, _ = load_caweather(day=day)
            assert len(std_t) == n, f'station count mismatch: {len(std_t)} vs {n}'
            payload = {'A': A, 'L': L, 'n': n, 'y_norm': y_norm,
                       'coords_km': coords_km, 'base_d': d_raw,
                       'std_t': std_t, 'mean_t': mean_t, 'p95_t': p95_t}
            run_seed_loop(args.dataset, args.d_field, args.K, args.w,
                          payload, args.outdir, method=args.method, day=day)
    else:
        payload = load_with_extras(args.dataset)
        run_seed_loop(args.dataset, args.d_field, args.K, args.w,
                      payload, args.outdir, method=args.method)


if __name__ == '__main__':
    main()

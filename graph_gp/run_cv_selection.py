"""Validation-fold model selection for the VCGP family (leakage-free).

Reports numbers under ONE fixed selection rule
(family member, and block K) chosen on a validation fold per seed, NOT the
post-hoc best member.

Protocol per seed (three-way node split, no test leakage):
    perm = rng.permutation(n)
    test        = first 30%              (held out; never used for fit/selection)
    validation  = next  20%
    train       = remaining 50%
  1. Fit each candidate's hyperparameters by marginal likelihood on TRAIN only,
     predict VALIDATION nodes, record validation NLL/pt.
  2. Select the candidate with lowest validation NLL/pt.
  3. Refit the selected candidate on TRAIN+VALIDATION (the full 70%), predict
     TEST, and report test NLL/pt and RMSE. (Oracle per-candidate test numbers
     are also recorded for reference, never used for selection.)

Auxiliary d_i comes from the leakage-free loaders in run_all_methods.py
(disjoint window for METR-LA/PEMS-BAY/PM2.5; CA Weather already clean).

Usage:
    python graph_gp/run_cv_selection.py --dataset metrla --outdir graph_gp/results_cv
    python graph_gp/run_cv_selection.py --dataset pemsbay --seeds 1729 1730   # smoke

Output: one JSON per (dataset, seed):
    {outdir}/cv_{dataset}_seed{seed}.json
"""
import argparse, os, sys, json, time
import numpy as np

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, 'graph_gp')

import torch
torch.set_default_dtype(torch.float64)

from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from baselines import gp_posterior
from run_all_methods import (
    fit_stationary, fit_het_joint, fit_semisup_joint,
    learn_gamma, learn_gamma_per_cluster, apply_blocking,
    nll_test, load_metrla, load_pemsbay, load_laqn, load_pm25, load_caweather,
)

LOADERS = {'metrla': load_metrla, 'pemsbay': load_pemsbay,
           'laqn': load_laqn, 'pm25': load_pm25}
# CA Weather is loaded per day-of-year (12 monthly snapshots, fixed graph,
# 2022 auxiliary vs 2023 target -- already leakage-free by construction).
CAWEATHER_DAYS = [14, 45, 73, 104, 134, 165, 195, 226, 257, 287, 318, 348]
SEEDS = list(range(1729, 1749))

# Selection pool per dataset. d is fixed (one leakage-free loader default per
# dataset), so selection is over the member and, for the piecewise model, the
# clustering K (and, for PM2.5's augmented block, the auxiliary weight w).
#   Simple sensor/air datasets: scalar members + coordinate-only blocks.
#   PM2.5: scalar members + augmented-K-means block_aug over (K, w) -- the
#     leader family -- so the winning (K, w) is chosen on VALIDATION, not
#     post-hoc on test.
AUG_CONFIGS = [(K, w) for K in (2, 5, 10, 20) for w in (0.5, 1.0)]
CANDIDATES_SIMPLE = ['stat', 'fem', 'het', 'lgamma',
                     'lgamma_block_K2', 'lgamma_block_K5', 'lgamma_block_K10']
CANDIDATES_PM25 = (['stat', 'fem', 'het', 'lgamma']
                   + [f'augblock_K{K}_w{w}' for K, w in AUG_CONFIGS])

# --grid full: additionally select over the auxiliary d and the ridge lambda
# (selection ranges over member, K, d AND lambda -- the base
# grid only spans member/K/w with a fixed d). Selection over lambda is realised
# by adding lgamma_fisher (lambda = data-driven Fisher ridge) so the pool
# contains both the unregularised member (lambda = 0, plain lgamma) and the
# Fisher-ridge member. Selection over d uses leakage-free auxiliaries only:
#   'def'  -> the disjoint-window loader default (annual std / disjoint CV)
#   'spk8' -> spatial std of TRAIN targets over 8 nearest train neighbours
#   'spk12'-> same over 12 neighbours (both use train values only -> no leak)
# 'augfull_*' is the full-graph piecewise-gamma ablation (every edge kept,
# gamma varies by cluster) requested to expose the cost of block-diagonalising.
GRID = 'base'
D_TAGS_FULL = ['def', 'spk8', 'spk12']


def _with_d(base_list, dtags):
    out = []
    for b in base_list:
        for dt in dtags:
            out.append(b if dt == 'def' else f'{b}__d{dt}')
    return out


def candidates_for(dataset):
    if GRID == 'base':
        # member + K (+ w for PM2.5/CA Weather), single fixed leakage-free d.
        return CANDIDATES_PM25 if dataset in ('pm25', 'caweather') else CANDIDATES_SIMPLE
    # GRID == 'full': member x K x d x lambda.
    if dataset in ('pm25', 'caweather'):
        d_using = (['het', 'lgamma', 'lgamma_fisher']
                   + [f'augblock_K{K}_w{w}' for K, w in AUG_CONFIGS])
        full_abl = [f'augfull_K{K}_w{w}' for K, w in AUG_CONFIGS]  # def-d only
        return ['stat', 'fem'] + _with_d(d_using, D_TAGS_FULL) + full_abl
    d_using = ['het', 'lgamma', 'lgamma_fisher',
               'lgamma_block_K2', 'lgamma_block_K5', 'lgamma_block_K10']
    return ['stat', 'fem'] + _with_d(d_using, D_TAGS_FULL)


def aug_block_laplacian(coords_km, d_raw, K, w, k_nn=8):
    """Augmented-K-means partition on [coords_std, w*d_std]; per-cluster k-NN
    graph on geographic coords -> block-diagonal Laplacian. Matches
    run_pm25_augmented.py (clustering uses coords+d only, never y)."""
    cs = (coords_km - coords_km.mean(0)) / coords_km.std(0)
    ds = ((d_raw - d_raw.mean()) / d_raw.std()).reshape(-1, 1)
    feats = np.column_stack([cs, w * ds])
    labels = KMeans(n_clusters=K, random_state=0, n_init=10).fit(feats).labels_
    n = len(labels)
    A = np.zeros((n, n))
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(k_nn + 1, len(idx))).fit(coords_km[idx])
        _, nb = nn.kneighbors(coords_km[idx])
        for li, neigh in enumerate(nb):
            for lj in neigh[1:]:
                A[idx[li], idx[lj]] = 1.0
    A = np.maximum(A, A.T)
    return np.diag(A.sum(1)) - A, labels


def _normalise(x):
    x = np.asarray(x, float)
    x = x / (x.mean() + 1e-12)
    return np.clip(x, 0.1, 10.0)


def _spatial_std_of_y(yt, ti, coords, n, k):
    """Leakage-free auxiliary: for every node, the std of TRAIN targets over its
    k nearest TRAIN neighbours. Uses train values only (never test)."""
    tc = coords[ti]
    kk = max(1, min(k, len(ti)))
    nn = NearestNeighbors(n_neighbors=kk).fit(tc)
    _, nbrs = nn.kneighbors(coords)
    return np.array([np.std(yt[nbrs[i]]) + 1e-6 for i in range(n)])


def build_d(dtag, d_raw, y_norm, ti, coords_km, n):
    if dtag == 'def':
        return d_raw
    if dtag.startswith('spk'):
        k = int(dtag[3:])
        return _normalise(_spatial_std_of_y(y_norm[ti], ti, coords_km, n, k))
    raise ValueError(f'unknown d-tag {dtag}')


def aug_full_laplacian(coords_km, d_raw, K, w, k_nn=8):
    """Full-graph piecewise-gamma partner of aug_block_laplacian: same augmented
    K-means labels, but ONE k-NN graph over ALL nodes (cross-cluster edges kept).
    Returns (L_full, labels)."""
    cs = (coords_km - coords_km.mean(0)) / coords_km.std(0)
    ds = ((d_raw - d_raw.mean()) / (d_raw.std() + 1e-12)).reshape(-1, 1)
    feats = np.column_stack([cs, w * ds])
    labels = KMeans(n_clusters=K, random_state=0, n_init=10).fit(feats).labels_
    n = len(labels)
    nn = NearestNeighbors(n_neighbors=min(k_nn + 1, n)).fit(coords_km)
    _, nb = nn.kneighbors(coords_km)
    A = np.zeros((n, n))
    for i, neigh in enumerate(nb):
        for j in neigh[1:]:
            A[i, j] = 1.0
    A = np.maximum(A, A.T)
    return np.diag(A.sum(1)) - A, labels


def _fisher_params(L, n, d, yt, ti, nu=0.5):
    """lgamma_fisher fit: vanilla lgamma -> Fisher Var(gamma) -> refit with
    lambda = 0.5/Var(gamma). Returns (tau, scale, noise, gamma, ...) or None."""
    from lgamma_ridge_metrla import learn_gamma_ridge
    from run_fisher_ridge import fisher_var_gamma
    het = fit_het_joint(L, n, d, yt, ti) or (0.1, 1.0, 0.1)
    th, sh, noh = het
    p0 = learn_gamma_ridge(L, n, d, yt, ti, lam=0.0, gamma_ref=-1.0,
                           init=(th, sh, noh), gamma_init=-1.0, nu=nu)
    if p0 is None:
        return None
    tau0, scale0, noise0, gamma0 = p0[0], p0[1], p0[2], p0[3]
    var_g, _ = fisher_var_gamma(tau0, scale0, noise0, gamma0, L, d, n, yt, ti, nu=nu)
    lam = 0.5 / max(var_g, 1e-12)
    return learn_gamma_ridge(L, n, d, yt, ti, lam=lam, gamma_ref=-1.0,
                             init=(th, sh, noh), gamma_init=-1.0, nu=nu)


def _sym(K):
    return 0.5 * (K + K.T)


def predict(member, L_full, n, d_raw, coords_km, y_norm, ti):
    """Fit `member` by MLL on observed nodes `ti`, return (mu, var) over all n.

    `member` may carry a '__d<tag>' suffix selecting a leakage-free auxiliary d
    for THIS fold (built from train indices `ti` only)."""
    if '__d' in member:
        base, dtag = member.split('__d', 1)
    else:
        base, dtag = member, 'def'
    yt = y_norm[ti]
    d = np.maximum(build_d(dtag, d_raw, y_norm, ti, coords_km, n), 1e-8)
    I = np.eye(n)
    if base == 'stat':
        t, s, no = fit_stationary(L_full, n, yt, ti)
        K = s * np.linalg.inv(I + 2 * t * L_full)
    elif base == 'fem':
        t, s, no = fit_semisup_joint(L_full, n, d, yt, ti)
        Q = np.diag(1.0 / d) + 2 * t * L_full
        K = s * np.linalg.inv(Q + 1e-8 * I)
    elif base == 'het':
        t, s, no = fit_het_joint(L_full, n, d, yt, ti)
        Ds = np.diag(np.sqrt(d))
        K = s * Ds @ np.linalg.inv(I + 2 * t * L_full) @ Ds
    elif base == 'lgamma':
        t_h, s_h, no_h = fit_het_joint(L_full, n, d, yt, ti)
        t, s, no, g = learn_gamma(L_full, n, d, yt, ti, init=(t_h, s_h, no_h))
        dp = np.clip(np.power(d, g / 2.0), 1e-4, 1e4)
        Q = np.diag(1.0 / d) + 2 * t * np.diag(dp) @ L_full @ np.diag(dp)
        K = s * np.linalg.inv(Q + 1e-8 * I)
    elif base == 'lgamma_fisher':
        pf = _fisher_params(L_full, n, d, yt, ti, nu=0.5)
        if pf is None:
            raise RuntimeError('lgamma_fisher fit failed')
        t, s, no, g = pf[0], pf[1], pf[2], pf[3]
        dp = np.clip(np.power(d, g / 2.0), 1e-4, 1e4)
        Q = np.diag(1.0 / d) + 2 * t * np.diag(dp) @ L_full @ np.diag(dp)
        K = s * np.linalg.inv(Q + 1e-8 * I)
    elif base.startswith('lgamma_block_K'):
        Kb = int(base.split('K')[1])
        _, L_bl, labels = apply_blocking(coords_km, Kb)
        t_h, s_h, no_h = fit_het_joint(L_bl, n, d, yt, ti)
        t, s, no, gk = learn_gamma_per_cluster(
            L_bl, n, d, yt, ti, labels, init=(t_h, s_h, no_h))
        gpn = gk[labels]
        dp = np.clip(np.power(d, gpn / 2.0), 1e-4, 1e4)
        Q = np.diag(1.0 / d) + 2 * t * np.outer(dp, dp) * L_bl
        K = s * np.linalg.inv(Q + 1e-8 * I)
    elif base.startswith('augblock_K'):
        # augblock_K{K}_w{w}: augmented-K-means block_aug (the PM2.5 leader).
        Kb = int(base.split('K')[1].split('_')[0])
        w = float(base.split('_w')[1])
        L_bl, labels = aug_block_laplacian(coords_km, d, Kb, w)
        t_h, s_h, no_h = fit_het_joint(L_bl, n, d, yt, ti)
        t, s, no, gk = learn_gamma_per_cluster(
            L_bl, n, d, yt, ti, labels, init=(t_h, s_h, no_h))
        gpn = gk[labels]
        dp = np.clip(np.power(d, gpn / 2.0), 1e-4, 1e4)
        Q = np.diag(1.0 / d) + 2 * t * np.outer(dp, dp) * L_bl
        K = s * np.linalg.inv(Q + 1e-8 * I)
    elif base.startswith('augfull_K'):
        # full-graph piecewise-gamma: every edge kept, gamma varies by cluster.
        Kb = int(base.split('K')[1].split('_')[0])
        w = float(base.split('_w')[1])
        L_fl, labels = aug_full_laplacian(coords_km, d, Kb, w)
        t_h, s_h, no_h = fit_het_joint(L_fl, n, d, yt, ti)
        t, s, no, gk = learn_gamma_per_cluster(
            L_fl, n, d, yt, ti, labels, init=(t_h, s_h, no_h))
        gpn = gk[labels]
        dp = np.clip(np.power(d, gpn / 2.0), 1e-4, 1e4)
        Q = np.diag(1.0 / d) + 2 * t * np.outer(dp, dp) * L_fl
        K = s * np.linalg.inv(Q + 1e-8 * I)
    else:
        raise ValueError(f'unknown member {member}')
    K = _sym(K)
    mu, var, _ = gp_posterior(K, y_norm[ti], ti, no)   # arg2 = observed VALUES
    return mu, var


def nll_pt(mu, var, y, idx):
    return nll_test(mu, var, y, idx) / max(1, len(idx))


def rmse(mu, y, idx):
    return float(np.sqrt(np.mean((y[idx] - mu[idx]) ** 2)))


def run_seed(dataset, seed, A, L, n, y_norm, d_raw, coords_km):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_test = int(0.30 * n)
    n_val = int(0.20 * n)
    xi = np.sort(perm[:n_test])                 # TEST (held out)
    vi = np.sort(perm[n_test:n_test + n_val])   # VALIDATION
    ti = np.sort(perm[n_test + n_val:])         # TRAIN
    tv = np.sort(perm[n_test:])                 # TRAIN+VAL (final refit)

    cands = candidates_for(dataset)
    nc = len(cands)
    # --- 1. selection on validation (fit on TRAIN only) ---
    val = {}
    for j, m in enumerate(cands, 1):
        tc = time.time()
        try:
            mu, var = predict(m, L, n, d_raw, coords_km, y_norm, ti)
            val[m] = nll_pt(mu, var, y_norm, vi)
            print(f'    [{dataset} s{seed}] val [{j:2d}/{nc}] {m:26s} '
                  f'val_nll={val[m]:.4f}  ({time.time()-tc:.1f}s)', flush=True)
        except Exception as e:
            print(f'    [{dataset} s{seed}] val [{j:2d}/{nc}] {m:26s} FAILED: {e}',
                  flush=True)
            val[m] = float('inf')
    selected = min(val, key=val.get)
    print(f'    [{dataset} s{seed}] selected -> {selected}', flush=True)

    # --- 2. selected config: refit on TRAIN+VAL, evaluate on TEST ---
    mu, var = predict(selected, L, n, d_raw, coords_km, y_norm, tv)
    sel_nll = nll_pt(mu, var, y_norm, xi)
    sel_rmse = rmse(mu, y_norm, xi)

    # --- 3. oracle: every candidate on TEST (for reference only) ---
    oracle = {}
    for j, m in enumerate(cands, 1):
        tc = time.time()
        try:
            mu_o, var_o = predict(m, L, n, d_raw, coords_km, y_norm, tv)
            oracle[m] = {'nll': nll_pt(mu_o, var_o, y_norm, xi),
                         'rmse': rmse(mu_o, y_norm, xi)}
            print(f'    [{dataset} s{seed}] oracle [{j:2d}/{nc}] {m:26s} '
                  f'test_nll={oracle[m]["nll"]:.4f}  ({time.time()-tc:.1f}s)',
                  flush=True)
        except Exception as e:
            print(f'    [{dataset} s{seed}] oracle [{j:2d}/{nc}] {m:26s} FAILED: {e}',
                  flush=True)
            oracle[m] = {'nll': float('nan'), 'rmse': float('nan')}

    return {
        'dataset': dataset, 'seed': int(seed), 'n': int(n), 'grid': GRID,
        'n_train': int(len(ti)), 'n_val': int(len(vi)), 'n_test': int(len(xi)),
        'selected_member': selected,
        'val_nll_per_candidate': {k: float(v) for k, v in val.items()},
        'selected_test_nll_pt': float(sel_nll),
        'selected_test_rmse': float(sel_rmse),
        'oracle_test': oracle,
        'oracle_best_member_nll': min(oracle, key=lambda k: oracle[k]['nll']),
    }


_BASELINE_KEYS = ('m05', 'm15', 'lnu', 'fuglreg_lam1.0', 'bktr', 'specpoly', 'ps')


def ensure_baselines(dataset, A, L, n, y_norm, d_raw, coords_km, feat,
                     seeds, results_dir, day=None):
    """Ensure the baseline + core-family leaderboard result files exist for each
    seed. Runs run_all() (stat/fem/het/lgamma + all classical baselines) ONLY for
    seeds whose result file is missing, and skips seeds already computed. Returns
    the list of per-seed result dicts. This makes the CV runner self-contained:
    one command gives the validation-selected number AND the full
    leaderboard, without a separate baseline sweep."""
    from run_all_methods import run_all
    os.makedirs(results_dir, exist_ok=True)
    if day is not None:
        pth = lambda s: f'{results_dir}/results_caweather_day{day}_seed{s}.json'
        ds_tag = f'caweather_day{day}'
    else:
        pth = lambda s: f'{results_dir}/results_{dataset}_seed{s}.json'
        ds_tag = dataset
    rows = []
    for s in seeds:
        p = pth(s)
        if os.path.exists(p):
            rows.append(json.load(open(p)))
            continue
        print(f'  [leaderboard] baselines missing for seed {s} -> running run_all '
              f'(first time only) ...', flush=True)
        res = run_all(A, L, n, y_norm, d_raw, coords_km, feat, s, ds_tag)
        if day is not None:
            res['day'] = int(day)
        with open(p, 'w') as fh:
            json.dump(res, fh)
        rows.append(res)
    return rows


def full_leaderboard(tag, n, base_rows, cv_nll, cv_rmse):
    """Print baselines + core family + the validation-selected row, sorted by
    NLL/pt (same n_test=30% normalization as the paper)."""
    ntest = max(1, n - int(0.7 * n))
    members = [('stat', 'stat'), ('fem', 'semisup'), ('het', 'het'),
               ('lgamma', 'lgamma')] + [(k, k) for k in _BASELINE_KEYS]
    table = {}
    for label, key in members:
        v = [r[f'nll_{key}'] / ntest for r in base_rows
             if f'nll_{key}' in r and r[f'nll_{key}'] == r[f'nll_{key}']]
        rr = [r[f'rmse_{key}'] for r in base_rows
              if f'rmse_{key}' in r and r[f'rmse_{key}'] == r[f'rmse_{key}']]
        if v:
            cat = 'baseline' if label in _BASELINE_KEYS else 'family'
            table[label] = (float(np.mean(v)),
                            float(np.mean(rr)) if rr else float('nan'), cat)
    table['lgamma_block (CV-selected)'] = (
        float(np.mean(cv_nll)), float(np.mean(cv_rmse)), 'family (CV-selected)')
    print(f'\n===== {tag}: full leaderboard (NLL/pt, n_test={ntest}) =====', flush=True)
    print(f'  {"method":30s} {"NLL/pt":>8} {"RMSE":>8}  category', flush=True)
    for m, (nl, rm, cat) in sorted(table.items(), key=lambda x: x[1][0]):
        print(f'  {m:30s} {nl:8.4f} {rm:8.4f}  {cat}', flush=True)


def _run_group(dataset, A, L, n, y_norm, d_raw, coords_km, feat, seeds, outdir,
               tag, results_dir, do_leaderboard=True, day=None):
    """Run all seeds for one (dataset[, day]) snapshot; write one JSON per seed;
    then (optionally) ensure baselines exist and print the full leaderboard."""
    print(f'[{tag}] n={n}  d_raw in [{d_raw.min():.3f},{d_raw.max():.3f}]  '
          f'seeds={len(seeds)}', flush=True)
    sel_nll, sel_rmse = [], []
    for seed in seeds:
        t0 = time.time()
        res = run_seed(dataset, seed, A, L, n, y_norm, d_raw, coords_km)
        if day is not None:
            res['day'] = int(day)
        gtag = '' if GRID == 'base' else '_full'
        out = f'{outdir}/cv_{tag}{gtag}_seed{seed}.json'
        with open(out, 'w') as fh:
            json.dump(res, fh, indent=2)
        sel_nll.append(res['selected_test_nll_pt'])
        sel_rmse.append(res['selected_test_rmse'])
        print(f'  seed {seed}: selected={res["selected_member"]:16s} '
              f'test NLL/pt={res["selected_test_nll_pt"]:.4f} '
              f'RMSE={res["selected_test_rmse"]:.4f}  ({time.time()-t0:.1f}s)',
              flush=True)
    sn = np.array(sel_nll); sr = np.array(sel_rmse)
    print(f'[{tag}] selection-rule test over {len(sn)} seeds: '
          f'NLL/pt={sn.mean():.4f}±{sn.std()/np.sqrt(max(1,len(sn))):.4f}  '
          f'RMSE={sr.mean():.4f}±{sr.std()/np.sqrt(max(1,len(sr))):.4f}', flush=True)

    if do_leaderboard:
        base_rows = ensure_baselines(dataset, A, L, n, y_norm, d_raw, coords_km,
                                     feat, seeds, results_dir, day=day)
        full_leaderboard(tag, n, base_rows, sel_nll, sel_rmse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True,
                    choices=list(LOADERS) + ['caweather'])
    ap.add_argument('--outdir', default='graph_gp/results_cv')
    ap.add_argument('--seeds', type=int, nargs='*', default=SEEDS)
    ap.add_argument('--day', type=int, default=None,
                    help='caweather day-of-year; if omitted, all 12 monthly snapshots')
    ap.add_argument('--grid', choices=['base', 'full'], default='base',
                    help="'base' selects member+K(+w) with fixed d; "
                         "'full' additionally selects the auxiliary d and the "
                         "ridge lambda (via lgamma_fisher) and includes the "
                         "full-graph piecewise-gamma ablation.")
    ap.add_argument('--members', nargs='*', default=None,
                    help='override candidate list (smoke tests)')
    ap.add_argument('--results_dir', default='graph_gp/results',
                    help='where baseline/family leaderboard result files live; '
                         'missing seeds are computed here via run_all()')
    ap.add_argument('--no_leaderboard', action='store_true',
                    help='skip the auto baseline run + full-leaderboard print')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    global GRID
    GRID = args.grid
    if args.members:
        global candidates_for
        _forced = list(args.members)
        candidates_for = lambda ds: _forced
    do_lb = not args.no_leaderboard

    if args.dataset == 'caweather':
        days = [args.day] if args.day is not None else CAWEATHER_DAYS
        for day in days:
            A, L, n, y_norm, d_raw, coords_km, feat = load_caweather(day=day)
            _run_group('caweather', A, L, n, y_norm, d_raw, coords_km, feat,
                       args.seeds, args.outdir, f'caweather_day{day}',
                       args.results_dir, do_leaderboard=do_lb, day=day)
    else:
        A, L, n, y_norm, d_raw, coords_km, feat = LOADERS[args.dataset]()
        _run_group(args.dataset, A, L, n, y_norm, d_raw, coords_km, feat,
                   args.seeds, args.outdir, args.dataset,
                   args.results_dir, do_leaderboard=do_lb)


if __name__ == '__main__':
    main()

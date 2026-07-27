#!/usr/bin/env python3
"""Reproduce the VCGP leaderboard on any of the five real datasets.

This script is the single entry point for reproducing the main empirical
table of the paper. For one (dataset, seed) pair it runs the entire
family (stat, fem, het, lgamma, lgamma_fisher) plus the outer
baselines (m05, m15, lnu, ps, bktr, fuglstad, fuglreg_lam{0.01..100},
specpoly, dk) and writes a JSON of all metrics. Aggregating over the
locked 20 seeds (1729--1748) produces the per-dataset leaderboard.

The first time a dataset is invoked, any missing raw files are
auto-downloaded into ``graph_gp/data_cache/``. METR-LA, PEMS-BAY, PM2.5
and CA Weather are downloaded directly. LAQN requires the (free) public
API; we ship a one-time fetcher (``graph_gp/fetch_laqn_cache.py``).

Usage
-----
Single seed::

    python release/reproduce.py --dataset metrla --seed 1729

Full 20-seed sweep on one dataset (sequential)::

    python release/reproduce.py --dataset pm25 --all-seeds

Build the aggregated leaderboard once results exist::

    python release/reproduce.py --dataset metrla --aggregate

All five datasets, all 20 seeds, all leaderboards (long; for cluster use)::

    python release/reproduce.py --all-datasets --all-seeds --aggregate

Outputs
-------
* Per-run JSON: ``graph_gp/results/results_<dataset>[_day<DOY>]_seed<seed>.json``
* Aggregated leaderboard: ``release/leaderboards/<dataset>.txt``
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
GRAPH_GP = THIS_DIR / 'graph_gp'
DATA_DIR = GRAPH_GP / 'data_cache'

# LAQN uses a live public API that drifts over time; the frozen paper snapshot
# ships in data_cache/ and is authoritative. Set True only via --allow-laqn-refetch.
ALLOW_LAQN_REFETCH = False
RESULTS_DIR = GRAPH_GP / 'results'
LEADERBOARD_DIR = THIS_DIR / 'leaderboards'

DATASETS = ('metrla', 'pemsbay', 'laqn', 'pm25', 'caweather')
SEEDS = tuple(range(1729, 1749))  # 20 locked seeds
CAWEATHER_DAYS = (14, 45, 73, 104, 134, 165,
                  195, 226, 257, 287, 318, 348)  # 15th of each month


# ----------------------------------------------------------------------
# Download specifications
# ----------------------------------------------------------------------
DOWNLOADS = {
    'metrla': {
        'dir': 'metr_la',
        'files': [
            ('https://github.com/liyaguang/DCRNN/raw/refs/heads/master/'
             'data/sensor_graph/adj_mx.pkl', 'adj_mx.pkl'),
            ('https://github.com/liyaguang/DCRNN/raw/refs/heads/master/'
             'data/sensor_graph/graph_sensor_locations.csv',
             'graph_sensor_locations.csv'),
            ('https://github.com/liyaguang/DCRNN/raw/refs/heads/master/'
             'data/sensor_graph/graph_sensor_ids.txt',
             'graph_sensor_ids.txt'),
            # The original DCRNN/Zenodo HDF5 release has been pulled. We
            # download the CSV mirror and convert it to the same HDF5
            # layout downstream loaders expect (see _ensure_metrla_h5).
            ('https://zenodo.org/api/records/5724362/files/'
             'METR-LA.csv/content', 'metr-la.csv'),
        ],
        'post': 'metrla_h5',
    },
    'pemsbay': {
        'dir': 'pems_bay',
        'files': [
            ('https://github.com/liyaguang/DCRNN/raw/refs/heads/master/'
             'data/sensor_graph/adj_mx_bay.pkl', 'adj_mx_bay.pkl'),
            ('https://github.com/liyaguang/DCRNN/raw/refs/heads/master/'
             'data/sensor_graph/graph_sensor_locations_bay.csv',
             'graph_sensor_locations_bay.csv'),
            ('https://zenodo.org/api/records/4263971/files/'
             'pems-bay.h5/content', 'pems-bay.h5'),
        ],
    },
    'caweather': {
        'dir': 'weather',
        'files': [
            ('https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt',
             'ghcnd-stations.txt'),
            ('https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2022.csv.gz',
             '2022.csv.gz'),
            ('https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_year/2023.csv.gz',
             '2023.csv.gz'),
        ],
    },
    'pm25': {
        'dir': 'pm25',
        # EPA AQS daily file is a zip; reproduce.py unpacks it after download.
        'files': [
            ('https://aqs.epa.gov/aqsweb/airdata/daily_88101_2023.zip',
             'daily_88101_2023.zip'),
        ],
        'unzip_to': 'daily_88101_2023',
    },
    'laqn': {
        # LAQN requires the public Kings College London Air Quality API.
        # We provide a one-shot fetcher; reproduce.py shells out to it.
        'dir': '.',
        'files': [],
        'fetcher': 'graph_gp/fetch_laqn_cache.py',
    },
}


# ----------------------------------------------------------------------
# Download helpers
# ----------------------------------------------------------------------
def _download_one(url: str, dest: Path) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        print(f'  [exists] {dest.name}  ({dest.stat().st_size} bytes)')
        return True
    print(f'  [fetching] {url}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except Exception as e:
        print(f'  [FAILED] {dest.name}: {e}')
        return False
    print(f'  [done]  {dest.name}  ({dest.stat().st_size} bytes)')
    return True


def _ensure_pm25_unzip(spec: dict, target_dir: Path) -> None:
    """Unpack the EPA AQS daily zip to the path the loader expects.

    The loader reads ``pm25/daily_88101_2023/daily_88101_2023.csv``. The EPA
    zip may store the CSV at its root (``daily_88101_2023.csv``) or inside a
    folder, so we extract, then locate the CSV wherever it landed and move it
    to the expected path. Robust to either layout.
    """
    import zipfile, shutil
    zip_name = spec['files'][0][1]
    csv_target = target_dir / spec['unzip_to'] / 'daily_88101_2023.csv'
    if csv_target.exists():
        return
    z = target_dir / zip_name
    if not z.exists():
        return
    print(f'  [unzipping] {z.name} -> {csv_target.relative_to(target_dir.parent)}')
    with zipfile.ZipFile(z) as zf:
        zf.extractall(target_dir)
    if csv_target.exists():
        return
    # CSV did not land at the expected path (zip had it at root or elsewhere):
    # find it and move it into place.
    hits = list(target_dir.rglob('daily_88101_2023.csv'))
    if not hits:
        raise FileNotFoundError(
            f'daily_88101_2023.csv not found after unzipping {z} under {target_dir}')
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(hits[0]), str(csv_target))


def _ensure_metrla_h5(target_dir: Path) -> None:
    """Convert the Zenodo CSV mirror to the legacy DCRNN HDF5 layout."""
    h5_path = target_dir / 'metr-la.h5'
    if h5_path.exists() and h5_path.stat().st_size > 0:
        return
    csv_path = target_dir / 'metr-la.csv'
    if not csv_path.exists():
        return
    print(f'  [converting] {csv_path.name} -> {h5_path.name}')
    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    df.to_hdf(h5_path, key='df', mode='w', format='fixed')


def ensure_data(dataset: str) -> None:
    spec = DOWNLOADS[dataset]
    target = DATA_DIR / spec['dir']
    target.mkdir(parents=True, exist_ok=True)
    print(f'-- {dataset}: {target} --')

    if dataset == 'laqn':
        cache_sites = DATA_DIR / 'laqn_sites.json'
        cache_data = DATA_DIR / 'laqn_NO2_2024-06-15.json'
        if cache_sites.exists() and cache_data.exists():
            print('  [exists] laqn_sites.json, laqn_NO2_2024-06-15.json '
                  '(frozen 52-node paper snapshot)')
            return
        # The frozen cache ships with the package and is authoritative: the
        # LAQN NO2 endpoint is a live API whose station set and values drift
        # over time, so a re-fetch does NOT reproduce the paper. We never fetch
        # silently -- restore the shipped cache, or opt in explicitly.
        if not ALLOW_LAQN_REFETCH:
            raise SystemExit(
                '  [ERROR] LAQN cache missing from graph_gp/data_cache/. The '
                'frozen 52-node paper snapshot ships with the package -- '
                'restore laqn_sites.json and laqn_NO2_2024-06-15.json. To '
                'download the same frozen snapshot from the pinned remote '
                'mirror instead, pass --allow-laqn-refetch. (Do NOT rebuild '
                'from the live API -- it drifts; use fetch_laqn_cache.py --live '
                'only if you deliberately want a fresh, non-reproducing snapshot.)')
        print('  [fetching] frozen LAQN snapshot from the pinned mirror '
              '(--allow-laqn-refetch set)')
        # fetch_laqn_cache.py (no --live) downloads the frozen files and writes
        # to ./data_cache/, so we run it from graph_gp/ so they land in
        # graph_gp/data_cache/.
        rc = subprocess.call(
            [sys.executable, 'fetch_laqn_cache.py'],
            cwd=str(GRAPH_GP),
        )
        if rc != 0:
            print('  [WARN] LAQN mirror download failed; check network access, '
                  'or restore the shipped graph_gp/data_cache/ files.')
        return

    ok = True
    for url, name in spec['files']:
        ok = _download_one(url, target / name) and ok

    if dataset == 'pm25' and ok:
        _ensure_pm25_unzip(spec, target)
    if dataset == 'metrla' and ok:
        _ensure_metrla_h5(target)


# ----------------------------------------------------------------------
# Run a single (dataset, seed) via the existing run_all_methods.py
# ----------------------------------------------------------------------
def run_one(dataset: str, seed: int, day: int | None = None,
            extra: list[str] | None = None) -> int:
    """Invoke graph_gp/run_all_methods.py for one (dataset, seed[, day])."""
    cmd = [sys.executable, '-u', str(GRAPH_GP / 'run_all_methods.py'),
           '--dataset', dataset, '--seed', str(seed)]
    if dataset == 'caweather':
        cmd += ['--day', str(day if day is not None else 195)]
    if extra:
        cmd += list(extra)
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def run_lgamma_block(dataset: str, seed: int, K: int,
                     day: int | None = None) -> int:
    """Coordinate-only block variant (run_all_methods --n_clusters). NOTE: this
    clusters on coordinates only; the paper's headline block_aug leader uses the
    d-augmented clustering in run_augmented() below."""
    cmd = [sys.executable, '-u', str(GRAPH_GP / 'run_all_methods.py'),
           '--dataset', dataset, '--seed', str(seed),
           '--n_clusters', str(K)]
    if dataset == 'caweather':
        cmd += ['--day', str(day if day is not None else 195)]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


# Headline block_aug leader per dataset (augmented K-means on [coords, w*d]).
AUG_LEADER = {'pm25': (20, 1.0), 'caweather': (10, 1.0)}


def run_augmented(dataset: str, seed: int, K: int, w: float,
                  day: int | None = None) -> int:
    """The paper's block_aug leader: augmented K-means on [coords, w*d_std] with
    a per-cluster gamma. PM2.5 -> run_pm25_augmented.py; CA Weather ->
    run_caweather_augmented.py. Writes results_*_aug_*.json read by aggregate()."""
    script = ('run_pm25_augmented.py' if dataset == 'pm25'
              else 'run_caweather_augmented.py')
    cmd = [sys.executable, '-u', str(GRAPH_GP / script),
           '--K', str(K), '--w', str(w), '--seed', str(seed)]
    if dataset == 'caweather':
        cmd += ['--day', str(day if day is not None else 195)]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


def run_fisher(dataset: str, seed: int, nu: float,
               day: int | None = None) -> int:
    """Fisher-ridge member (lgamma_fisher). Writes fisher_ridge[_nu<nu>]_*.json,
    which the aggregator reads. Runs both nu=1/2 and nu=3/2 for the Table 4
    default and the METR-LA RMSE leader."""
    cmd = [sys.executable, '-u', str(GRAPH_GP / 'run_fisher_ridge.py'),
           '--dataset', dataset, '--seed', str(seed), '--nu', str(nu)]
    if dataset == 'caweather':
        cmd += ['--day', str(day if day is not None else 195)]
    return subprocess.call(cmd, cwd=str(REPO_ROOT))


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
METHODS_OUTER = ('m05', 'm15', 'lnu', 'ps', 'bktr', 'specpoly',
                 'fuglstad',
                 'fuglreg_lam0.01', 'fuglreg_lam0.1', 'fuglreg_lam1.0',
                 'fuglreg_lam10.0', 'fuglreg_lam100.0', 'dk')
# `fem` is stored under the legacy key `semisup` in the JSON outputs.
METHODS_FAMILY = ('stat', 'het', 'semisup', 'lgamma')
FAMILY_DISPLAY = {'semisup': 'fem', 'bktr': 'se_kriging'}


def _ntest(rec: dict) -> int:
    """Held-out count for per-point NLL. run_all/fisher store the total NLL and
    the node count `n`; the split is 70/30, so n_test = n - int(0.7 n)."""
    if rec.get('n_test'):
        return int(rec['n_test'])
    n = int(rec.get('n', 1))
    return max(1, n - int(0.7 * n))


def _collect(dataset: str) -> list[dict]:
    rows: list[dict] = []
    if dataset == 'caweather':
        for day in CAWEATHER_DAYS:
            for seed in SEEDS:
                p = RESULTS_DIR / f'results_caweather_day{day}_seed{seed}.json'
                if p.exists():
                    d = json.load(open(p))
                    d['_seed'] = seed
                    d['_day'] = day
                    rows.append(d)
    else:
        for seed in SEEDS:
            p = RESULTS_DIR / f'results_{dataset}_seed{seed}.json'
            if p.exists():
                d = json.load(open(p))
                d['_seed'] = seed
                rows.append(d)
    return rows


def _stats(values: list[float]) -> tuple[float, float, int]:
    import math
    n = len(values)
    if n == 0:
        return float('nan'), float('nan'), 0
    mean = sum(values) / n
    if n == 1:
        return mean, float('nan'), 1
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return mean, se, n


def aggregate(dataset: str) -> str:
    rows = _collect(dataset)
    if not rows:
        return f'-- no results found for {dataset} --'

    methods = list(METHODS_FAMILY) + list(METHODS_OUTER)
    header = [f'{"="*78}',
              f'{dataset}  ({len(rows)} runs collected)',
              f'{"="*78}',
              f'  {"method":<26s}  {"NLL/pt mean":>14s}  '
              f'{"SE":>8s}  {"RMSE":>7s}  {"n":>4s}']
    # Accumulate (display, nll_mean, nll_se, rmse_mean, n) and sort by NLL/pt
    # ascending at the end, so the print is an actual leaderboard.
    data = []
    for m in methods:
        nll_vals = [r[f'nll_{m}'] / _ntest(r)
                    for r in rows if f'nll_{m}' in r]
        rmse_vals = [r[f'rmse_{m}'] for r in rows if f'rmse_{m}' in r]
        nll_mean, nll_se, n_nll = _stats(nll_vals)
        rmse_mean, _, _ = _stats(rmse_vals)
        display = FAMILY_DISPLAY.get(m, m)
        data.append((display, nll_mean, nll_se, rmse_mean, n_nll))

    # Fisher-ridge members live in separate JSONs (fisher_ridge[_nu1_5]_*.json),
    # each holding a vanilla and a Fisher-ridge fit. Surface all three family
    # members the paper reports: lgamma_fisher (nu=1/2 Fisher), lgamma_m15
    # (nu=3/2 vanilla, K = s Q^{-2}) and lgamma_fisher_m15 (nu=3/2 Fisher).
    fisher_rows = [('lgamma_fisher', 0.5, 'fisher'),
                   ('lgamma_m15', 1.5, 'vanilla'),
                   ('lgamma_fisher_m15', 1.5, 'fisher')]
    for label, nu, key in fisher_rows:
        if dataset == 'caweather':
            continue  # CA Weather fisher uses per-day files; skip here.
        tag = '' if abs(nu - 0.5) < 1e-8 else '_nu' + f'{nu:g}'.replace('.', '_')
        nll_vals, rmse_vals = [], []
        for seed in SEEDS:
            p = RESULTS_DIR / f'fisher_ridge{tag}_{dataset}_seed{seed}.json'
            if p.exists():
                d = json.load(open(p))
                if f'nll_{key}' in d:
                    nll_vals.append(d[f'nll_{key}'] / _ntest(d))
                if f'rmse_{key}' in d:
                    rmse_vals.append(d[f'rmse_{key}'])
        nll_mean, nll_se, n_nll = _stats(nll_vals)
        rmse_mean, _, _ = _stats(rmse_vals)
        data.append((label, nll_mean, nll_se, rmse_mean, n_nll))

    # block_aug leader (d-augmented K-means) -- the PM2.5 / CA Weather headline.
    if dataset in AUG_LEADER:
        K, w = AUG_LEADER[dataset]
        nll_vals, rmse_vals = [], []
        if dataset == 'caweather':
            paths = [RESULTS_DIR / f'results_caweather_aug_day{day}_K{K}_w{w:.1f}_seed{seed}.json'
                     for day in CAWEATHER_DAYS for seed in SEEDS]
        else:
            paths = [RESULTS_DIR / f'results_pm25_aug_K{K}_w{w:.1f}_seed{seed}.json'
                     for seed in SEEDS]
        for p in paths:
            if p.exists():
                d = json.load(open(p))
                nll = d.get('nll_aug')
                if nll is not None and nll == nll:  # skip NaN fallbacks
                    nll_vals.append(nll / _ntest(d))
                    rmse_vals.append(d.get('rmse_aug', float('nan')))
        if nll_vals:
            nll_mean, nll_se, n_nll = _stats(nll_vals)
            rmse_mean, _, _ = _stats(rmse_vals)
            data.append((f'lgamma_block_aug(K{K},w{w:g})',
                         nll_mean, nll_se, rmse_mean, n_nll))

    # Sort into a leaderboard by NLL/pt ascending; rows with no runs (NaN mean)
    # sink to the bottom.
    import math
    data.sort(key=lambda t: (math.isnan(t[1]), t[1]))
    body = [f'  {d:<26s}  {nm:>14.4f}  {se:>8.4f}  {rm:>7.4f}  {n:>4d}'
            for (d, nm, se, rm, n) in data]
    return '\n'.join(header + body)


def write_leaderboard(dataset: str) -> None:
    LEADERBOARD_DIR.mkdir(parents=True, exist_ok=True)
    out = LEADERBOARD_DIR / f'{dataset}.txt'
    txt = aggregate(dataset)
    out.write_text(txt + '\n')
    print(txt)
    print(f'\n[wrote] {out}')


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description='Reproduce the VCGP leaderboard on real datasets.')
    p.add_argument('--dataset', choices=DATASETS,
                   help='one of: ' + ', '.join(DATASETS))
    p.add_argument('--all-datasets', action='store_true',
                   help='run on all five real datasets')
    p.add_argument('--seed', type=int, help='single seed (default 1729)')
    p.add_argument('--all-seeds', action='store_true',
                   help='loop over all 20 locked seeds 1729..1748')
    p.add_argument('--day', type=int, default=None,
                   help='day-of-year for caweather; default loops over 12 months')
    p.add_argument('--n_clusters', type=int, default=1,
                   help='if >1, run only the lgamma_block variant '
                        '(used for PM2.5 K=2 and CA Weather K=10)')
    p.add_argument('--download-only', action='store_true',
                   help='only fetch raw data, no fitting')
    p.add_argument('--aggregate', action='store_true',
                   help='aggregate cached results into a leaderboard')
    p.add_argument('--skip', nargs='*', default=[],
                   help='methods to skip (e.g. --skip dk bktr)')
    p.add_argument('--skip-fisher', action='store_true',
                   help='do not also run the Fisher-ridge member '
                        '(lgamma_fisher); by default it is run for the '
                        'non-caweather datasets so the leaderboard is complete')
    p.add_argument('--skip-augmented', action='store_true',
                   help='do not also run the d-augmented block_aug leader '
                        '(run_*_augmented.py) on PM2.5 / CA Weather; by default '
                        'it is run so the headline block_aug row is reproduced')
    p.add_argument('--allow-laqn-refetch', action='store_true',
                   help='if the shipped frozen cache is absent, download the '
                        'same frozen snapshot from the pinned remote mirror '
                        '(reproduces the paper). Without this flag a missing '
                        'cache is a hard error rather than a silent fetch.')
    args = p.parse_args()
    global ALLOW_LAQN_REFETCH
    ALLOW_LAQN_REFETCH = args.allow_laqn_refetch

    targets = list(DATASETS) if args.all_datasets else (
        [args.dataset] if args.dataset else [])
    if not targets and not args.aggregate:
        p.error('specify --dataset or --all-datasets')

    if args.aggregate and not targets:
        targets = list(DATASETS)

    seeds = list(SEEDS) if args.all_seeds else [args.seed or 1729]

    for ds in targets:
        ensure_data(ds)
        if args.download_only:
            continue
        if args.aggregate and not args.all_seeds and not args.seed:
            write_leaderboard(ds)
            continue

        days_to_run = (CAWEATHER_DAYS if (ds == 'caweather'
                                          and args.day is None)
                       else [args.day])
        for seed in seeds:
            for day in days_to_run:
                if args.n_clusters > 1:
                    run_lgamma_block(ds, seed, args.n_clusters, day=day)
                else:
                    run_one(ds, seed, day=day, extra=(
                        ['--skip'] + args.skip if args.skip else None))
                    # Emit the Fisher-ridge JSONs the aggregator reads, so the
                    # leaderboard (incl. the Table 4 default and METR-LA RMSE
                    # leader) is reproducible end to end. CA Weather fisher uses
                    # per-day files handled separately.
                    if not args.skip_fisher and ds != 'caweather':
                        for nu in (0.5, 1.5):
                            run_fisher(ds, seed, nu, day=day)
                    # The paper's headline leader on PM2.5 / CA Weather is the
                    # d-augmented block (run_*_augmented.py), not the coords-only
                    # block above; run it so reproduce.py regenerates that row too.
                    if not args.skip_augmented and ds in AUG_LEADER:
                        K, w = AUG_LEADER[ds]
                        run_augmented(ds, seed, K, w, day=day)

        if args.aggregate:
            write_leaderboard(ds)

    return 0


if __name__ == '__main__':
    sys.exit(main())

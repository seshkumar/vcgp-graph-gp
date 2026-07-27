"""Aggregate spectral-BIC runs per dataset and print the 20-seed leaderboard
head-to-head against the relevant cached top baseline.

Baselines used for comparison (from memory/family_vs_outer_headline.md):
  METR-LA  : lgamma_fisher nu=3/2  (RMSE top-1 in family-vs-baseline)
  PEMS-BAY : het                   (RMSE top-1; lgamma_fisher nu=1/2 also competitive)
  LAQN     : lgamma                (NLL/RMSE top-1 in family)
  PM2.5    : fuglreg_lam1.0        (outer NLL top-1)
  CA Weath.: per-month paired against fuglreg_lam1.0 / lgamma as per the
             CA Weather aug leaderboard
"""
import os, sys, json, glob
import numpy as np

RESULTS = 'graph_gp/results'


def load_spectral_bic(dataset, seed, day=None):
    tag = dataset + (f'_day{day}' if day is not None else '')
    path = f'{RESULTS}/spectral_bic_{tag}_seed{seed}.json'
    if not os.path.exists(path): return None
    with open(path) as f:
        return json.load(f)


def load_cached_baseline(dataset, seed, day=None):
    """Return cached top-baseline NLL/RMSE for the dataset."""
    tag = dataset + (f'_day{day}' if day is not None else '')
    path = f'{RESULTS}/results_{tag}_seed{seed}.json'
    if not os.path.exists(path): return None
    with open(path) as f:
        d = json.load(f)
    # Pick the relevant baseline per dataset
    if dataset == 'pm25':
        return dict(nll=d.get('nll_fuglreg_lam1.0'),
                    rmse=d.get('rmse_fuglreg_lam1.0'),
                    label='fuglreg_lam1.0')
    if dataset == 'metrla':
        # prefer cached lgamma_fisher m15 result if present, else lgamma
        path2 = f'{RESULTS}/fisher_ridge_nu1_5_metrla_seed{seed}.json'
        if os.path.exists(path2):
            with open(path2) as f:
                d2 = json.load(f)
            return dict(nll=d2.get('nll_fisher'), rmse=d2.get('rmse_fisher'),
                        label='lgamma_fisher_nu3/2')
        return dict(nll=d.get('nll_lgamma'), rmse=d.get('rmse_lgamma'),
                    label='lgamma')
    if dataset == 'pemsbay':
        return dict(nll=d.get('nll_het'), rmse=d.get('rmse_het'),
                    label='het')
    if dataset == 'laqn':
        return dict(nll=d.get('nll_lgamma'), rmse=d.get('rmse_lgamma'),
                    label='lgamma')
    if dataset == 'caweather':
        return dict(nll=d.get('nll_fuglreg_lam1.0'),
                    rmse=d.get('rmse_fuglreg_lam1.0'),
                    label='fuglreg_lam1.0')
    return None


def paired_stats(a, b):
    """Paired stats on (a - b); return mean, se, t."""
    d = np.array(a) - np.array(b)
    n = len(d)
    if n < 2: return d.mean(), float('nan'), float('nan')
    se = d.std(ddof=1) / np.sqrt(n)
    t = d.mean() / se if se > 0 else float('nan')
    return d.mean(), se, t


def table_head():
    return (f'  {"seed":<6}{"K*":>4}{"NLL_spec":>10}{"NLL_base":>10}{"dNLL":>8}'
            f'  {"RMSE_spec":>11}{"RMSE_base":>11}{"dRMSE":>9}')


def aggregate_single(dataset, day=None):
    seeds = range(1729, 1749)
    rows = []
    for s in seeds:
        sb = load_spectral_bic(dataset, s, day=day)
        bl = load_cached_baseline(dataset, s, day=day)
        if sb is None or bl is None or bl['nll'] is None:
            continue
        if sb['K_star'] is None: continue
        sr = sb['K_star_row']
        rows.append(dict(seed=s, K_star=sb['K_star'],
                         nll_spec=sr['nll'], rmse_spec=sr['rmse'],
                         nll_base=bl['nll'], rmse_base=bl['rmse'],
                         bic=sr['bic'], label=bl['label']))
    tag = dataset + (f' day={day}' if day is not None else '')
    if not rows:
        print(f'\n=== {tag}: no results ==='); return None
    print(f'\n=== {tag}   (vs {rows[0]["label"]})   n={len(rows)} seeds ===')
    print(table_head())
    for r in rows:
        dn = r['nll_spec'] - r['nll_base']
        dr = r['rmse_spec'] - r['rmse_base']
        print(f'  {r["seed"]:<6}{r["K_star"]:>4}{r["nll_spec"]:>10.2f}'
              f'{r["nll_base"]:>10.2f}{dn:>+8.2f}  '
              f'{r["rmse_spec"]:>11.3f}{r["rmse_base"]:>11.3f}{dr:>+9.3f}')
    # means
    arr = np.array([[r['nll_spec'], r['rmse_spec'],
                     r['nll_base'], r['rmse_base']] for r in rows])
    dn_mean, dn_se, dn_t = paired_stats(arr[:, 0], arr[:, 2])
    dr_mean, dr_se, dr_t = paired_stats(arr[:, 1], arr[:, 3])
    Ks = [r['K_star'] for r in rows]
    K_mode = max(set(Ks), key=Ks.count)
    print(f'  {"mean":<6}{"":>4}{arr[:,0].mean():>10.2f}{arr[:,2].mean():>10.2f}'
          f'{dn_mean:>+8.2f}  '
          f'{arr[:,1].mean():>11.3f}{arr[:,3].mean():>11.3f}{dr_mean:>+9.3f}')
    print(f'  paired dNLL  mean {dn_mean:+.2f}  se {dn_se:.2f}  t={dn_t:+.2f}')
    print(f'  paired dRMSE mean {dr_mean:+.3f}  se {dr_se:.3f}  t={dr_t:+.2f}')
    print(f'  K* distribution: {sorted({k: Ks.count(k) for k in set(Ks)}.items())}  mode={K_mode}')
    return rows


def main():
    DATASETS = ['metrla', 'pemsbay', 'laqn', 'pm25']
    for ds in DATASETS:
        aggregate_single(ds)
    DAYS = [14, 45, 73, 104, 134, 165, 195, 226, 257, 287, 318, 348]
    for day in DAYS:
        aggregate_single('caweather', day=day)


if __name__ == '__main__':
    main()

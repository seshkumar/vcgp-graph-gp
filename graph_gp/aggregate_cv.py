#!/usr/bin/env python
"""Aggregate the per-seed validation-fold CV results into one summary.

Reads the JSON files written by `run_cv_selection.py --grid full` (one per
seed, e.g. results_cv/cv_pm25_full_seed1729.json) and prints the CV-selected
NLL/pt and RMSE (mean +/- SE, ddof=1 -- the leaderboard convention), the
per-seed selection, and the composition (how often validation picks the
external annual-std auxiliary and the unregularised member, lambda=0).

Usage:
    python graph_gp/aggregate_cv.py [dataset] [--outdir DIR]

    python graph_gp/aggregate_cv.py pm25
    python graph_gp/aggregate_cv.py metrla --outdir graph_gp/results_cv
"""
import argparse
import glob
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', nargs='?', default='pm25')
    ap.add_argument('--outdir', default='graph_gp/results_cv')
    args = ap.parse_args()

    pat = os.path.join(args.outdir, f'cv_{args.dataset}_full_seed*.json')
    files = sorted(glob.glob(pat))
    if not files:
        raise SystemExit(f'No CV files found matching {pat}')

    rows = [json.load(open(f)) for f in files]
    nll = np.array([r['selected_test_nll_pt'] for r in rows])
    rmse = np.array([r['selected_test_rmse'] for r in rows])
    n = len(rows)
    se = lambda v: v.std(ddof=1) / n ** 0.5

    print(f'CV-selected {args.dataset}  ({n} seeds)')
    print(f'  NLL/pt = {nll.mean():.4f} +/- {se(nll):.4f}')
    print(f'  RMSE   = {rmse.mean():.4f} +/- {se(rmse):.4f}')

    print('  per-seed selection:')
    for r in sorted(rows, key=lambda r: r['seed']):
        print(f'    seed {r["seed"]}: {r["selected_member"]:<26s} '
              f'NLL/pt {r["selected_test_nll_pt"]:.4f}')

    # signal-derived auxiliary d is tagged 'spk' (spatial-kNN); anything else
    # uses the external annual-std d. Fisher ridge -> 'fisher' in the name.
    sig_d = [r['seed'] for r in rows if 'spk' in r['selected_member']]
    fisher = [r['seed'] for r in rows if 'fisher' in r['selected_member']]
    print(f'  external annual-std d selected : {n - len(sig_d)}/{n}'
          f'   (signal-derived on {sorted(sig_d)})')
    print(f'  unregularised (lambda=0)       : {n - len(fisher)}/{n}'
          f'   (Fisher ridge on {sorted(fisher)})')


if __name__ == '__main__':
    main()

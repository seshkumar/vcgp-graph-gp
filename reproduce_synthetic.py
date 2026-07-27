#!/usr/bin/env python3
"""Reproduce the two synthetic experiments.

* ``identifiability``  -- a 20x20 grid graph with a known auxiliary field;
  for each gamma_true in {-1, 0, 1, 3} (paper convention) we draw 10
  spatial snapshots, refit gamma by joint MLL, and report
  recovered gamma_hat. Produces ``identifiability.pdf``.

* ``two_regime``       -- a stochastic-block-model graph with two
  communities, one with gamma_true = -1 (propagation) and one with
  gamma_true = +2 (isolation). Evaluates stat / m05 / m15 / het /
  scalar-lgamma / lgamma_block (K=2) over 20 seeds. Produces
  ``synthetic_two_regime.pdf``.

Both experiments are self-contained and need no external data.

Usage
-----
::

    python release/reproduce_synthetic.py --experiment identifiability
    python release/reproduce_synthetic.py --experiment two_regime
    python release/reproduce_synthetic.py --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR
GRAPH_GP = THIS_DIR / 'graph_gp'

EXPERIMENTS = {
    'identifiability': {
        'run': GRAPH_GP / 'identifiability_experiment.py',
        'plot': None,  # plotter is built in
        'output': 'docs/nonstationary_graph_gp/neurips2026/figs/identifiability.pdf',
    },
    'two_regime': {
        'run': GRAPH_GP / 'synthetic_two_regime.py',
        'plot': GRAPH_GP / 'plot_synthetic_two_regime.py',
        'output': 'docs/nonstationary_graph_gp/neurips2026/figs/synthetic_two_regime.pdf',
    },
}


def run_experiment(name: str) -> int:
    spec = EXPERIMENTS[name]
    print(f'\n--- {name} ---')
    print(f'    run:  {spec["run"]}')
    rc = subprocess.call([sys.executable, '-u', str(spec['run'])],
                         cwd=str(REPO_ROOT))
    if rc != 0:
        return rc
    if spec['plot'] is not None:
        print(f'    plot: {spec["plot"]}')
        rc = subprocess.call([sys.executable, '-u', str(spec['plot'])],
                             cwd=str(REPO_ROOT))
    print(f'    figure -> {spec["output"]}')
    return rc


def main() -> int:
    p = argparse.ArgumentParser(
        description='Reproduce the two synthetic experiments.')
    p.add_argument('--experiment', choices=list(EXPERIMENTS),
                   help='one of: ' + ', '.join(EXPERIMENTS))
    p.add_argument('--all', action='store_true',
                   help='run both experiments')
    args = p.parse_args()

    targets = list(EXPERIMENTS) if args.all else (
        [args.experiment] if args.experiment else [])
    if not targets:
        p.error('specify --experiment or --all')

    for name in targets:
        rc = run_experiment(name)
        if rc != 0:
            return rc
    return 0


if __name__ == '__main__':
    sys.exit(main())

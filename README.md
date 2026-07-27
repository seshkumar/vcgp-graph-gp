# Reproducibility kit — Variance-Coupled Graph Gaussian Processes

This directory is a self-contained package: the driver scripts plus the
bundled `graph_gp/` runner library reproduce the tables and figures in the
paper without referencing any code outside this directory. All randomness is
controlled by 20 locked seeds (`1729`–`1748`). Outputs land under the bundled
`graph_gp/data_cache/` and `graph_gp/results/`.

This is the corrected package. Relative to the earlier release it (a) uses the
leakage-free auxiliary loaders, (b) ships every runner needed for the headline
members (Fisher-ridge, block-augmented, spectral-BIC), (c) ships the validation-
fold model-selection scripts, (d) makes `reproduce.py` emit the Fisher-ridge
JSONs it aggregates, and (e) corrects two baseline labels (see Notes).

---

## 1. Package contents

Drivers:

| Script | Purpose |
|---|---|
| `reproduce.py` | real-dataset leaderboards (family + baselines + Fisher-ridge) |
| `reproduce_synthetic.py` | the two synthetic experiments (identifiability + SBM) |

Runner library under `graph_gp/`:

| File | Produces |
|---|---|
| `run_all_methods.py` | core family (`stat`, `fem`, `het`, `lgamma`) + outer baselines |
| `run_fisher_ridge.py` | `lgamma_fisher` (Fisher-ridge; `--nu 0.5` and `--nu 1.5`) |
| `run_pm25_augmented.py` | PM2.5 `lgamma_block_aug` over the augmented-K-means (K, w) grid |
| `run_caweather_augmented.py` | CA Weather `lgamma_block_aug` |
| `run_spectral_bic.py`, `aggregate_spectral_bic.py` | spectral-clustering + BIC-selected block model |
| `run_cv_selection.py` | validation-fold model selection over `(member, K, d, λ)` (`--grid full`) |
| `run_dataset_di_sweep.py` | auxiliary-`d` ablation (varies `d` over leakage-free choices) |
| `run_deep_ensemble_gnn.py` | Deep Ensemble GNN baseline (M = 5 GCNs, mean/variance heads); needs `torch_geometric` |
| `deep_kernel_baseline.py` | the `dk` deep-kernel-GP baseline (computed; not a headline row) |
| `paciorek_schervish_baseline.py` | the `ps` per-node non-stationary GP baseline |
| `baselines.py` | shared GP inference and stationary/Matern kernels |
| `synthetic_*.py`, `identifiability_experiment.py` | synthetic-experiment code |

Each script runs on a single CPU; the full sweeps parallelise trivially over
`(dataset, seed)` on any scheduler.

---

## 2. Setup

```bash
conda create -n graphgp python=3.11 -y
conda activate graphgp
pip install -r requirements.txt
```

CPU is sufficient. `geometric_kernels` is used by the graph-Matern baselines.
`pyBKTR` is optional and only needed for `run_bktr_baseline.py`.

**LAQN data.** The LAQN NO2 endpoint is a live public API whose station set and
values change over time, so a fresh query does not reproduce the paper. We
therefore ship the frozen 52-node snapshot used in the paper under
`graph_gp/data_cache/` (`laqn_sites.json`, `laqn_NO2_2024-06-15.json`), and
`reproduce.py` reads it directly. If the cache is ever missing, `reproduce.py`
stops with an error rather than silently querying the live API; pass
`--allow-laqn-refetch` to download the *same frozen snapshot* from the pinned
remote mirror (`fetch_laqn_cache.py`). Only `fetch_laqn_cache.py --live`
rebuilds from the live API, which drifts and will not reproduce the reported
numbers. For a permanent public release, host the two cache files on an archive
with a DOI and point `REMOTE_CACHE_BASE` in `fetch_laqn_cache.py` at it.

---

## 3. Real-dataset leaderboards

### One seed, one dataset (smoke test)

```bash
python reproduce.py --dataset metrla --seed 1729
```

On first run this fetches any missing data into `graph_gp/data_cache/`, then
writes `results_metrla_seed1729.json` and the Fisher-ridge JSONs
`fisher_ridge_metrla_seed1729.json` and `fisher_ridge_nu1_5_metrla_seed1729.json`.
(The Fisher-ridge member is run automatically for the non-CA-Weather datasets;
pass `--skip-fisher` to disable it.)

### Full 20-seed sweep and leaderboard

```bash
python reproduce.py --dataset pm25 --all-seeds
python reproduce.py --dataset pm25 --aggregate     # writes leaderboards/pm25.txt
```

### All five datasets

```bash
python reproduce.py --all-datasets --all-seeds --aggregate
```

Single-CPU wall time is on the order of hours per dataset; on a cluster launch
one job per (dataset, seed) and aggregate afterwards.

### Per-dataset leaders

Every dataset is scored against the same standard set — the VCGP family
(`stat`, `fem`, `het`, `lgamma`, `lgamma_fisher` at nu = 1/2 and nu = 3/2) and
the outer baselines (`m05`, `m15`, `lnu`, `ps`, `bktr`, `fuglstad`,
`fuglreg_lam{0.01,0.1,1,10,100}`, `specpoly`, `dk`). The leader differs by
dataset:

| Dataset | Leader | How it is produced |
|---|---|---|
| METR-LA | `lgamma_fisher` (nu = 3/2) | default run (Fisher-ridge is run automatically) |
| PEMS-BAY | `het` | default run |
| LAQN | `lgamma` | default run |
| PM2.5 | `lgamma_block_aug` (K = 20, w = 1.0) | default run (`run_pm25_augmented.py` is run automatically) |
| CA Weather | `lgamma_block_aug` (K = 10, w = 1.0) | default run (`run_caweather_augmented.py` is run automatically) |

The block-augmented leaders use augmented K-means on `[lat, lon, w * d_std]` to
form the clusters (`run_pm25_augmented.py` / `run_caweather_augmented.py`).
`reproduce.py` runs the headline `(K, w)` for PM2.5 and CA Weather automatically
(pass `--skip-augmented` to disable), and the `lgamma_block_aug` row appears in
the aggregated leaderboard. To sweep other `(K, w)`, call the runner directly with
`--K <K> --w <w>` over K in {2,5,10,20}, w in {0.5,1.0}. Selecting `K`/`w` post
hoc on the test set overstates performance; the honest, validation-selected
numbers are produced by `run_cv_selection.py` (next section).

### Skipping expensive baselines

```bash
python reproduce.py --dataset pm25 --all-seeds --skip dk bktr --skip-fisher --skip-augmented
```

---

## 4. Validation-fold model selection (honest, non-post-hoc)

`run_cv_selection.py` reports the numbers under a single fixed selection rule:
for each seed it splits the nodes into train / validation / test, selects the
configuration on the validation fold, and reports only that configuration on the
held-out test set. The test set never enters selection or fitting. It also runs
the baselines if their result files are absent and prints the full leaderboard
with the selected row.

Two grids are available via `--grid`:

- `--grid base` (default): selects the family member, the block `K`, and (for
  PM2.5/CA Weather) the augmentation weight `w`, with a single fixed auxiliary `d`.
- `--grid full`: additionally selects the auxiliary `d`
  (`{default, spatial_k8, spatial_k12}`, all leakage-free) and the ridge `λ`
  (via `lgamma_fisher`), and includes the full-graph piecewise-γ ablation. This is
  the `(member, K, d, λ)` selection rule the paper's CV numbers use.

```bash
python graph_gp/run_cv_selection.py --dataset pm25 --grid full       # 20 seeds; prints leaderboard
python graph_gp/run_cv_selection.py --dataset metrla --grid full
python graph_gp/run_cv_selection.py --dataset caweather --grid full  # loops the 12 months
```

Full-grid outputs are tagged `cv_<dataset>[_day<DOY>]_full_seed<seed>.json` so they
never overwrite base-grid results.

Note on data size: a three-way split needs enough nodes. It is informative on
PM2.5 (n ≈ 1000) and CA Weather (12 monthly snapshots); on the small
single-snapshot datasets (LAQN, n = 52; METR-LA, single snapshot) validation
selection is statistically unstable, and the post-hoc leaderboard together with
the fixed member `het` are the appropriate summaries there.

---

## 5. Synthetic experiments

```bash
python reproduce_synthetic.py --experiment identifiability
python reproduce_synthetic.py --experiment two_regime
python reproduce_synthetic.py --all
```

Both are self-contained (no external data).

---

## 6. Datasets and the auxiliary signal `d_i`

All five real datasets are public benchmarks; the first run auto-downloads the
raw files into `graph_gp/data_cache/<name>/`.

The variance field `d_i` is data-derived (not learned from the regression
likelihood) and, importantly, is computed from a window **disjoint** from the
prediction target, so it does not leak target information:

| Dataset | Auxiliary `d_i` | Window relative to the target |
|---|---|---|
| METR-LA | temporal coefficient of variation of speeds | readings after the target day (disjoint) |
| PEMS-BAY | temporal coefficient of variation of speeds | readings after the target day (disjoint) |
| LAQN | five-level site-type ordinal | static (no temporal overlap) |
| PM2.5 | annual standard deviation | interleaved odd days; target = mean of even days (disjoint) |
| CA Weather | 2022 daily-mean variance | 2022 (target year is 2023) |

`d_i` is normalised to mean 1 and clipped to `[0.1, 10]` before fitting.

---

## 7. Notes on the baselines (labels)

- **BKTR.** The `bktr` column in `run_all_methods.py` (and hence in the main
  leaderboard for the sensor/air datasets) is an SE-kernel spatial-GP baseline
  fitted by grid search, not the low-rank tensor method of Lei, Labbé & Sun
  (*Bayesian Analysis* 20(3):919–947, 2025). The cited method is provided
  separately as `run_bktr_baseline.py` (requires `pyBKTR`). Read the `bktr`
  leaderboard row as "SE-kernel kriging" (printed as `se_kriging`).
- **Deep Ensemble GNN.** The Appendix E numbers come from
  `run_deep_ensemble_gnn.py` (an ensemble of M = 5 graph convolutional networks
  with mean/variance heads). `deep_kernel_baseline.py` is a different baseline
  (the `dk` deep-kernel GP) that is computed by the pipeline but is not a
  headline row.
- **Sign convention of gamma.** The code implements the precision as
  `Q = D^{-1} + 2 tau D^{+gamma/2} L D^{+gamma/2}`, whereas the paper writes
  `D^{-gamma/2}`. The two differ by the sign of gamma: `gamma_paper = -gamma_code`.
  Reported gamma-hat values in the paper are in the paper convention.

---

## 8. Outputs

```
graph_gp/results/results_<dataset>[_day<DOY>][_k<K>]_seed<seed>.json
graph_gp/results/fisher_ridge[_nu1_5]_<dataset>_seed<seed>.json
graph_gp/results/results_pm25_aug_K<K>_w<w>_seed<seed>.json
graph_gp/results/spectral_bic_<dataset>_seed<seed>.json
graph_gp/results_cv/cv_<dataset>[_day<DOY>]_seed<seed>.json
leaderboards/<dataset>.txt
```

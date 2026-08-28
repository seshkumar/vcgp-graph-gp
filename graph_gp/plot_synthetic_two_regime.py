"""Figure: synthetic two-regime (Prop 1 motivator).

Left panel: NLL per method (bar, 20 seeds, error bars = SE).
Right panel: recovered γ_A, γ_B (20 per-seed points) + true values.
"""
import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 10, 'axes.grid': True,
                             'grid.alpha': 0.3, 'axes.spines.top': False,
                             'axes.spines.right': False})
import numpy as np

D = 'graph_gp/results/synth_two_regime'
OUT = 'graph_gp/results/synth_two_regime/synthetic_two_regime.pdf'
os.makedirs(os.path.dirname(OUT), exist_ok=True)

SIGN = 1.0

rows = json.load(open(f'{D}/all_seeds.json'))
methods = ['stat', 'm05', 'm15', 'het', 'lgamma', 'lgamma_block']
labels = ['stat (γ=0)', 'Matérn-1/2', 'Matérn-3/2', 'het (γ=1)', 'lgamma (scalar)', 'lgamma_block (K=2)']
colors = ['#cccccc', '#aaccee', '#6699dd', '#666666', '#dd6666', '#cc2222']

means = []; ses = []
for m in methods:
    vals = [r[f'nll_{m}'] for r in rows if f'nll_{m}' in r]
    means.append(np.mean(vals))
    ses.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={'width_ratios': [1.4, 1]})

x = np.arange(len(methods))
axL.bar(x, means, yerr=ses, color=colors, edgecolor='black', linewidth=0.5, capsize=3)
axL.set_xticks(x)
axL.set_xticklabels(labels, rotation=25, ha='right')
axL.set_ylabel('test NLL (lower = better)')
axL.set_title('(a) Synthetic two-regime: test NLL across methods (20 seeds)')
axL.axhline(min(means), color='#cc2222', linewidth=0.8, linestyle=':', alpha=0.6)

gA = [SIGN * r['gamma_k'][0] for r in rows if 'gamma_k' in r]
gB = [SIGN * r['gamma_k'][1] for r in rows if 'gamma_k' in r]
gs = [SIGN * r['gamma_scalar'] for r in rows if 'gamma_scalar' in r]

axR.scatter([0]*len(gA), gA, color='#cc2222', alpha=0.55, s=26, label='γ̂_A (sparse block)')
axR.scatter([1]*len(gB), gB, color='#6699dd', alpha=0.55, s=26, label='γ̂_B (dense block)')
axR.scatter([2]*len(gs), gs, color='#777777', alpha=0.55, s=26, label='scalar γ̂ (single regime)')
axR.axhline(-1.0, xmin=0.0, xmax=0.35, color='#cc2222', linewidth=1.2)
axR.axhline(2.0, xmin=0.33, xmax=0.68, color='#6699dd', linewidth=1.2)
axR.text(0.02, -0.90, 'γ_true=-1 (propagation)', fontsize=9, color='#cc2222')
axR.text(1.02, 2.10, 'γ_true=+2 (isolation)', fontsize=9, color='#6699dd')
axR.set_xticks([0, 1, 2])
axR.set_xticklabels(['block γ_A', 'block γ_B', 'scalar γ*'])
axR.set_ylabel('γ̂ (recovered)')
axR.set_title('(b) γ recovery (20 seeds)')
axR.axhline(0, color='black', linewidth=0.4, alpha=0.5)
axR.set_ylim(-2.5, 5.5)
axR.legend(fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig(OUT, bbox_inches='tight')
plt.close()
print(f'wrote {OUT}')

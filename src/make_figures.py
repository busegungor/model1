"""Figures for the Results section, read from the experiment output folders.

Expected layout under --root (default: outputs):
  mechanism/            cost_curves.csv, outcomes_by_Q.csv, planning_views.csv   (mechanism_check.py)
  sequential/ sequential_mu50/ sequential_mu150/   sequential_repetitions.csv   (sequential_learning.py)
  prior_sensitivity/    paired_cost_differences_ci95.csv                        (prior_sensitivity_analysis.py)
  misspecification/     paired_vs_correct_kappa.csv, regret_by_assumed_kappa.csv (misspecification_check.py)
  timing_value/         paired_vs_saa.csv, regret_by_setting.csv                 (timing_value_check.py)
  timing_robustness/    robustness_by_setting.csv                               (timing_robustness_check.py)
Writes PDF and PNG files to --out (default: outputs/figures).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist

# fixed colour per rule (validated categorical palette); markers and line styles as secondary encoding
STYLE = {'SAA': dict(color='#2a78d6', marker='o', ls='-'),
         'Bayesian': dict(color='#eb6834', marker='s', ls='--'),
         'No timing': dict(color='#1baf7a', marker='^', ls='-.'),
         'Ignore returns': dict(color='#eda100', marker='D', ls=':')}
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
plt.rcParams.update({'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.edgecolor': MUTED,
                     'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.6,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'savefig.dpi': 300, 'pdf.fonttype': 42})


def ci(x):
    x = np.asarray(x, float); k = len(x)
    h = t_dist.ppf(.975, k - 1) * x.std(ddof=1) / np.sqrt(k)
    return x.mean(), h


def save(fig, out, name):
    fig.savefig(out / f'{name}.pdf', bbox_inches='tight'); fig.savefig(out / f'{name}.png', bbox_inches='tight')
    plt.close(fig)


def fig_mechanism(root, out):
    """RQ1: true cost curve and operating outcomes; orders of the three planning views."""
    c = pd.read_csv(root / 'mechanism' / 'cost_curves.csv').set_index('Q')
    g = pd.read_csv(root / 'mechanism' / 'outcomes_by_Q.csv').set_index('Q')
    v = pd.read_csv(root / 'mechanism' / 'planning_views.csv').set_index('view').planned_Q
    q0, q1 = 150, 400
    base = c.true_cost.min()
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    cc = c.loc[q0:q1]
    ax[0].plot(cc.index, cc.true_cost - base, color=INK, lw=1.6)
    for lab, key, st in (('$Q^\\star$', 'Full information', dict(color=INK, marker='o')),
                         ('No timing', 'No timing', STYLE['No timing']),
                         ('Ignore returns', 'Ignore returns', STYLE['Ignore returns'])):
        q = int(v[key]); y = c.true_cost[q] - base
        ax[0].plot([q], [y], marker=st['marker'], color=st['color'], ms=7, mec='white', mew=1.2, ls='none', zorder=3)
        off, ha = {'Full information': ((12, 4), 'left'), 'No timing': ((2, 45), 'left'),
                   'Ignore returns': ((-12, -4), 'right')}[key]
        ax[0].annotate(f'{lab}\n$Q={q}$', (q, y), textcoords='offset points', xytext=off, ha=ha,
                       fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkB=5)
                       if key == 'No timing' else None)
    ax[0].set_xlabel('Order quantity $Q$'); ax[0].set_ylabel('$\\ell(Q)-\\ell(Q^\\star)$')
    ax[0].set_title('(a) Expected cost above the optimum', loc='left')
    gg = g.loc[q0:q1]
    for col, lab, st in (('lost_sales', 'Lost sales', dict(color='#2a78d6', ls='-')),
                         ('leftover', 'Leftover stock', dict(color='#eb6834', ls='--')),
                         ('late_resalable', 'Resalable, ready after season', dict(color='#1baf7a', ls='-.'))):
        ax[1].plot(gg.index, gg[col], label=lab, color=st['color'], ls=st['ls'])
    for key in ('Full information', 'No timing'):
        ax[1].axvline(int(v[key]), color=MUTED, lw=0.8, ls=':')
    ax[1].set_xlabel('Order quantity $Q$'); ax[1].set_ylabel('Units per season')
    ax[1].set_title('(b) Outcomes under the true process', loc='left'); ax[1].legend(loc='upper left')
    fig.tight_layout(); save(fig, out, 'fig_mechanism')


def fig_learning(root, out):
    """RQ2: relative regret by season and scale, with 95% CIs over repetitions."""
    dirs = [('sequential', 20), ('sequential_mu50', 50), ('sequential_mu150', 150)]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for a, (d, mu) in zip(ax, dirs):
        r = pd.read_csv(root / d / 'sequential_repetitions.csv')
        r = r[r.t > 1].copy(); r['rr'] = 100 * r.regret / (r.expected_cost - r.regret)
        for m in ('SAA', 'Bayesian', 'No timing'):
            s = r[r.method == m].groupby('t').rr.apply(lambda x: pd.Series(ci(x), index=['m', 'h'])).unstack()
            st = STYLE[m]
            a.plot(s.index, s.m, color=st['color'], ls=st['ls'], marker=st['marker'], ms=3.5, label=m)
            a.fill_between(s.index, np.maximum(s.m - s.h, 1e-3), s.m + s.h, color=st['color'], alpha=.12, lw=0)
        a.set_yscale('log'); a.set_title(f'$\\mu={mu}$', loc='left'); a.set_xlabel('Season $t$')
        a.set_xticks([2, 5, 10, 15, 20])
    ax[0].set_ylabel('Relative regret (%)')
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, out, 'fig_learning')


def fig_prior_misspec(root, out):
    """RQ2: (a) prior choice by number of past seasons; (b) misspecified dispersion."""
    p = pd.read_csv(root / 'prior_sensitivity' / 'paired_cost_differences_ci95.csv')
    names = {'beta_1_1': ('Beta(1,1)', '#eb6834', 's'),
             'hypothetical_mu15_m6': ('Prior mean 15', '#1baf7a', '^'),
             'hypothetical_mu25_m6': ('Prior mean 25', '#eda100', 'D')}
    ns = sorted(p.n_history.unique()); x = np.arange(len(ns))
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15]))
    for k, (key, (lab, col, mk)) in enumerate(names.items()):
        s = p[p.method == key].set_index('n_history').loc[ns]
        xx = x + (k - 1) * 0.18
        ax[0].errorbar(xx, s.mean_cost_difference, yerr=[s.mean_cost_difference - s.ci95_lower,
                       s.ci95_upper - s.mean_cost_difference], fmt=mk, color=col, ms=5, capsize=2,
                       elinewidth=1, label=lab)
    ax[0].axhline(0, color=MUTED, lw=0.8)
    ax[0].set_xticks(x, [str(n) for n in ns]); ax[0].set_xlabel('Past seasons $n$')
    ax[0].set_ylabel('Cost difference to SAA'); ax[0].set_title('(a) Prior of the Bayesian rule', loc='left')
    ax[0].legend(loc='lower right')
    m = pd.read_csv(root / 'misspecification' / 'paired_vs_correct_kappa.csv')
    order = ['baseline', 'ratio_0.43', 'ratio_0.90', 'rA_0.30', 'rA_0.70', 'no_returns']
    labels = ['Baseline (0.66)', 'Ratio 0.43', 'Ratio 0.90', '$r_A=0.30$', '$r_A=0.70$', 'No returns']
    diffs = [('SAA-Bayes_kB12', 'SAA', STYLE['SAA']['color'], 'o'),
             ('Bayes_kB4-Bayes_kB12', '$\\kappa_B=4$', '#eb6834', 's'),
             ('Bayes_kB50-Bayes_kB12', '$\\kappa_B=50$', '#1baf7a', '^')]
    y = np.arange(len(order))[::-1]
    for k, (d, lab, col, mk) in enumerate(diffs):
        s = m[m['diff'] == d].set_index('setting').loc[order]
        yy = y + (1 - k) * 0.22
        ax[1].errorbar(s['mean'], yy, xerr=[s['mean'] - s.ci95_lower, s.ci95_upper - s['mean']], fmt=mk,
                       color=col, ms=5, capsize=2, elinewidth=1, label=lab)
    ax[1].axvline(0, color=MUTED, lw=0.8); ax[1].set_yticks(y, labels); ax[1].grid(axis='y', visible=False)
    ax[1].set_xlabel('Cost difference to Bayes with $\\kappa_B=\\kappa=12$')
    ax[1].set_title('(b) Assumed demand dispersion', loc='left'); ax[1].legend(loc='center right')
    fig.tight_layout(); save(fig, out, 'fig_prior_misspec')


def fig_timing(root, out):
    """RQ3: (a) value of timing vs. late resalable returns; (b) robustness of trained orders."""
    pv = pd.read_csv(root / 'timing_value' / 'paired_vs_saa.csv')
    rs = pd.read_csv(root / 'timing_value' / 'regret_by_setting.csv')
    late = rs[rs.method == 'SAA'][['factor', 'value', 'late_resalable_at_Qstar']]
    d = pv[pv['diff'] == 'No timing-SAA'].merge(late, on=['factor', 'value'])
    lab = {('baseline', '-'): 'Baseline', ('processing_days', '1'): '$L^P=1$', ('processing_days', '3'): '$L^P=3$',
           ('processing_days', '5'): '$L^P=5$', ('return_window_days', '14'): '$\\omega=14$',
           ('season_weeks', '12'): '$W=12$', ('season_weeks', '26'): '$W=26$',
           ('return_attempt_probability', '0.3'): '$r_A=0.30$', ('return_attempt_probability', '0.7'): '$r_A=0.70$',
           ('return_resalable_probability', '0.7'): '$k_C=0.70$'}
    PLACED = {'$L^P=3$': (15.5, 80), 'Baseline': (13.0, 68), '$L^P=1$': (21.0, 12)}
    OFFSETS = {'$r_A=0.70$': ((6, -8), 'left'), '$\\omega=14$': ((-6, -3), 'right'), '$W=26$': ((-6, -3), 'right'),
               '$k_C=0.70$': ((6, -8), 'left')}
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    st = STYLE['No timing']
    ax[0].errorbar(d.late_resalable_at_Qstar, d['mean'], yerr=[d['mean'] - d.ci95_lower, d.ci95_upper - d['mean']],
                   fmt=st['marker'], color=st['color'], ms=5, capsize=2, elinewidth=1)
    for _, r in d.iterrows():
        name = lab.get((r.factor, str(r.value)), f'{r.factor}={r.value}')
        if name in PLACED:
            ax[0].annotate(name, (r.late_resalable_at_Qstar, r['mean']), xytext=PLACED[name], fontsize=7, color=INK,
                           arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.6, shrinkB=4))
            continue
        off, ha = OFFSETS.get(name, ((5, -3), 'left'))
        ax[0].annotate(name, (r.late_resalable_at_Qstar, r['mean']), textcoords='offset points',
                       xytext=off, ha=ha, fontsize=7, color=INK)
    rho = np.corrcoef(d.late_resalable_at_Qstar, d['mean'])[0, 1]
    ax[0].axhline(0, color=MUTED, lw=0.8); ax[0].set_xlim(10, 36)
    ax[0].text(0.03, 0.95, f'correlation {rho:.2f}', transform=ax[0].transAxes, fontsize=8, color=MUTED, va='top')
    ax[0].set_xlabel('Resalable returns ready after the season at $Q^\\star$')
    ax[0].set_ylabel('Excess regret, No timing $-$ SAA'); ax[0].set_title('(a) Value of modeling timing', loc='left')
    r = pd.read_csv(root / 'timing_robustness' / 'robustness_by_setting.csv')
    order = [('baseline', '-'), ('processing_days', '1'), ('processing_days', '3'), ('processing_days', '5'),
             ('dispatch_shift_days', '-7'), ('dispatch_shift_days', '7')]
    names = ['Baseline', '$L^P=1$', '$L^P=3$', '$L^P=5$', 'Dispatch $-7$ d', 'Dispatch $+7$ d']
    r['value'] = r['value'].astype(str)
    y = np.arange(len(order))[::-1]
    for k, m in enumerate(('SAA', 'Bayesian', 'No timing')):
        s = r[r.method == m].set_index(['factor', 'value']).loc[order]
        yy = y + (1 - k) * 0.22; stm = STYLE[m]
        ax[1].errorbar(s.avg_gap, yy, xerr=[s.avg_gap - s.ci95_lower, s.ci95_upper - s.avg_gap], fmt=stm['marker'],
                       color=stm['color'], ms=5, capsize=2, elinewidth=1, label=m)
    ax[1].set_yticks(y, names); ax[1].grid(axis='y', visible=False); ax[1].axvline(0, color=MUTED, lw=0.8)
    ax[1].set_xlabel('Gap to re-optimized order'); ax[1].set_title('(b) Trained orders, timing changed', loc='left')
    ax[1].legend(loc='upper right')
    fig.tight_layout(); save(fig, out, 'fig_timing')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='outputs')
    ap.add_argument('--out', default='outputs/figures')
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for f in (fig_mechanism, fig_learning, fig_prior_misspec, fig_timing):
        f(root, out); print('wrote', f.__name__)

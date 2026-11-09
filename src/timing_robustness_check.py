"""Robustness of FIXED, trained order quantities to a change in return timing.

Each repetition draws n past seasons under the baseline configuration and trains
SAA, Bayesian and 'No timing' once. The trained orders are then evaluated, without
retraining, on common test scenarios whose return timing is changed (processing
time, or a shift of the customer's dispatch day). Only timing changes; demand,
return indicators and physical condition are those of the baseline scenarios.
The gap is measured against an order re-optimized for the changed setting.
Repetitions give paired 95% CIs. Complements timing_value_check.py, where every
rule is retrained under each setting.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from config import ExperimentConfig, Economics
from data_generation import generate_history, sample_true_scenario
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from benchmark_solvers import solve_no_timing
from solver_core import optimize
from evaluation import evaluate
from numerical_experiments import _shift_scenario

RULES = [('SAA', solve_saa), ('Bayesian', solve_bayesian), ('No timing', solve_no_timing)]


def settings(cfg):
    """(factor, value, processing_days, dispatch_shift); baseline = processing at cfg value, shift 0."""
    out = [('baseline', '-', cfg.processing_days, 0)]
    out += [('processing_days', v, v, 0) for v in (1, 3, 5) if v != cfg.processing_days]
    out += [('dispatch_shift_days', v, cfg.processing_days, v) for v in (-7, 7)]
    return out


def ci95(x):
    x = np.asarray(x, float); k = len(x)
    h = t_dist.ppf(.975, k - 1) * x.std(ddof=1) / np.sqrt(k)
    return x.mean(), x.mean() - h, x.mean() + h


def run(n_history=20, repetitions=30, seed=20261108, mean_weekly_units=20., scenarios=60,
        test_scenarios=1000, reference_scenarios=1000, output_dir='outputs/timing_robustness'):
    cfg = ExperimentConfig(mean_weekly_units=float(mean_weekly_units),
                           saa_scenarios=scenarios, bayes_scenarios=scenarios)
    ec = Economics()
    # trained orders: one per rule and repetition, all from baseline histories
    orders = {m: [] for m, _ in RULES}
    for rep in range(repetitions):
        hist = generate_history(n_history, np.random.default_rng(np.random.SeedSequence([seed, rep, 2])), cfg, ec)
        for i, (m, solve) in enumerate(RULES):
            orders[m].append(solve(hist, np.random.default_rng(np.random.SeedSequence([seed, rep, 3, i])), cfg, ec).q)
    g_ref, g_test = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 1]).spawn(2)]
    ref_pool = [sample_true_scenario(g_ref, cfg) for _ in range(reference_scenarios)]
    test_pool = [sample_true_scenario(g_test, cfg) for _ in range(test_scenarios)]
    rows, paired = [], []
    for factor, value, proc, shift in settings(cfg):
        ref_sc = [_shift_scenario(s, proc, shift, cfg.return_window_days) for s in ref_pool]
        test_sc = [_shift_scenario(s, proc, shift, cfg.return_window_days) for s in test_pool]
        ref = optimize('ref', ref_sc, ec)
        ref_cost = evaluate(ref.q, test_sc, ec)['mean_mismatch_cost']
        cache = {}  # orders are integers, so many repetitions share a value
        gap = {}
        for m, _ in RULES:
            g = []
            for q in orders[m]:
                if q not in cache:
                    cache[q] = evaluate(q, test_sc, ec)['mean_mismatch_cost']
                g.append(cache[q] - ref_cost)
            gap[m] = np.array(g)
            mean, lo, hi = ci95(gap[m])
            rows.append(dict(factor=factor, value=value, method=m, n_history=n_history,
                             repetitions=repetitions, reference_Q=ref.q, reference_cost=ref_cost,
                             avg_Q=np.mean(orders[m]), sd_Q=np.std(orders[m], ddof=1),
                             avg_gap=mean, ci95_lower=lo, ci95_upper=hi,
                             gap_pct=100 * mean / ref_cost))
        for m in ('Bayesian', 'No timing'):
            mean, lo, hi = ci95(gap[m] - gap['SAA'])
            paired.append(dict(factor=factor, value=value, diff=f'{m}-SAA', mean=mean,
                               ci95_lower=lo, ci95_upper=hi, mean_pct=100 * mean / ref_cost))
        print(f'completed {factor}={value}')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    dr, dp = pd.DataFrame(rows), pd.DataFrame(paired)
    dr.to_csv(out / 'robustness_by_setting.csv', index=False)
    dp.to_csv(out / 'paired_vs_saa.csv', index=False)
    print(dr[['factor', 'value', 'method', 'reference_Q', 'avg_Q', 'avg_gap', 'ci95_lower', 'ci95_upper']].round(2).to_string(index=False))
    print(dp.round(2).to_string(index=False))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-history', type=int, default=20)
    ap.add_argument('--repetitions', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20261108)
    ap.add_argument('--mean-weekly-units', type=float, default=20.)
    ap.add_argument('--scenarios', type=int, default=60)
    ap.add_argument('--test-scenarios', type=int, default=1000)
    ap.add_argument('--reference-scenarios', type=int, default=1000)
    ap.add_argument('--output-dir', default='outputs/timing_robustness')
    a = ap.parse_args()
    run(a.n_history, a.repetitions, a.seed, a.mean_weekly_units, a.scenarios,
        a.test_scenarios, a.reference_scenarios, a.output_dir)
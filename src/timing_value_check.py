"""Value of modeling return timing across operating conditions (one factor at a time).

For each setting, every rule is trained on n past seasons generated under that
setting and evaluated on a common test set from the same setting. Rules: SAA,
Bayesian, 'No timing' (SAA with all return lead times set to zero) and
'Ignore returns'. Paired differences are reported relative to SAA, so
'No timing - SAA' is the cost of ignoring return timing in that setting.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from config import ExperimentConfig, Economics
from data_generation import generate_history, sample_true_scenario
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from benchmark_solvers import solve_ignore_returns, solve_no_timing
from solver_core import optimize
from evaluation import evaluate

RULES = [('SAA', solve_saa), ('Bayesian', solve_bayesian),
         ('No timing', solve_no_timing), ('Ignore returns', solve_ignore_returns)]

def settings(base):
    """(factor, value, config overrides); the baseline appears once."""
    weeks0 = base.season_days // 7
    out = [('baseline', '-', {})]
    out += [('processing_days', v, {'processing_days': v}) for v in (1, 3, 5)]
    out += [('return_window_days', v, {'return_window_days': v}) for v in (14,)]
    out += [('season_weeks', w, {'season_days': 7 * w, 'mean_weekly_units': base.mean_weekly_units * weeks0 / w})
            for w in (12, 26)]
    out += [('return_attempt_probability', v, {'return_attempt_probability': v}) for v in (0.30, 0.70)]
    out += [('return_resalable_probability', v, {'return_resalable_probability': v}) for v in (0.70,)]
    return out

def ci95(x):
    x = np.asarray(x, float); k = len(x)
    h = t_dist.ppf(.975, k - 1) * x.std(ddof=1) / np.sqrt(k)
    return x.mean(), x.mean() - h, x.mean() + h

def run(n_history=20, repetitions=30, seed=20261107, mean_weekly_units=20., scenarios=60,
        test_scenarios=1000, reference_scenarios=1000, output_dir='outputs/timing_value'):
    base = ExperimentConfig(mean_weekly_units=float(mean_weekly_units),
                            saa_scenarios=scenarios, bayes_scenarios=scenarios)
    ec = Economics()
    rows, paired = [], []
    for si, (factor, value, over) in enumerate(settings(base)):
        cfg = replace(base, **over)
        g_ref, g_test = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, si, 1]).spawn(2)]
        ref = optimize('ref', [sample_true_scenario(g_ref, cfg) for _ in range(reference_scenarios)], ec)
        test = [sample_true_scenario(g_test, cfg) for _ in range(test_scenarios)]
        ref_eval = evaluate(ref.q, test, ec)
        reg = {m: [] for m, _ in RULES}; qs = {m: [] for m, _ in RULES}
        for rep in range(repetitions):
            hist = generate_history(n_history, np.random.default_rng(np.random.SeedSequence([seed, si, rep, 2])), cfg, ec)
            for i, (m, solve) in enumerate(RULES):
                q = solve(hist, np.random.default_rng(np.random.SeedSequence([seed, si, rep, 3, i])), cfg, ec).q
                qs[m].append(q); reg[m].append(evaluate(q, test, ec)['mean_mismatch_cost'] - ref_eval['mean_mismatch_cost'])
        for m, _ in RULES:
            mean, lo, hi = ci95(reg[m])
            rows.append(dict(factor=factor, value=value, method=m, n_history=n_history, reference_Q=ref.q,
                             reference_cost=ref_eval['mean_mismatch_cost'],
                             late_resalable_at_Qstar=ref_eval['mean_late_resalable'],
                             resales_at_Qstar=ref_eval['mean_resales_actual'],
                             avg_Q=np.mean(qs[m]), avg_regret=mean, ci95_lower=lo, ci95_upper=hi,
                             regret_pct=100 * mean / ref_eval['mean_mismatch_cost']))
            if m != 'SAA':
                mean, lo, hi = ci95(np.array(reg[m]) - np.array(reg['SAA']))
                paired.append(dict(factor=factor, value=value, diff=f'{m}-SAA', mean=mean,
                                   ci95_lower=lo, ci95_upper=hi,
                                   mean_pct=100 * mean / ref_eval['mean_mismatch_cost']))
        print(f'completed {factor}={value}')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    dr, dp = pd.DataFrame(rows), pd.DataFrame(paired)
    dr.to_csv(out / 'regret_by_setting.csv', index=False); dp.to_csv(out / 'paired_vs_saa.csv', index=False)
    print(dp.round(2).to_string(index=False))
    print(dr[['factor', 'value', 'method', 'reference_Q', 'avg_Q', 'late_resalable_at_Qstar', 'avg_regret']].round(2).to_string(index=False))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-history', type=int, default=20)
    ap.add_argument('--repetitions', type=int, default=30)
    ap.add_argument('--seed', type=int, default=20261107)
    ap.add_argument('--mean-weekly-units', type=float, default=20.)
    ap.add_argument('--scenarios', type=int, default=60)
    ap.add_argument('--test-scenarios', type=int, default=1000)
    ap.add_argument('--reference-scenarios', type=int, default=1000)
    ap.add_argument('--output-dir', default='outputs/timing_value')
    a = ap.parse_args()
    run(a.n_history, a.repetitions, a.seed, a.mean_weekly_units, a.scenarios,
        a.test_scenarios, a.reference_scenarios, a.output_dir)
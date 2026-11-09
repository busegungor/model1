"""Cost of a misspecified Bayesian dispersion across cost ratios and return rates.

For each setting: (i) the full-information order Q* under true dispersions
kappa in {4, 12, 50}, and (ii) the regret of SAA and of the Bayesian rule with
assumed dispersion kappa_B in {4, 12, 50}, trained on n past seasons
(true kappa = 12). Procurement cost c is varied within the range of
Mostard & Teunter (2006, Table 2): c = 30 (ratio 0.43), 20 (0.66), 9.4 (0.90).
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
from solver_core import optimize
from evaluation import evaluate

SETTINGS = [  # name, procurement cost c, return-attempt probability r_A
    ('baseline',        20.0, 0.53),
    ('ratio_0.43',      30.0, 0.53),
    ('ratio_0.90',       9.4, 0.53),
    ('rA_0.30',         20.0, 0.30),
    ('rA_0.70',         20.0, 0.70),
    ('no_returns',      20.0, 0.00),
]
KAPPAS = (4.0, 12.0, 50.0)

def ci95(x):
    x = np.asarray(x, float); k = len(x)
    h = t_dist.ppf(.975, k - 1) * x.std(ddof=1) / np.sqrt(k)
    return x.mean(), x.mean() - h, x.mean() + h

def run(n_history=20, repetitions=10, seed=20261106, mean_weekly_units=20.,
        scenarios=200, test_scenarios=2000, reference_scenarios=1000,
        output_dir='outputs/misspecification'):
    rows_q, rows_r, rows_p = [], [], []
    for si, (name, c, r_a) in enumerate(SETTINGS):
        ec = replace(Economics(), procurement_cost=c)
        ratio = (ec.price - c) / (ec.price - ec.salvage_value)
        # (i) sensitivity of Q* to the true dispersion
        for kappa in KAPPAS:
            cfg_k = ExperimentConfig(mean_weekly_units=mean_weekly_units, weekly_dispersion=kappa,
                                     return_attempt_probability=r_a)
            g = np.random.default_rng(np.random.SeedSequence([seed, si, int(kappa), 0]))
            q_star = optimize('ref', [sample_true_scenario(g, cfg_k) for _ in range(reference_scenarios)], ec).q
            rows_q.append(dict(setting=name, c=c, critical_ratio=ratio, r_A=r_a, true_kappa=kappa, Q_star=q_star))
        # (ii) regret of SAA and of Bayesian with assumed kappa_B (true kappa = 12)
        cfg = ExperimentConfig(mean_weekly_units=mean_weekly_units, return_attempt_probability=r_a,
                               saa_scenarios=scenarios, bayes_scenarios=scenarios)
        g_ref, g_test = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, si, 1]).spawn(2)]
        ref = optimize('ref', [sample_true_scenario(g_ref, cfg) for _ in range(reference_scenarios)], ec)
        test = [sample_true_scenario(g_test, cfg) for _ in range(test_scenarios)]
        ref_cost = evaluate(ref.q, test, ec)['mean_mismatch_cost']
        reg = {m: [] for m in ['SAA'] + [f'Bayes_kB{int(k)}' for k in KAPPAS]}
        qs = {m: [] for m in reg}
        for rep in range(repetitions):
            hist = generate_history(n_history, np.random.default_rng(np.random.SeedSequence([seed, si, rep, 2])), cfg, ec)
            rng_seq = np.random.SeedSequence([seed, si, rep, 3])
            q = solve_saa(hist, np.random.default_rng(rng_seq), cfg, ec).q
            qs['SAA'].append(q); reg['SAA'].append(evaluate(q, test, ec)['mean_mismatch_cost'] - ref_cost)
            for k in KAPPAS:
                cfg_b = replace(cfg, bayesian_assumed_dispersion=k)
                q = solve_bayesian(hist, np.random.default_rng(rng_seq), cfg_b, ec).q
                m = f'Bayes_kB{int(k)}'
                qs[m].append(q); reg[m].append(evaluate(q, test, ec)['mean_mismatch_cost'] - ref_cost)
        for m in reg:
            mean, lo, hi = ci95(reg[m])
            rows_r.append(dict(setting=name, critical_ratio=ratio, r_A=r_a, method=m, n_history=n_history,
                               reference_Q=ref.q, reference_cost=ref_cost, avg_Q=np.mean(qs[m]),
                               avg_regret=mean, ci95_lower=lo, ci95_upper=hi,
                               regret_pct=100 * mean / ref_cost))
        base = np.array(reg['Bayes_kB12'])
        for m in reg:
            if m != 'Bayes_kB12':
                mean, lo, hi = ci95(np.array(reg[m]) - base)
                rows_p.append(dict(setting=name, diff=f'{m}-Bayes_kB12', mean=mean, ci95_lower=lo, ci95_upper=hi))
        print(f'completed {name}')
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    dq, dr = pd.DataFrame(rows_q), pd.DataFrame(rows_r)
    dq.to_csv(out / 'q_star_by_true_kappa.csv', index=False); dr.to_csv(out / 'regret_by_assumed_kappa.csv', index=False)
    dp = pd.DataFrame(rows_p); dp.to_csv(out / 'paired_vs_correct_kappa.csv', index=False)
    print(dp.round(2).to_string(index=False))
    print(dq.pivot(index='setting', columns='true_kappa', values='Q_star').to_string())
    print(dr.round(2).to_string(index=False))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-history', type=int, default=20)
    ap.add_argument('--repetitions', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20261106)
    ap.add_argument('--mean-weekly-units', type=float, default=20.)
    ap.add_argument('--scenarios', type=int, default=200)
    ap.add_argument('--test-scenarios', type=int, default=2000)
    ap.add_argument('--reference-scenarios', type=int, default=1000)
    ap.add_argument('--output-dir', default='outputs/misspecification')
    a = ap.parse_args()
    run(a.n_history, a.repetitions, a.seed, a.mean_weekly_units, a.scenarios,
        a.test_scenarios, a.reference_scenarios, a.output_dir)
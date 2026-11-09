"""Sequential learning: Q_t is chosen before season t using seasons 1..t-1 only."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from config import ExperimentConfig, Economics
from data_generation import HistoricalSeason, sample_true_scenario
from simulation import simulate
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from solver_core import optimize
from evaluation import evaluate
from scipy.stats import t as t_dist

from benchmark_solvers import solve_ignore_returns, solve_no_timing

SOLVERS = [('SAA', solve_saa), ('Bayesian', solve_bayesian),
           ('Ignore returns', solve_ignore_returns), ('No timing', solve_no_timing)]

def run(T=20, repetitions=20, seed=20261103, mean_weekly_units=20.,
        scenarios=60, test_scenarios=160, reference_scenarios=200,
        output_dir='outputs/sequential', bayes_kappa=12.0, methods=None):
    cfg = ExperimentConfig(mean_weekly_units=float(mean_weekly_units),
                           saa_scenarios=scenarios, bayes_scenarios=scenarios,
                           evaluation_scenarios=test_scenarios,
                           reference_scenarios=reference_scenarios,
                           bayesian_assumed_dispersion=float(bayes_kappa))
    active = [m for m, _ in SOLVERS if methods is None or m in methods]
    ec = Economics()
    g_ref, g_test = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 1]).spawn(2)]
    ref = optimize('reference_MC', [sample_true_scenario(g_ref, cfg) for _ in range(reference_scenarios)], ec)
    test = [sample_true_scenario(g_test, cfg) for _ in range(test_scenarios)]
    ref_cost = evaluate(ref.q, test, ec)['mean_mismatch_cost']
    # Cold start (season 1, no data): same fixed rule as generate_history
    q1 = int(round(cfg.history_stock_fraction * (cfg.season_days // 7) * cfg.mean_weekly_units))
    rows = []
    for rep in range(1, repetitions + 1):
        world = np.random.default_rng(np.random.SeedSequence([seed, rep, 0]))
        hist = {m: [] for m in active}
        cum = {m: 0.0 for m in active}
        for t in range(1, T + 1):
            sc = sample_true_scenario(world, cfg, t)        # common true season for both methods
            for i, (m, solve) in enumerate(SOLVERS):
                if m not in active:            # skipped methods keep the seeds of the others unchanged
                    continue
                if hist[m]:
                    rng = np.random.default_rng(np.random.SeedSequence([seed, rep, t, i + 1]))
                    q = solve(hist[m], rng, cfg, ec).q       # uses seasons 1..t-1 only
                else:
                    q = q1
                out = simulate(q, sc, ec)
                hist[m].append(HistoricalSeason(t, sc.demand_times, out.records, q,
                                                cfg.season_days, sc.order_ids))
                exp_cost = evaluate(q, test, ec)['mean_mismatch_cost']
                if t > 1:                      # season 1 is the common cold start; learning starts at t = 2
                    cum[m] += exp_cost - ref_cost
                rows.append(dict(repetition=rep, t=t, n_past_seasons=t - 1, method=m, Q=q,
                                 reference_Q=ref.q, realized_cost=out.mismatch_cost,
                                 expected_cost=exp_cost, regret=exp_cost - ref_cost,
                                 cumulative_regret=cum[m]))
    out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out_dir / 'sequential_repetitions.csv', index=False)
    summary = df.groupby(['t', 'method'], as_index=False).agg(
        avg_Q=('Q', 'mean'), sd_Q=('Q', 'std'), avg_regret=('regret', 'mean'),
        sd_regret=('regret', 'std'), avg_cumulative_regret=('cumulative_regret', 'mean'))
    summary.to_csv(out_dir / 'sequential_summary.csv', index=False)
    paired = pd.concat([paired_by_season(df, a=m) for m in active if m != 'SAA'], ignore_index=True)
    paired.to_csv(out_dir / 'sequential_paired_ci95.csv', index=False)
    print(paired.round(2).to_string(index=False))

def paired_by_season(df, a='Bayesian', b='SAA'):
    """Per season t: mean and 95% t-interval of (a - b) for regret and cumulative regret,
    paired over repetitions (both methods see the same true seasons and test set)."""
    out = []
    for col in ('regret', 'cumulative_regret'):
        wide = df.pivot(index=['repetition', 't'], columns='method', values=col)
        diff = (wide[a] - wide[b]).groupby(level='t')
        for t, d in diff:
            d = d.to_numpy(dtype=float); k = len(d)
            half = t_dist.ppf(.975, k - 1) * d.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
            out.append(dict(t=int(t), metric=col, diff=f'{a}-{b}', repetitions=k,
                            mean=d.mean(), ci95_lower=d.mean() - half, ci95_upper=d.mean() + half))
    return pd.DataFrame(out).sort_values(['metric', 't'])

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--T',type=int,default=20)
    ap.add_argument('--seed',type=int,default=20261103)
    ap.add_argument('--output-dir',default='outputs/sequential')
    # ap.add_argument('--input-dir')
    ap.add_argument('--repetitions',type=int,default=20)
    ap.add_argument('--mean-weekly-units',type=float,default=20)
    ap.add_argument('--scenarios',type=int,default=60)
    ap.add_argument('--test-scenarios',type=int,default=160)
    ap.add_argument('--reference-scenarios',type=int,default=200)
    ap.add_argument('--bayes-kappa',type=float,default=12.)
    ap.add_argument('--methods',nargs='+',default=None)
    args=ap.parse_args()
    run(T=args.T, repetitions=args.repetitions, seed=args.seed,
        mean_weekly_units=args.mean_weekly_units, scenarios=args.scenarios,
        test_scenarios=args.test_scenarios, reference_scenarios=args.reference_scenarios,
        output_dir=args.output_dir, bayes_kappa=args.bayes_kappa, methods=args.methods)
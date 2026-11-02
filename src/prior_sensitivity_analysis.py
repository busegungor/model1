"""Paired Bayesian-prior and information-availability study for the existing model.

DOES NOT modify the newsvendor equations or the true data-generating process.
The informative priors are *hypothetical design scenarios*; there is no
independent fashion-SKU evidence supporting the illustrative prior means.

Independently generated histories are paired across methods. The SAME fixed
out-of-sample test scenarios are used to evaluate ALL methods/replications.
The confidence intervals describe variability over training histories and
optimization simulation, conditional on this finite common test set.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t as t_distribution

from config import Economics, ExperimentConfig
from data_generation import generate_history, sample_true_scenario
from scenario_generation import bayesian_scenarios
from saa_solver import solve_saa
from solver_core import optimize
from evaluation import evaluate


@dataclass(frozen=True)
class PriorSpec:
    label: str
    alpha: float
    beta: float
    illustrative_weekly_mean: float | None = None
    equivalent_weeks: float | None = None


def make_demand_priors(kappa: float,
                       centers=(15., 25.),
                       equivalent_weeks=6.) -> list[PriorSpec]:
    """Construct hypothetical priors on NB *success probability* phi.

    NB(kappa,phi): E[D|phi]=kappa*(1-phi)/phi.
    Beta(1 + m*kappa, m*mu0) has *prior-predictive mean* mu0, for m>0.
    Here m is a pseudo-week parameter, NOT an independently observed sample.
    Beta(1,1) is uniform on phi, NOT uniform/weak on demand mean.
    """
    if kappa <= 0 or equivalent_weeks <= 0:
        raise ValueError('kappa and equivalent_weeks must be positive')
    priors = [PriorSpec('beta_1_1', 1., 1.)]
    for center in centers:
        if center <= 0:
            raise ValueError('All prior centers must be positive')
        prior_alpha = 1. + equivalent_weeks * kappa
        prior_beta = equivalent_weeks * float(center)
        center_label = format(float(center), '.6g').replace('.', 'p')
        strength_label = format(float(equivalent_weeks), '.6g').replace('.', 'p')
        priors.append(PriorSpec(f'hypothetical_mu{center_label}_m{strength_label}',
                                prior_alpha, prior_beta, float(center),
                                float(equivalent_weeks)))
    return priors


def information_counts(history, cfg) -> dict:
    """Count unique independent seasons, observed weekly demand, and sale units.

    Annual completed seasons imply all recorded fulfilled-sale return outcomes
    have matured by the next annual ordering date (model assumption).
    Demand is fully observed by synthetic design, even for unserved units.
    """
    sold = [record for season in history for record in season.completed_sales]
    attempts = [record for record in sold if record.mark.attempts_return]
    accepted = [record for record in attempts if record.mark.accepted]
    rejected = [record for record in attempts if not record.mark.accepted]
    return {
        'n_history': int(len(history)),
        'n_weekly_demand_observations': int(len(history) * (cfg.season_days // 7)),
        'n_potential_demand_units': int(sum(len(s.demand_times) for s in history)),
        'n_customer_order_events': int(sum(len(set(s.order_ids)) if s.order_ids
                                            else len(s.demand_times) for s in history)),
        'n_fulfilled_sale_units': int(len(sold)),
        'n_return_resolved_sale_units': int(len(sold)),
        'n_return_attempts': int(len(attempts)),
        'n_accepted_return_units': int(len(accepted)),
        'n_rejected_late_attempts': int(len(rejected)),
        'n_resalability_observations': int(len(accepted)),
        'n_physically_resalable_returns': int(sum(record.mark.resalable for record in accepted)),
    }


def paired_intervals(df: pd.DataFrame, reference: str = 'SAA') -> pd.DataFrame:
    """t intervals for method minus SAA over paired independent histories."""
    wide = df.pivot(index=['n_history', 'repetition'], columns='method',
                    values='mean_mismatch_cost')
    output = []
    for n, data in wide.groupby(level='n_history'):
        for method in wide.columns:
            if method == reference:
                continue
            delta = (data[method] - data[reference]).dropna().to_numpy(dtype=float)
            reps = len(delta)
            mean = float(delta.mean()) if reps else float('nan')
            sd = float(delta.std(ddof=1)) if reps > 1 else float('nan')
            halfwidth = (float(t_distribution.ppf(.975, reps - 1) * sd / np.sqrt(reps))
                         if reps > 1 else float('nan'))
            output.append({
                'n_history': int(n), 'method': method, 'reference_method': reference,
                'repetitions': reps, 'mean_cost_difference': mean,
                'sd_paired_difference': sd,
                'ci95_lower': mean - halfwidth, 'ci95_upper': mean + halfwidth,
            })
    return pd.DataFrame(output)


def run(sample_sizes=(1, 2, 5, 10), repetitions=20, seed=20261031,
        output_dir='outputs/prior_sensitivity', mean_weekly_units=20.,
        saa_scenarios=60, bayes_scenarios=60, test_scenarios=160,
        prior_centers=(15., 25.), prior_equivalent_weeks=6.):
    """One fixed true NB DGP across priors, same histories and holdout for all.

    No training/test oracle information is used to construct any prior.
    The two prior centers are user-specified *illustrative counterfactual* inputs.
    """
    if repetitions < 2:
        raise ValueError('At least 2 independent repetitions needed for CI')
    if min(sample_sizes) < 1:
        raise ValueError('Historical season count must be >= 1')
    cfg = ExperimentConfig(mean_weekly_units=float(mean_weekly_units),
                           saa_scenarios=int(saa_scenarios),
                           bayes_scenarios=int(bayes_scenarios),
                           evaluation_scenarios=int(test_scenarios))
    ec = Economics()
    priors = make_demand_priors(cfg.bayesian_assumed_dispersion, prior_centers,
                               prior_equivalent_weeks)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # A single independent, common holdout prevents method-specific test noise.
    test_rng = np.random.default_rng(np.random.SeedSequence([seed, 99991]))
    test = [sample_true_scenario(test_rng, cfg) for _ in range(test_scenarios)]
    rows = []
    counts_rows = []
    for n in sample_sizes:
        for repetition in range(1, repetitions + 1):
            # All methods use the SAME independently generated historical dataset.
            history_rng = np.random.default_rng(np.random.SeedSequence([seed, int(n), repetition, 101]))
            history = generate_history(int(n), history_rng, cfg, ec)
            info = information_counts(history, cfg)
            counts_rows.append({'repetition': repetition, **info})
            saa_rng = np.random.default_rng(np.random.SeedSequence([seed, int(n), repetition, 202]))
            saa = solve_saa(history, saa_rng, cfg, ec)
            for p in [None, *priors]:
                if p is None:
                    method = 'SAA'
                    sol = saa
                    alpha = beta = np.nan
                    center = pseudo = np.nan
                else:
                    method = p.label
                    # Identical seed per prior gives partially coupled Monte Carlo streams.
                    # It does NOT imply exact common random numbers after differing beta draws.
                    rng = np.random.default_rng(np.random.SeedSequence([seed, int(n), repetition, 303]))
                    simulated = bayesian_scenarios(history, cfg.bayes_scenarios, rng, cfg,
                                                   prior_alpha=p.alpha, prior_beta=p.beta)
                    sol = optimize(method, simulated, ec)
                    alpha, beta = p.alpha, p.beta
                    center = p.illustrative_weekly_mean
                    pseudo = p.equivalent_weeks
                metrics = evaluate(sol.q, test, ec)
                rows.append({
                    'n_history': int(n), 'repetition': repetition, 'method': method,
                    'prior_alpha': alpha, 'prior_beta': beta,
                    'hypothetical_prior_weekly_mean': center,
                    'prior_pseudo_weeks': pseudo,
                    **info, **metrics,
                })
            if repetition % 5 == 0 or repetition == repetitions:
                print(f'completed n={n}, repetitions={repetition}/{repetitions}', flush=True)
                pd.DataFrame(rows).to_csv(out / 'prior_sensitivity_repetitions_partial.csv', index=False)

    df = pd.DataFrame(rows)
    observations = pd.DataFrame(counts_rows)
    df.to_csv(out / 'prior_sensitivity_repetitions.csv', index=False)
    observations.to_csv(out / 'information_availability.csv', index=False)
    summary = df.groupby(['n_history', 'method'], as_index=False).agg(
        repetitions=('repetition', 'size'),
        avg_test_cost=('mean_mismatch_cost', 'mean'),
        sd_test_cost=('mean_mismatch_cost', 'std'),
        avg_Q=('order', 'mean'), sd_Q=('order', 'std'),
        avg_fill_rate=('fill_rate', 'mean'),
        avg_actual_resales=('mean_resales_actual', 'mean'),
        avg_n_weekly_demand=('n_weekly_demand_observations', 'mean'),
        avg_n_fulfilled_sale_units=('n_fulfilled_sale_units', 'mean'),
        avg_n_return_resolved_units=('n_return_resolved_sale_units', 'mean'),
        avg_n_accepted_returns=('n_accepted_return_units', 'mean'),
    )
    summary.to_csv(out / 'prior_sensitivity_summary.csv', index=False)
    paired = paired_intervals(df)
    paired.to_csv(out / 'paired_cost_differences_ci95.csv', index=False)
    # Isolate the effect of prior choice among Bayesian methods themselves.
    prior_paired = paired_intervals(df, reference='beta_1_1')
    prior_paired = prior_paired[prior_paired.method.str.startswith('hypothetical_')]
    prior_paired.to_csv(out / 'paired_prior_vs_beta_1_1.csv', index=False)
    print('\nBayesian-minus-SAA paired cost differences (negative favors Bayesian):')
    print(paired[['n_history', 'method', 'repetitions', 'mean_cost_difference',
                  'ci95_lower', 'ci95_upper']].round(2).to_string(index=False))
    return summary, paired, observations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-sizes', nargs='+', type=int, default=[1, 2, 5, 10])
    parser.add_argument('--repetitions', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261031)
    parser.add_argument('--output-dir', default='outputs/prior_sensitivity')
    parser.add_argument('--mean-weekly-units', type=float, default=20.)
    parser.add_argument('--saa-scenarios', type=int, default=60)
    parser.add_argument('--bayes-scenarios', type=int, default=60)
    parser.add_argument('--test-scenarios', type=int, default=160)
    parser.add_argument('--prior-centers', nargs='+', type=float, default=[15., 25.])
    parser.add_argument('--prior-equivalent-weeks', type=float, default=6.)
    args = parser.parse_args()
    run(args.sample_sizes, args.repetitions, args.seed, args.output_dir,
        args.mean_weekly_units, args.saa_scenarios, args.bayes_scenarios,
        args.test_scenarios, args.prior_centers, args.prior_equivalent_weeks)

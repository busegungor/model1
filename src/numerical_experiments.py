"""Small, reproducible checks and numerical experiments for the EXISTING model.

No alternative demand family, no new optimizer, no new inventory assumptions.
An experiment may be exploratory until its Monte Carlo and replication stability
are checked. The reference_MC solution is a finite-simulation benchmark, not an
analytic oracle. The same holdout scenarios compare candidate orders.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import Economics, ExperimentConfig
from data_generation import generate_history, load_history_csv, sample_true_scenario
from evaluation import evaluate
from scenario_generation import empirical_scenarios, bayesian_scenarios
from simulation import Scenario, simulate
from solver_core import _curve_one, optimize
from benchmark_solvers import solve_no_timing


def history_for_experiment(cfg, ec, n_history, seed, input_dir=None):
    if input_dir:
        return load_history_csv(input_dir, cfg)
    return generate_history(n_history, np.random.default_rng(seed), cfg, ec)


def audit(history, cfg, ec, output_dir):
    """Check history counts, return-policy identities and scalar/grid cost equivalence."""
    sold = [r for season in history for r in season.completed_sales]
    potential = sum(len(s.demand_times) for s in history)
    assert sold and potential >= len(sold)
    accepted = [r for r in sold if r.mark.accepted]
    late = [r for r in sold if r.mark.attempts_return and not r.mark.accepted]
    assert all(1 <= r.mark.customer_days <= cfg.return_window_days for r in accepted)
    assert all(r.mark.customer_days > cfg.return_window_days for r in late)
    assert all(r.mark.resalable <= r.mark.accepted for r in sold)
    # Independently verify the fast optimizer on a small scenario without changing it.
    sc = sample_true_scenario(np.random.default_rng(923), replace(cfg, season_days=14, mean_weekly_units=5))
    cap = len(sc.demand_times) + 2
    curve = _curve_one(sc, ec, cap)
    for q in range(cap + 1):
        assert abs(curve[q] - simulate(q, sc, ec).mismatch_cost) < 1e-6
    weekly = np.zeros((len(history), cfg.season_days // 7), dtype=int)
    for i, season in enumerate(history):
        for t in season.demand_times:
            weekly[i, min(int(t // 7), weekly.shape[1]-1)] += 1
    row = dict(n_history=len(history), potential_demand_units=potential,
               observed_sale_units=len(sold), accepted_return_units=len(accepted),
               late_rejected_units=len(late), actual_resales_in_history=sum(r.sale_number_for_unit > 1 for r in sold),
               mean_weekly_units_observed=float(weekly.mean()),
               mean_weekly_units_target=cfg.mean_weekly_units,
               weekly_demand_variance_observed=float(weekly.var(ddof=1)),
               weekly_demand_variance_target=cfg.mean_weekly_units+cfg.mean_weekly_units**2/cfg.weekly_dispersion,
               accepted_return_fraction_of_sales=len(accepted)/len(sold),
               physically_resalable_fraction_of_accepted=(sum(r.mark.resalable for r in accepted)/len(accepted) if accepted else np.nan),
               cost_implementation_verified=True, return_policy_verified=True)
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out / 'audit.csv', index=False)
    return pd.DataFrame([row])


def convergence(history, cfg, ec, levels, seed, test_scenarios, reference_scenarios, output_dir):
    """Nested predictive samples, identical holdout; benchmark is optional.

    Compare stability of orders and out-of-sample cost as scenario count grows.
    This does NOT change the amount of historical data n.
    """
    levels = sorted(set(int(x) for x in levels))
    if not levels or levels[0] < 1: raise ValueError('Scenario levels must be positive')
    g = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    empirical = empirical_scenarios(history, levels[-1], g[0], cfg)
    bayesian = bayesian_scenarios(history, levels[-1], g[1], cfg)
    holdout = [sample_true_scenario(g[3], cfg) for _ in range(test_scenarios)]
    ref = None
    ref_test = np.nan
    if reference_scenarios > 0:
        reference = [sample_true_scenario(g[2], cfg) for _ in range(reference_scenarios)]
        ref = optimize('reference_MC', reference, ec)
        ref_test = evaluate(ref.q, holdout, ec)['mean_mismatch_cost']
    rows = []
    # One cost-curve calculation per scenario, reused for all nested M levels.
    for label, pool in [('SAA', empirical), ('Bayesian', bayesian)]:
        cap=max(len(sc.demand_times) for sc in pool)
        summed_curve = np.zeros(cap+1,dtype=float)
        prefix_cap = 0
        for i, sc in enumerate(pool, start=1):
            summed_curve += _curve_one(sc, ec, cap)
            prefix_cap=max(prefix_cap,len(sc.demand_times))
            if i not in levels: continue
            curve = summed_curve[:prefix_cap+1] / i
            q=int(np.argmin(curve))
            metrics = evaluate(q, holdout, ec)
            rows.append(dict(method=label, scenario_count=i, n_history=len(history),
                             reference_scenarios=reference_scenarios, test_scenarios=test_scenarios,
                             reference_order=(ref.q if ref is not None else np.nan),
                             estimated_order=q, fitted_cost=float(curve[q]),
                             test_cost=metrics['mean_mismatch_cost'],
                             estimated_gap_vs_reference_MC=metrics['mean_mismatch_cost']-ref_test,
                             fill_rate=metrics['fill_rate'],
                             actual_resales=metrics['mean_resales_actual']))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows); df.to_csv(out / 'mc_convergence.csv', index=False)
    return df


def _shift_scenario(sc, processing_days=None, dispatch_shift=0, return_window=30):
    """Preserve POTENTIAL demand, return indicators and physical condition; change ONLY timing.

    The day shifts are *controlled simulation settings*, not empirical estimates.
    Accepted dispatches are clipped to the policy window, so return eligibility
    and potential return marks do not change. Realized returns CAN change because
    changed stock availability affects the number of served sales.
    """
    marks = []
    for mark in sc.marks:
        customer_days = mark.customer_days
        if mark.accepted:
            customer_days = max(1, min(return_window, customer_days + dispatch_shift))
        marks.append(replace(mark, customer_days=customer_days,
                             processing_days=(processing_days if processing_days is not None else mark.processing_days)
                             if mark.resalable else 0))
    return replace(sc, marks=tuple(marks))


def timing_sensitivity(history, cfg, ec, seed, scenarios, output_dir,
                       processing_values=(1, 2, 3, 5), dispatch_shifts=(-7, 0, 7)):
    """Sensitivity of FIXED, historically trained SAA/Bayesian policies to return timing.

    Both policies are trained on the same observed history (once) and evaluated on
    identical latent potential events for every timing setting.  This evaluates
    policy robustness, NOT the value of relearning after a policy change.
    The MC reference is reoptimized for each setting, so is not a third learner.
    """
    gs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)]
    policies = {
        'SAA': optimize('SAA', empirical_scenarios(history, cfg.saa_scenarios, gs[0], cfg), ec).q,
        'Bayesian': optimize('Bayesian', bayesian_scenarios(history, cfg.bayes_scenarios, gs[1], cfg), ec).q,
        # separate stream, so the SAA and Bayesian draws above are unchanged
        'No timing': solve_no_timing(history, np.random.default_rng(np.random.SeedSequence([seed, 77])), cfg, ec).q,
    }
    reference_pool = [sample_true_scenario(gs[2], cfg) for _ in range(scenarios)]
    holdout_pool = [sample_true_scenario(gs[3], cfg) for _ in range(scenarios)]
    rows = []
    settings = ([('processing_days', x, x, 0) for x in processing_values] +
                [('dispatch_shift_days', x, cfg.processing_days, x) for x in dispatch_shifts])
    for factor, value, processing, shift in settings:
        reference_scenarios = [_shift_scenario(sc, processing, shift, cfg.return_window_days)
                               for sc in reference_pool]
        test_scenarios = [_shift_scenario(sc, processing, shift, cfg.return_window_days)
                          for sc in holdout_pool]
        ref = optimize('reference_MC', reference_scenarios, ec)
        reference_eval = evaluate(ref.q, test_scenarios, ec)
        for method, q in policies.items():
            result = evaluate(q, test_scenarios, ec)
            rows.append(dict(factor=factor, value=value, method=method, n_history=len(history),
                             reference_scenarios=scenarios, test_scenarios=scenarios,
                             fixed_baseline_Q=q, reference_MC_Q=ref.q,
                             fixed_Q_cost=result['mean_mismatch_cost'],
                             reference_MC_cost=reference_eval['mean_mismatch_cost'],
                             gap_vs_reference_MC=(result['mean_mismatch_cost']-
                                                  reference_eval['mean_mismatch_cost']),
                             fixed_Q_resales=result['mean_resales_actual'],
                             reference_MC_resales=reference_eval['mean_resales_actual'],
                             fixed_Q_lost_sales=result['mean_lost_sales'],
                             mean_returns=result['mean_returns']))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / 'intraperiod_sensitivity.csv', index=False)
    return df


def operational_sensitivity(cfg, ec, seed, n_history, scenarios, test_scenarios, output_dir,
                            factors=('return_probability', 'resalability', 'season_length')):
    """One-factor-at-a-time sensitivity under the SAME Negative Binomial DGP family.

    Sensitivity values are explicitly designed stress tests, NOT fitted numbers.
    Season-length settings hold expected TOTAL demand constant to focus on time.
    RNG streams are aligned within each factor for partial noise control;
    conditional sampling still creates some unmatched random draws.
    """
    specs = {
        'return_probability': [(0.30, {'return_attempt_probability': 0.30}),
                               (cfg.return_attempt_probability, {}),
                               (0.70, {'return_attempt_probability': 0.70})],
        'resalability': [(0.70, {'return_resalable_probability': 0.70}),
                         (cfg.return_resalable_probability, {})],
        'season_length': [(weeks, {'season_days': 7*weeks,
                                   'mean_weekly_units': cfg.mean_weekly_units*(cfg.season_days//7)/weeks})
                          for weeks in (12, 18, 26)],
    }
    rows=[]
    for factor_idx, factor in enumerate(factors):
        if factor not in specs: raise ValueError(f'Unknown sensitivity factor: {factor}')
        for val, replacements in specs[factor]:
            ccfg = replace(cfg, **replacements)
            # Multiple stable RNG streams; procedures receive same observation history.
            sc_seed=np.random.SeedSequence([seed, factor_idx+1]); g=[np.random.default_rng(s) for s in sc_seed.spawn(5)]
            hist=generate_history(n_history,g[0],ccfg,ec)
            sa=empirical_scenarios(hist, scenarios,g[1],ccfg)
            ba=bayesian_scenarios(hist,scenarios,g[2],ccfg)
            test=[sample_true_scenario(g[3],ccfg) for _ in range(test_scenarios)]
            ref=[sample_true_scenario(g[4],ccfg) for _ in range(scenarios)]
            policies=[optimize('SAA',sa,ec),optimize('Bayesian',ba,ec),optimize('reference_MC',ref,ec)]
            for pol in policies:
                metrics=evaluate(pol.q,test,ec)
                rows.append(dict(factor=factor,value=val,method=pol.method,n_history=n_history,
                                 scenario_count=scenarios,test_scenarios=test_scenarios,
                                 mean_total_demand_target=ccfg.mean_weekly_units*(ccfg.season_days//7),
                                 Q=pol.q,mean_mismatch_cost=metrics['mean_mismatch_cost'],
                                 actual_resales=metrics['mean_resales_actual'],
                                 fill_rate=metrics['fill_rate']))
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows);df.to_csv(out/'operational_sensitivity.csv',index=False)
    return df


def main():
    ap = argparse.ArgumentParser(description='Numerical checks for existing Negative Binomial baseline only')
    ap.add_argument('experiment',choices=['audit','convergence','intraperiod','operational'])
    ap.add_argument('--input-dir', help='Existing historical_data CSV folder (audit/convergence/intraperiod)')
    ap.add_argument('--output-dir', default='outputs/experiments')
    ap.add_argument('--n-history',type=int,default=5)
    ap.add_argument('--mean-weekly-units',type=float,default=10.,help='Use 150 for the full baseline')
    ap.add_argument('--season-days',type=int,default=126)
    ap.add_argument('--seed',type=int,default=20261030)
    ap.add_argument('--scenario-count',type=int,default=30)
    ap.add_argument('--test-scenarios',type=int,default=50)
    ap.add_argument('--reference-scenarios',type=int,default=0,
                    help='0 skips the slow optional reference benchmark for MC convergence')
    ap.add_argument('--mc-levels',nargs='+',type=int,default=[10,20,30])
    ap.add_argument('--factors',nargs='+',choices=['return_probability','resalability','season_length'],
                    default=['return_probability','resalability','season_length'])
    args=ap.parse_args()
    cfg=ExperimentConfig(season_days=args.season_days,mean_weekly_units=args.mean_weekly_units,
                         saa_scenarios=args.scenario_count, bayes_scenarios=args.scenario_count)
    ec=Economics()
    if args.experiment=='operational':
        if args.input_dir: ap.error('operational sensitivity generates new historical data; do not pass --input-dir')
        df=operational_sensitivity(cfg,ec,args.seed,args.n_history,args.scenario_count,
                                  args.test_scenarios,args.output_dir,args.factors)
    else:
        hist=history_for_experiment(cfg,ec,args.n_history,args.seed,args.input_dir)
        if args.experiment=='audit':df=audit(hist,cfg,ec,args.output_dir)
        elif args.experiment=='convergence':
            df=convergence(hist,cfg,ec,args.mc_levels,args.seed,args.test_scenarios,
                           args.reference_scenarios,args.output_dir)
        else:
            df=timing_sensitivity(hist,cfg,ec,args.seed,args.scenario_count,args.output_dir)
    print(df.round(3).to_string(index=False))
    print(f'Files written to {Path(args.output_dir).resolve()}')


if __name__=='__main__': main()

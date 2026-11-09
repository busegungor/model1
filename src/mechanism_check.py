"""Why the benchmarks order what they order (baseline, full information).

On one common set of true scenarios, computes
  * the true expected mismatch cost l(Q) for every integer Q,
  * the cost curve the 'No timing' rule plans with (same scenarios, all return
    lead times set to zero) and its minimizer,
  * the cost curve the 'Ignore returns' rule plans with (same demand, no returns)
    and its minimizer,
  * lost sales, leftover stock, resalable returns ready only after the season and
    resales under the true process on a grid of Q.
With true scenarios instead of estimated ones, the minimizers are the orders each
planning view converges to with unlimited data, so their gap to Q* is bias, not
estimation error.
"""
import argparse
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from config import ExperimentConfig, Economics
from data_generation import sample_true_scenario
from simulation import Scenario, PotentialMark
from solver_core import optimize
from evaluation import evaluate
from benchmark_solvers import INSTANT


def run(scenarios=1000, seed=20261109, mean_weekly_units=20., q_min=150, q_max=400, q_step=5,
        output_dir='outputs/mechanism'):
    cfg = ExperimentConfig(mean_weekly_units=float(mean_weekly_units))
    ec = Economics()
    rng = np.random.default_rng(seed)
    true_sc = [sample_true_scenario(rng, cfg) for _ in range(scenarios)]
    nt_sc = [Scenario(sc.demand_times,
                      tuple(replace(m, outbound_days=0, customer_days=0, reverse_days=0, processing_days=INSTANT)
                            for m in sc.marks), sc.horizon) for sc in true_sc]
    ir_sc = [Scenario(sc.demand_times, tuple(PotentialMark() for _ in sc.demand_times), sc.horizon)
             for sc in true_sc]
    true, nt, ir = (optimize(name, s, ec) for name, s in
                    (('true', true_sc), ('No timing', nt_sc), ('Ignore returns', ir_sc)))
    n = min(len(true.cost_curve), len(nt.cost_curve), len(ir.cost_curve))
    curves = pd.DataFrame(dict(Q=np.arange(n), true_cost=true.cost_curve[:n],
                               no_timing_planning_cost=nt.cost_curve[:n],
                               ignore_returns_planning_cost=ir.cost_curve[:n]))
    grid = sorted(set(range(q_min, q_max + 1, q_step)) | {true.q, nt.q, ir.q})
    rows = []
    for q in grid:
        e = evaluate(q, true_sc, ec)
        rows.append(dict(Q=q, true_cost=e['mean_mismatch_cost'], lost_sales=e['mean_lost_sales'],
                         leftover=e['mean_leftover'], late_resalable=e['mean_late_resalable'],
                         resales=e['mean_resales_actual'], sales=e['mean_sales'], fill_rate=e['fill_rate']))
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out / 'cost_curves.csv', index=False)
    pd.DataFrame(rows).to_csv(out / 'outcomes_by_Q.csv', index=False)
    summ = pd.DataFrame([dict(view=v, planned_Q=s.q, true_cost_at_planned_Q=true.cost_curve[s.q],
                              excess_true_cost=true.cost_curve[s.q] - true.cost_curve[true.q])
                         for v, s in (('Full information', true), ('No timing', nt), ('Ignore returns', ir))])
    summ.to_csv(out / 'planning_views.csv', index=False)
    print(summ.round(2).to_string(index=False))
    print(pd.DataFrame(rows).set_index('Q').loc[[true.q, nt.q, ir.q]].round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261109)
    ap.add_argument('--mean-weekly-units', type=float, default=20.)
    ap.add_argument('--output-dir', default='outputs/mechanism')
    a = ap.parse_args()
    run(a.scenarios, a.seed, a.mean_weekly_units, output_dir=a.output_dir)

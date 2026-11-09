"""Descriptive statistics of the generated seasons (one row per quantity).

Seasons are drawn from the true data-generating process and operated at the
full-information order Q* (approximated by SAA on independent scenarios), so
the table shows what a season looks like under good ordering.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from config import ExperimentConfig, Economics
from data_generation import sample_true_scenario
from simulation import simulate
from solver_core import optimize

FIELDS = [  # (label, function of outcome)
    ('Potential demand (units)',            lambda o: o.potential_demand),
    ('Sales, incl. resales (units)',        lambda o: o.sales),
    ('Lost sales (units)',                  lambda o: o.lost_sales),
    ('Return attempts',                     lambda o: o.accepted_returns + o.rejected_late_attempts),
    ('Rejected late attempts',              lambda o: o.rejected_late_attempts),
    ('Accepted returns',                    lambda o: o.accepted_returns),
    ('Resalable, ready within season',      lambda o: o.ontime_resalable_returns),
    ('Resalable, ready after season',       lambda o: o.late_resalable_returns),
    ('Non-resalable',                       lambda o: o.nonresalable_returns),
    ('Resales',                             lambda o: o.actual_resales),
    ('Leftover stock (units)',              lambda o: o.leftover),
]

def run(mean_weekly_units=20., seasons=1000, reference_scenarios=1000, seed=20261106,
        output_dir='outputs/data_summary'):
    cfg, ec = ExperimentConfig(mean_weekly_units=float(mean_weekly_units)), Economics()
    g_ref, g_sea = [np.random.default_rng(s) for s in np.random.SeedSequence([seed, 7]).spawn(2)]
    q = optimize('ref', [sample_true_scenario(g_ref, cfg) for _ in range(reference_scenarios)], ec).q
    out = [simulate(q, sample_true_scenario(g_sea, cfg), ec) for _ in range(seasons)]
    rows = [dict(quantity=lab, mean=np.mean([f(o) for o in out]), sd=np.std([f(o) for o in out], ddof=1))
            for lab, f in FIELDS]
    df = pd.DataFrame(rows)
    sales = sum(o.sales for o in out); demand = sum(o.potential_demand for o in out)
    print(f'mu={mean_weekly_units}, Q*={q}, seasons={seasons}, fill rate={sales/demand:.3f}')
    print(df.round(1).to_string(index=False))
    p = Path(output_dir); p.mkdir(parents=True, exist_ok=True)
    df.assign(mu=mean_weekly_units, Q_star=q, fill_rate=sales / demand).to_csv(
        p / f'data_summary_mu{int(mean_weekly_units)}.csv', index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--mean-weekly-units', type=float, default=20.)
    ap.add_argument('--seasons', type=int, default=1000)
    ap.add_argument('--reference-scenarios', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20261106)
    ap.add_argument('--output-dir', default='outputs/data_summary')
    a = ap.parse_args()
    run(a.mean_weekly_units, a.seasons, a.reference_scenarios, a.seed, a.output_dir)
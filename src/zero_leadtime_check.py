"""Numerical check of the zero-lead-time proposition (net-demand quantile).

In the limit with all return lead times zero and unit-size customer orders, the
proposition states that l(Q) has nondecreasing increments and is minimized by the
alpha-quantile of net demand N = sum_i eta_i, eta_i = 1 - (accepted_i * resalable_i),
with alpha = (v - c)/(v - s) and
    v = [p q_K - (c_B + c_P) a - (c_N - s_N) q_X] / (1 - q_R),
    a = r_A (1 - lambda), q_R = a k_C, q_X = a (1 - k_C), q_K = 1 - a.
Weekly demand is NB(mu, kappa), so seasonal demand D ~ NB(W kappa, phi) and
N | D ~ Bin(D, 1 - q_R); the quantile is computed exactly from this mixture.
The script compares it with the minimizer of the simulated loss under
(i) unit-size orders and (ii) the geometric order sizes of the data-generating process.
"""
import argparse
from dataclasses import replace
import numpy as np
from scipy.stats import nbinom, binom
from config import ExperimentConfig, Economics
from data_generation import sample_true_scenario
from simulation import Scenario
from solver_core import optimize
from benchmark_solvers import INSTANT


def proposition_quantile(cfg, ec):
    a = cfg.return_attempt_probability * (1 - cfg.late_dispatch_given_attempt)
    qR, qX = a * cfg.return_resalable_probability, a * (1 - cfg.return_resalable_probability)
    qK = 1 - a
    h = ec.reverse_transport_cost + ec.return_processing_cost
    m = ec.nonresalable_disposal_cost - ec.nonresalable_salvage_value
    v = (ec.price * qK - h * a - m * qX) / (1 - qR)
    alpha = (v - ec.procurement_cost) / (v - ec.salvage_value)
    W = cfg.season_days // 7
    kappa, mu = cfg.weekly_dispersion, cfg.mean_weekly_units
    phi = kappa / (kappa + mu)
    d = np.arange(0, int(nbinom.ppf(1 - 1e-12, W * kappa, phi)) + 1)
    pd = nbinom.pmf(d, W * kappa, phi)
    qs = np.arange(0, d.max() + 1)
    cdf = np.array([(pd * binom.cdf(q, d, 1 - qR)).sum() for q in qs])
    return dict(v=v, alpha=alpha, q_star=int(qs[np.argmax(cdf >= alpha)]),
                mean_N=float((pd * d).sum() * (1 - qR)))


def zero_lead(sc, unit_orders):
    t = np.asarray(sc.demand_times, float)
    if unit_orders:  # separate the units of one order by a tiny time step
        t = t + np.arange(len(t)) * 1e-6
    marks = tuple(replace(m, outbound_days=0, customer_days=0, reverse_days=0, processing_days=INSTANT)
                  for m in sc.marks)
    return Scenario(tuple(t), marks, sc.horizon)


def run(scenarios=4000, seed=20261110, mu=20.):
    cfg, ec = ExperimentConfig(mean_weekly_units=float(mu)), Economics()
    prop = proposition_quantile(cfg, ec)
    rng = np.random.default_rng(seed)
    base = [sample_true_scenario(rng, cfg) for _ in range(scenarios)]
    N = np.array([sum(1 - (m.accepted and m.resalable) for m in sc.marks) for sc in base])
    emp_q = int(np.quantile(N, prop['alpha'], method='inverted_cdf'))
    print(f"v={prop['v']:.3f}  alpha={prop['alpha']:.4f}  E[N]={prop['mean_N']:.2f} (sample {N.mean():.2f})")
    print(f"exact alpha-quantile of N: {prop['q_star']}   empirical alpha-quantile ({scenarios} scen.): {emp_q}")
    for unit in (True, False):
        sol = optimize('zero', [zero_lead(sc, unit) for sc in base], ec)
        inc = np.diff(sol.cost_curve)
        lo, hi = max(prop['q_star'] - 40, 0), prop['q_star'] + 40
        # expected increment implied by the proposition, using the sample distribution of N
        Qs = np.arange(lo, hi)
        pred = (ec.procurement_cost - ec.salvage_value) - (prop['v'] - ec.salvage_value) * \
            np.array([(N >= q + 1).mean() for q in Qs])
        err = inc[lo:hi] - pred
        print(f"{'unit orders' if unit else 'geometric orders'}: simulated minimizer {sol.q}; "
              f"increment vs. proposition on Q in [{lo},{hi}): mean abs diff {np.abs(err).mean():.3f}, "
              f"max abs diff {np.abs(err).max():.3f} (increments range {inc[lo:hi].min():.2f} to {inc[lo:hi].max():.2f})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenarios', type=int, default=4000)
    ap.add_argument('--seed', type=int, default=20261110)
    ap.add_argument('--mean-weekly-units', type=float, default=20.)
    a = ap.parse_args()
    run(a.scenarios, a.seed, a.mean_weekly_units)
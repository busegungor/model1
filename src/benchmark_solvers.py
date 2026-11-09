"""Benchmark ordering rules that learn from the same observed history as SAA.

Both rules are evaluated in the same event-time world as SAA and Bayesian;
only their *planning* view of returns is simplified.
"""
from dataclasses import replace
import numpy as np
from config import ExperimentConfig, Economics
from simulation import Scenario, PotentialMark
from scenario_generation import empirical_scenarios
from solver_core import optimize

INSTANT = 1e-9   # days


def solve_ignore_returns(history, rng: np.random.Generator, cfg: ExperimentConfig, ec: Economics):
    """Classical data-driven newsvendor: plans as if no unit were ever returned.

    SAA over the observed seasonal demand paths with all return marks removed,
    i.e. an empirical quantile of total seasonal demand.
    """
    scenarios = [Scenario(h.demand_times, tuple(PotentialMark() for _ in h.demand_times), cfg.season_days)
                 for h in history]
    return optimize('Ignore returns', scenarios, ec)


def solve_no_timing(history, rng: np.random.Generator, cfg: ExperimentConfig, ec: Economics):
    """Netting benchmark: same empirical scenarios as SAA, but every accepted
    resalable return is available for resale immediately after its sale (all lead
    times zero), as in a net-demand formulation without return timing.

    A tiny positive processing time (1e-9 days) keeps the return-ready event
    strictly after its own sale event in the event ordering.
    """
    scenarios = []
    for sc in empirical_scenarios(history, cfg.saa_scenarios, rng, cfg):
        marks = tuple(replace(m, outbound_days=0, customer_days=0, reverse_days=0, processing_days=INSTANT)
                      for m in sc.marks)
        scenarios.append(Scenario(sc.demand_times, marks, cfg.season_days))
    return optimize('No timing', scenarios, ec)
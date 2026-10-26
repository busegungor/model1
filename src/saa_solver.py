"""Empirical SAA on historical potential-demand paths and resolved sale outcomes."""
import numpy as np
from config import ExperimentConfig, Economics
from data_generation import HistoricalSeason, completed_sale_cohort
from scenario_generation import empirical_scenarios
from solver_core import optimize, Solution


def solve_saa(history: list[HistoricalSeason], rng: np.random.Generator,
              cfg: ExperimentConfig, ec: Economics) -> Solution:
    completed=completed_sale_cohort(history,cfg)
    scenarios=empirical_scenarios(history,completed,cfg.saa_scenarios,rng,cfg)
    return optimize("SAA",scenarios,ec)
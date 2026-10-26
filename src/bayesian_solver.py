"""Posterior predictive ordering under a Bayesian-bootstrap/Beta baseline model."""
import numpy as np
from config import ExperimentConfig, Economics
from data_generation import HistoricalSeason, completed_sale_cohort
from scenario_generation import bayesian_scenarios
from solver_core import optimize, Solution


def solve_bayesian(history: list[HistoricalSeason], rng: np.random.Generator,
                   cfg: ExperimentConfig, ec: Economics) -> Solution:
    completed=completed_sale_cohort(history,cfg)
    scenarios=bayesian_scenarios(history,completed,cfg.bayes_scenarios,rng,cfg)
    return optimize("Bayesian",scenarios,ec)
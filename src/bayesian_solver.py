import numpy as np
from config import ExperimentConfig,Economics
from scenario_generation import bayesian_scenarios
from solver_core import optimize

def solve_bayesian(history,rng:np.random.Generator,cfg:ExperimentConfig,ec:Economics):
    return optimize('Bayesian',bayesian_scenarios(history,cfg.bayes_scenarios,rng,cfg),ec)

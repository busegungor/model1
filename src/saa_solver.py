import numpy as np
from config import ExperimentConfig,Economics
from scenario_generation import empirical_scenarios
from solver_core import optimize

def solve_saa(history,rng:np.random.Generator,cfg:ExperimentConfig,ec:Economics):
    return optimize('SAA',empirical_scenarios(history,cfg.saa_scenarios,rng,cfg),ec)

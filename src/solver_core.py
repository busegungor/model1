"""Common-random-number finite integer optimizer for both procedures."""
from dataclasses import dataclass
import numpy as np
from config import Economics
from simulation import Scenario, simulate


@dataclass(frozen=True)
class Solution:
    method: str
    q: int
    estimated_cost: float
    q_grid: tuple[int,...]
    cost_curve: tuple[float,...]


def optimize(method: str, scenarios: list[Scenario], ec: Economics) -> Solution:
    if not scenarios:
        raise ValueError("Expected non-empty scenarios")
    # c>g and at most one sale per potential demand event imply that ordering
    # beyond maximum scenario demand cannot improve scenario-average loss.
    cap=max(len(sc.demand_times) for sc in scenarios)
    grid=tuple(range(cap+1))
    curve=[]
    for q in grid:
        vals=[simulate(q,sc,ec).mismatch_cost for sc in scenarios]
        curve.append(float(np.mean(vals)))
    # Smallest minimizer breaks ties reproducibly.
    ix=int(np.argmin(curve))
    return Solution(method,grid[ix],curve[ix],grid,tuple(curve))
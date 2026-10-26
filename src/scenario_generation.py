"""Make policy-independent counterfactual scenarios using only *eligible history*.

Observed historical returns are not copied as fixed aggregate totals for new Q values.
Each potential demand event gets a new, privately sampled mark. Those marks only
become sales/returns if the simulated inventory actually serves the event.
"""
import numpy as np
from config import ExperimentConfig
from data_generation import HistoricalSeason, ResolvedSale
from simulation import Scenario, PotentialMark


def empirical_scenarios(history: list[HistoricalSeason], completed: list[ResolvedSale],
                        count: int, rng: np.random.Generator, cfg: ExperimentConfig) -> list[Scenario]:
    if not history or not completed:
        raise ValueError("SAA requires at least one complete historical sale")
    output = []
    for _ in range(count):
        times = history[int(rng.integers(len(history)))].demand_times
        marks = []
        for _ in times:
            record = completed[int(rng.integers(len(completed)))]
            marks.append(PotentialMark(record.returned, record.resalable,
                                       record.return_delay, record.processing_delay))
        output.append(Scenario(times, tuple(marks), cfg.selling_days))
    return output


def bayesian_scenarios(history: list[HistoricalSeason], completed: list[ResolvedSale],
                       count: int, rng: np.random.Generator, cfg: ExperimentConfig,
                       beta_prior: tuple[float, float] = (1., 1.)) -> list[Scenario]:
    """Baseline Bayesian posterior predictive model.

    Demand: Bayesian bootstrap over observed complete trajectories (limited support).
    Returns/resalability: independent Beta--Bernoulli posteriors.
    Delay: smoothed discrete categorical posterior on declared support values.

    Bayesian bootstrap with risk-neutral expected cost has the same *marginal mean*
    demand distribution as empirical resampling. Differences mostly reflect return/
    condition/delay smoothing; alternative demand likelihoods are a research choice.
    """
    if not history or not completed:
        raise ValueError("Bayesian baseline requires historical observations")
    a, b = beta_prior
    if a<=0 or b<=0:
        raise ValueError("Beta hyperparameters must be positive")
    nr = sum(row.returned for row in completed)
    nret = len(completed)
    ngood = sum(row.resalable for row in completed if row.returned)
    ralpha, rbeta = a+nr, b+(nret-nr)
    kalpha, kbeta = a+ngood, b+(nr-ngood)
    return_pool = [row.return_delay for row in completed if row.returned]
    processing_pool = [row.processing_delay for row in completed if row.returned and row.resalable]
    # Prior masses on fixed permitted delays allow nonzero predictive support even
    # if a particular empirical delay has not yet occurred.
    def smoothed_choices(support, observed):
        arr=np.asarray(support,dtype=float)
        mass=np.ones(len(arr),dtype=float)
        for val in observed:
            matches=np.flatnonzero(np.isclose(arr, val))
            if len(matches): mass[matches[0]] += 1.
        return arr, mass/mass.sum()
    lags, lag_weights = smoothed_choices(cfg.return_delays, return_pool)
    proc, proc_weights = smoothed_choices(cfg.processing_delays, processing_pool)
    output=[]
    for _ in range(count):
        # Bayesian bootstrap (Dirichlet weights) does not introduce new demand levels.
        weights=rng.dirichlet(np.ones(len(history)))
        times=history[int(rng.choice(len(history),p=weights))].demand_times
        r=float(rng.beta(ralpha,rbeta))
        k=float(rng.beta(kalpha,kbeta))
        marks=[]
        for _ in times:
            ret=bool(rng.random()<r)
            good=bool(rng.random()<k) if ret else False
            delay=float(rng.choice(lags,p=lag_weights)) if ret else 0.
            processing=float(rng.choice(proc,p=proc_weights)) if good else 0.
            marks.append(PotentialMark(ret,good,delay,processing))
        output.append(Scenario(times,tuple(marks),cfg.selling_days))
    return output
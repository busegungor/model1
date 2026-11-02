"""Model-based and empirical predictive scenarios use OBSERVED history only.
No true return probability or DGP mean enters the estimators. The Bayesian
dispersion and prior are predeclared MODEL assumptions (require sensitivity).
"""
import numpy as np
from config import ExperimentConfig
from simulation import Scenario,PotentialMark
from data_generation import completed_sale_cohort,HistoricalSeason


def empirical_scenarios(history,count,rng,cfg):
    sales=completed_sale_cohort(history,cfg)
    scenarios=[]
    for _ in range(count):
        ts=history[int(rng.integers(len(history)))].demand_times
        marks=[sales[int(rng.integers(len(sales)))].mark for _ in ts]
        scenarios.append(Scenario(ts,tuple(marks),cfg.season_days))
    return scenarios


def _weekly_counts(history,cfg):
    d=np.zeros(len(history)*int(cfg.season_days//7),dtype=int)
    for i,s in enumerate(history):
        for t in s.demand_times: d[i*int(cfg.season_days//7)+int(t//7)]+=1
    return d


def demand_probability_posterior(history, cfg, prior_alpha=1.0, prior_beta=1.0):
    """Exact Beta conjugate update for NB demand with fixed assumed dispersion.

    For Y ~ NB(kappa, phi), P(Y=y) proportional to
    phi**kappa * (1-phi)**y, and E[Y] = kappa*(1-phi)/phi.
    The prior is on NB probability phi, NOT directly on demand mean.
    The assumed dispersion is an estimator input, not read from DGP truth.
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError('Beta prior parameters must be positive')
    counts = _weekly_counts(history, cfg)
    if len(counts) == 0:
        raise ValueError('Historical demand observations required')
    kappa = cfg.bayesian_assumed_dispersion
    return (float(prior_alpha + len(counts)*kappa),
            float(prior_beta + counts.sum()))


def bayesian_scenarios(history,count,rng,cfg,prior_alpha=1.,prior_beta=1.):
    sales=completed_sale_cohort(history,cfg)
    n=len(sales)
    attempts=sum(r.mark.attempts_return for r in sales)
    accepted=sum(r.mark.accepted for r in sales)
    good=sum(r.mark.resalable for r in sales)
    # Beta(1,1) parameter priors (explicit experimental choices)
    a_attempt,b_attempt=1+attempts,1+n-attempts
    a_late,b_late=1+attempts-accepted,1+accepted
    a_good,b_good=1+good,1+accepted-good
    outbound=[r.mark.outbound_days for r in sales]
    customer=[r.mark.customer_days for r in sales if r.mark.accepted]
    reverse=[r.mark.reverse_days for r in sales if r.mark.attempts_return]
    late_d=[r.mark.customer_days for r in sales if r.mark.attempts_return and not r.mark.accepted]
    reason=[r.mark.reason for r in sales if r.mark.accepted]
    phi_a,phi_b=demand_probability_posterior(history,cfg,prior_alpha,prior_beta)
    kappa=cfg.bayesian_assumed_dispersion
    scenarios=[]
    weeks=cfg.season_days//7
    for _ in range(count):
        phi=float(rng.beta(phi_a,phi_b))
        p_attempt=float(rng.beta(a_attempt,b_attempt))
        p_late=float(rng.beta(a_late,b_late))
        p_good=float(rng.beta(a_good,b_good))
        times=[]
        for w in range(weeks):
            units=int(rng.negative_binomial(kappa,phi))
            times.extend(float(7*w+rng.uniform(0.01,6.99)) for __ in range(units))
        times.sort()
        marks=[]
        for __ in times:
            od=int(rng.choice(outbound))
            attempt=bool(rng.random()<p_attempt)
            if not attempt:
                marks.append(PotentialMark(outbound_days=od))
                continue
            late=bool(rng.random()<p_late)
            # Model support is a predeclared POLICY assumption, not hidden truth.
            cust=int(rng.choice(late_d)) if (late and late_d) else (
                int(rng.integers(cfg.return_window_days+1,cfg.late_dispatch_last_day+1))
                if late else (int(rng.choice(customer)) if customer else 1))
            rev=int(rng.choice(reverse)) if reverse else 2
            accepted=not late
            good=bool(rng.random()<p_good) if accepted else False
            why=str(rng.choice(reason)) if accepted and reason else ('late_request' if late else 'unspecified')
            marks.append(PotentialMark(True,accepted,good,od,cust,rev,
                                       cfg.processing_days if good else 0,why))
        scenarios.append(Scenario(tuple(times),tuple(marks),cfg.season_days))
    return scenarios

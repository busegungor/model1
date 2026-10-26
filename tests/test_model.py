import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from config import Economics, ExperimentConfig
from simulation import Scenario, PotentialMark, simulate
from data_generation import generate_history, completed_sale_cohort, sample_true_scenario, history_records
from scenario_generation import empirical_scenarios, bayesian_scenarios
from solver_core import optimize

EC=Economics()


def test_no_returns_classical_cost():
    sc=Scenario((2.,4.,5.),tuple(PotentialMark() for _ in range(3)))
    for q in range(6):
        o=simulate(q,sc,EC)
        expected=(EC.price-EC.procurement_cost)*max(3-q,0)+(EC.procurement_cost-EC.salvage_value)*max(q-3,0)
        assert o.mismatch_cost==expected


def test_one_unit_resold_twice():
    # Unit sold day 1, returns ready day 3, resold day 4,
    # return ready day 6, resold day 7 -> 3 sales with Q=1.
    sc=Scenario((1.,4.,7.),(
        PotentialMark(True,True,1.,1.),
        PotentialMark(True,True,1.,1.),
        PotentialMark(False,False,0.,0.)))
    o=simulate(1,sc,EC)
    assert (o.sales,o.lost_sales,o.ontime_resalable_returns,o.leftover)==(3,0,2,0)
    assert o.total_returns==2


def test_terminal_available_vs_late():
    s=Scenario((29.,), (PotentialMark(True,True,.25,.25),),30.)
    o=simulate(1,s,EC)
    assert (o.ontime_resalable_returns,o.late_resalable_returns,o.leftover)==(1,0,1)
    s2=Scenario((29.,), (PotentialMark(True,True,2.,1.),),30.)
    o2=simulate(1,s2,EC)
    assert (o2.ontime_resalable_returns,o2.late_resalable_returns,o2.leftover)==(0,1,0)


def test_nonresalable_return_not_removed_twice():
    o=simulate(1,Scenario((1.,),(PotentialMark(True,False,2.,0.),)),EC)
    assert o.leftover==0 and o.nonresalable_returns==1 and o.total_returns==1
    assert o.mismatch_cost==EC.price+EC.return_processing_cost+EC.nonresalable_disposal_cost


def test_return_never_precedes_sale():
    sc=Scenario((1.,2.),(PotentialMark(True,True,0.,0.),PotentialMark()))
    o=simulate(1,sc,EC)
    # Same-timestamp return becomes available for the next event, not earlier.
    assert o.sales==2 and o.ontime_resalable_returns==1


def test_pending_not_counted_as_no_return():
    rng=np.random.default_rng(41)
    cfg=ExperimentConfig()
    history=generate_history(4,rng,cfg,EC)
    cohort=completed_sale_cohort(history,cfg)
    assert len(cohort)<sum(len(s.observed_sales) for s in history)
    statuses={r['status'] for r in history_records(history,cfg)}
    assert 'pending' in statuses


def test_empirical_and_bayes_use_same_physical_cost():
    rng=np.random.default_rng(55)
    cfg=ExperimentConfig()
    hist=generate_history(4,rng,cfg,EC)
    eligible=completed_sale_cohort(hist,cfg)
    saa=empirical_scenarios(hist,eligible,4,rng,cfg)
    bay=bayesian_scenarios(hist,eligible,4,rng,cfg)
    assert len(saa)==len(bay)==4
    for sc in saa+bay:
        simulate(5,sc,EC).check(EC)


def test_candidate_bound_and_reproducibility():
    sc=Scenario((1.,2.,3.),tuple(PotentialMark() for _ in range(3)))
    sol=optimize('test',[sc],EC)
    assert sol.q==3 and sol.q_grid[-1]==3
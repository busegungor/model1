import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from config import ExperimentConfig,Economics
from simulation import Scenario,PotentialMark,simulate
from solver_core import _curve_one,optimize
from data_generation import (sample_true_scenario,generate_history,export_history_csv,
    load_history_csv,maxent_daily_weights,conditional_dispatch_weights)
from scenario_generation import empirical_scenarios,bayesian_scenarios,demand_probability_posterior

EC=Economics()

def test_classical_no_returns():
    sc=Scenario((2.,4.,5.),tuple(PotentialMark() for _ in range(3)),30.)
    for q in range(5):
        out=simulate(q,sc,EC)
        assert abs(out.mismatch_cost-((EC.price-EC.procurement_cost)*max(3-q,0)+
                      (EC.procurement_cost-EC.salvage_value)*max(q-3,0)))<1e-9

def test_repeat_sale_same_physical_unit():
    mark=PotentialMark(True,True,True,1,1,1,1,'too_large')
    sc=Scenario((1.,6.,12.),(mark,mark,PotentialMark()),20.)
    o=simulate(1,sc,EC)
    assert o.sales==3 and o.actual_resales==2 and o.accepted_returns==2
    assert o.records[0].physical_unit_id==o.records[1].physical_unit_id==o.records[2].physical_unit_id

def test_late_dispatch_rejected_no_refund_or_inflow():
    m=PotentialMark(True,False,False,2,31,2,0,'late_request')
    sc=Scenario((1.,40.),(m,PotentialMark()),126.)
    o=simulate(1,sc,EC)
    assert (o.accepted_returns,o.rejected_late_attempts,o.sales,o.lost_sales)==(0,1,1,1)
    assert o.net_cash_cost==EC.procurement_cost-EC.price

def test_on_time_dispatch_return_arriving_after_season_is_refunded():
    m=PotentialMark(True,True,True,2,30,5,2,'too_small')
    out=simulate(1,Scenario((120.,),(m,),126.),EC)
    assert out.accepted_returns==1 and out.late_resalable_returns==1
    assert out.ontime_resalable_returns==0 and out.leftover==0

def test_curve_equals_scalar_all_q():
    rng=np.random.default_rng(8)
    cfg=ExperimentConfig(season_days=14,mean_weekly_units=3,return_attempt_probability=.7,
                         late_dispatch_given_attempt=.2)
    cases=[sample_true_scenario(rng,cfg) for _ in range(4)]
    for sc in cases:
        cap=len(sc.demand_times)+3
        curve=_curve_one(sc,EC,cap)
        for q in range(cap+1):
            assert abs(curve[q]-simulate(q,sc,EC).mismatch_cost)<1e-6

def test_calibrated_mean_and_early_share():
    cfg=ExperimentConfig()
    d=np.asarray(cfg.outbound_days)
    w=maxent_daily_weights(d,cfg.outbound_mean_target,cfg.outbound_p_le2_target)
    assert abs(w@d-4.93)<1e-6 and abs(w[d<=2].sum()-.079)<1e-6
    rev=np.asarray(cfg.reverse_days)
    assert abs(maxent_daily_weights(rev,2.24)@rev-2.24)<1e-6
    p=conditional_dispatch_weights(cfg)
    wb=maxent_daily_weights(rev,cfg.reverse_mean_target)
    receipt=np.zeros(len(p)+len(rev)+1)
    for b,w in zip(rev,wb): receipt[np.arange(1,len(p)+1)+b]+=p*w
    assert abs(receipt[:16].sum()-.72)<.03      # Fig. 16: 72% received within 15 days
    assert abs(conditional_dispatch_weights(ExperimentConfig(return_window_days=14)).sum()-1)<1e-12


def test_csv_round_trip_no_hidden_marks(tmp_path):
    cfg=ExperimentConfig(season_days=14,mean_weekly_units=5,seed=21)
    hist=generate_history(3,np.random.default_rng(3),cfg,EC)
    sizes=export_history_csv(hist,cfg,EC,tmp_path)
    assert sizes['demand_events']>0
    loaded=load_history_csv(tmp_path,cfg)
    assert [len(h.demand_times) for h in hist]==[len(h.demand_times) for h in loaded]
    assert [len(h.completed_sales) for h in hist]==[len(h.completed_sales) for h in loaded]
    demand=pd.read_csv(tmp_path/'demand_events.csv')
    sales=pd.read_csv(tmp_path/'sale_units.csv')
    assert (demand.demand_qty==demand.sales_qty+demand.lost_sales_qty).all()
    assert (sales.loc[sales.return_status=='rejected_late','refund_issued']==False).all()
    assert not set(['true_return_probability','hidden_mark']).intersection(sales.columns)
    assert not (sales.loc[sales.return_status=='rejected_late','resale_ready_datetime'].notna()).any()
    # both methods need be trained on parsed observations
    saa=empirical_scenarios(loaded,2,np.random.default_rng(7),cfg)
    bay=bayesian_scenarios(loaded,2,np.random.default_rng(8),cfg)
    assert len(saa)==len(bay)==2
    a,b=demand_probability_posterior(loaded,cfg)
    assert a>1 and b>1

def test_reproducibility():
    c=ExperimentConfig(season_days=14,mean_weekly_units=5)
    a=sample_true_scenario(np.random.default_rng(99),c)
    b=sample_true_scenario(np.random.default_rng(99),c)
    assert a==b


def test_negative_binomial_beta_conjugacy_formula():
    """For fixed kappa, Beta posterior has alpha+=n*kappa, beta+=sum(y)."""
    from data_generation import HistoricalSeason
    from scenario_generation import demand_probability_posterior
    cfg=ExperimentConfig(season_days=14,bayesian_assumed_dispersion=4.)
    h=[HistoricalSeason(1,tuple([1.]*3+[8.]*5),tuple(),8,14)]
    assert demand_probability_posterior(h,cfg,prior_alpha=2.,prior_beta=3.) == (10.,11.)


def test_intraperiod_timing_changes_actual_resale():
    early=PotentialMark(True,True,True,1,1,1,1,'too_small')
    late=PotentialMark(True,True,True,1,30,1,1,'too_small')
    sc_early=Scenario((1.,6.),(early,PotentialMark()),20.)
    sc_late=Scenario((1.,6.),(late,PotentialMark()),20.)
    e=simulate(1,sc_early,EC)
    l=simulate(1,sc_late,EC)
    assert (e.sales,e.actual_resales,e.lost_sales)==(2,1,0)
    assert (l.sales,l.actual_resales,l.lost_sales)==(1,0,1)
    assert e.mismatch_cost < l.mismatch_cost

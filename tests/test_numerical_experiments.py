"""Targeted checks for the numerical experiment wrapper (no model change)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from config import Economics,ExperimentConfig
from data_generation import generate_history, sample_true_scenario
from numerical_experiments import _shift_scenario,audit,convergence,timing_sensitivity
from simulation import simulate


def test_timing_changes_do_not_change_return_quantities():
    cfg=ExperimentConfig(season_days=14,mean_weekly_units=3,
                         return_attempt_probability=0.9,return_resalable_probability=0.9)
    sc=sample_true_scenario(np.random.default_rng(11),cfg)
    altered=_shift_scenario(sc,processing_days=5,dispatch_shift=6)
    assert sc.demand_times == altered.demand_times
    for first,second in zip(sc.marks,altered.marks):
        assert (first.attempts_return,first.accepted,first.resalable)==(
            second.attempts_return,second.accepted,second.resalable)
        if first.accepted:
            assert 1 <= second.customer_days <= cfg.return_window_days
    # Physical simulator must still enforce all conservation checks.
    simulate(4,altered,Economics()).check(Economics())


def test_experimental_files_small_smoke(tmp_path):
    cfg=ExperimentConfig(season_days=14,mean_weekly_units=3,saa_scenarios=3,bayes_scenarios=3)
    ec=Economics()
    hist=generate_history(3,np.random.default_rng(102),cfg,ec)
    check=audit(hist,cfg,ec,tmp_path)
    assert check.cost_implementation_verified.iloc[0]
    conv=convergence(hist,cfg,ec,[2,3],22,4,5,tmp_path)
    assert set(conv.scenario_count)=={2,3}
    assert set(conv.method)=={'SAA','Bayesian'}
    intr=timing_sensitivity(hist,cfg,ec,33,4,tmp_path,(1,2),(-3,0,3))
    assert len(intr)==10
    assert set(intr.method)=={'SAA','Bayesian'}
    assert intr.groupby(['factor','value']).size().eq(2).all()
    assert (tmp_path/'audit.csv').exists()
    assert (tmp_path/'mc_convergence.csv').exists()
    assert (tmp_path/'intraperiod_sensitivity.csv').exists()

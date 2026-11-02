"""Regression tests for CSV-first model interfaces. Replace the old test_more.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from config import Economics, ExperimentConfig
from data_generation import generate_history,export_history_csv,load_history_csv,completed_sale_cohort
from scenario_generation import demand_probability_posterior


def test_completed_season_cohorts_survive_csv_roundtrip(tmp_path):
    cfg=ExperimentConfig(season_days=14,mean_weekly_units=4)
    ec=Economics()
    hist=generate_history(2,np.random.default_rng(5),cfg,ec)
    export_history_csv(hist,cfg,ec,tmp_path)
    observed=load_history_csv(tmp_path,cfg)
    assert len(completed_sale_cohort(hist,cfg))==len(completed_sale_cohort(observed,cfg))
    assert sum(len(x.demand_times) for x in hist)==sum(len(x.demand_times) for x in observed)


def test_bayesian_assumed_dispersion_can_differ_from_dgp():
    ec=Economics()
    cfg1=ExperimentConfig(season_days=14,mean_weekly_units=4,weekly_dispersion=5,
                          bayesian_assumed_dispersion=3)
    hist=generate_history(2,np.random.default_rng(7),cfg1,ec)
    a1,b1=demand_probability_posterior(hist,cfg1)
    from dataclasses import replace
    cfg2=replace(cfg1,bayesian_assumed_dispersion=8)
    a2,b2=demand_probability_posterior(hist,cfg2)
    assert b1==b2
    assert a1!=a2

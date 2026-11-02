import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import pytest

from config import Economics, ExperimentConfig
from data_generation import generate_history
from prior_sensitivity_analysis import make_demand_priors, information_counts, paired_intervals, run
from scenario_generation import demand_probability_posterior


def test_hypothetical_prior_has_target_prior_predictive_mean():
    k = 12.
    priors = make_demand_priors(k, [15., 25.], 6.)
    assert priors[0].label == 'beta_1_1'
    for prior, mu in zip(priors[1:], [15., 25.]):
        # E[k(1-phi)/phi] = k*beta/(alpha-1) for alpha>1.
        assert k * prior.beta/(prior.alpha - 1.) == pytest.approx(mu)


def test_prior_conjugate_update_does_not_change_history():
    cfg = ExperimentConfig(mean_weekly_units=8., season_days=14)
    hist = generate_history(2, np.random.default_rng(73), cfg, Economics())
    prior = make_demand_priors(cfg.bayesian_assumed_dispersion, [15.], 6.)[1]
    a, b = demand_probability_posterior(hist, cfg, prior.alpha, prior.beta)
    assert a == pytest.approx(prior.alpha + 4*cfg.bayesian_assumed_dispersion)
    assert b == pytest.approx(prior.beta + sum(len(s.demand_times) for s in hist))


def test_information_is_observed_history_only():
    cfg=ExperimentConfig(mean_weekly_units=8., season_days=14)
    hist=generate_history(2,np.random.default_rng(4),cfg,Economics())
    info=information_counts(hist,cfg)
    assert info['n_history'] == 2
    assert info['n_weekly_demand_observations'] == 4
    assert info['n_potential_demand_units'] >= info['n_fulfilled_sale_units']
    assert info['n_return_resolved_sale_units'] == info['n_fulfilled_sale_units']
    assert info['n_resalability_observations'] == info['n_accepted_return_units']
    assert info['n_accepted_return_units'] <= info['n_return_attempts'] <= info['n_fulfilled_sale_units']


def test_paired_same_history_and_test_csv(tmp_path):
    summary, paired, info = run(sample_sizes=[1], repetitions=2, seed=4242,
                                output_dir=tmp_path,mean_weekly_units=6.,
                                saa_scenarios=3,bayes_scenarios=3,test_scenarios=4)
    assert len(info)==2
    assert len(summary)==4 # SAA + three beta priors
    assert len(paired)==3
    assert (paired.repetitions == 2).all()
    assert (summary.avg_n_weekly_demand == 18.).all()
    assert (tmp_path/'paired_cost_differences_ci95.csv').is_file()
    assert (tmp_path/'paired_prior_vs_beta_1_1.csv').is_file()

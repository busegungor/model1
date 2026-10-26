"""Exploratory sample-size study with *independent* training histories.

The truth here is an illustrative synthetic DGP, not a sector-calibrated model.
For publication-quality results, increase repetitions/scenarios and calibrate inputs.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from config import Economics, ExperimentConfig
from data_generation import generate_history, sample_true_scenario
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from solver_core import optimize
from evaluation import evaluate


def run(sample_sizes=(5,10,20,40), repetitions=3, seed=20261026, output_dir='outputs'):
    cfg=ExperimentConfig()
    ec=Economics()
    rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)]
    reference=[sample_true_scenario(rngs[0],cfg) for _ in range(cfg.reference_scenarios)]
    oracle=optimize('reference_MC_oracle',reference,ec)
    test=[sample_true_scenario(rngs[1],cfg) for _ in range(cfg.evaluation_scenarios)]
    oracle_test=evaluate(oracle.q,test,ec)['mean_mismatch_cost']
    rows=[]
    for n in sample_sizes:
        for repetition in range(1,repetitions+1):
            # Each (n,repetition) must have an independent history/optimizer stream.
            # Stable seed incorporates n, unlike simply reusing the same repetitions.
            ss=np.random.SeedSequence([seed+1, int(n), int(repetition)])
            hg,sg,bg=[np.random.default_rng(s) for s in ss.spawn(3)]
            hist=generate_history(n,hg,cfg,ec)
            for sol in (solve_saa(hist,sg,cfg,ec),solve_bayesian(hist,bg,cfg,ec)):
                record=evaluate(sol.q,test,ec)
                rows.append(dict(method=sol.method,n_history=n,repetition=repetition,
                    **record,regret_vs_reference_MC_oracle=record['mean_mismatch_cost']-oracle_test))
    folder=Path(output_dir)
    folder.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(folder/'sample_size_repetitions.csv',index=False)
    agg=df.groupby(['n_history','method'],as_index=False).agg(
        avg_test_cost=('mean_mismatch_cost','mean'),
        sd_test_cost=('mean_mismatch_cost','std'),
        avg_regret=('regret_vs_reference_MC_oracle','mean'),
        avg_order=('order','mean'),
        avg_fill_rate=('fill_rate','mean'))
    agg.to_csv(folder/'sample_size_summary.csv',index=False)
    print(agg.to_string(index=False))
    return agg


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--sample-sizes',nargs='+',type=int,default=[5,10,20,40])
    ap.add_argument('--repetitions',type=int,default=3)
    ap.add_argument('--seed',type=int,default=20261026)
    ap.add_argument('--output-dir',default='outputs')
    a=ap.parse_args()
    run(a.sample_sizes,a.repetitions,a.seed,a.output_dir)
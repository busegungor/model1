"""Independent training histories, fixed holdout; n = completed seasons, NOT CSV lines.
Use a smaller mean weekly volume for rapid numerical prototyping.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from config import ExperimentConfig,Economics
from data_generation import generate_history,sample_true_scenario,export_history_csv,load_history_csv
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from solver_core import optimize
from evaluation import evaluate


def run(sample_sizes=(2,5,10,20),repetitions=3,seed=20261029,output_dir='outputs',
        mean_weekly_units=150.,processing_days=2,season_days=126,
        saa_scenarios=20,bayes_scenarios=20,evaluation_scenarios=40,reference_scenarios=60):
    cfg=ExperimentConfig(mean_weekly_units=mean_weekly_units,
                         processing_days=processing_days,season_days=season_days,
                         saa_scenarios=saa_scenarios,bayes_scenarios=bayes_scenarios,
                         evaluation_scenarios=evaluation_scenarios,reference_scenarios=reference_scenarios)
    ec=Economics()
    a,b=np.random.SeedSequence(seed).spawn(2)
    ga,gb=np.random.default_rng(a),np.random.default_rng(b)
    reference=[sample_true_scenario(ga,cfg) for _ in range(cfg.reference_scenarios)]
    oracle=optimize('reference_MC',reference,ec)
    test=[sample_true_scenario(gb,cfg) for _ in range(cfg.evaluation_scenarios)]
    oracle_test=evaluate(oracle.q,test,ec)['mean_mismatch_cost']
    rows=[]
    for n in sample_sizes:
        for rep in range(1,repetitions+1):
            rng=[np.random.default_rng(s) for s in np.random.SeedSequence([seed+7,int(n),rep]).spawn(3)]
            hist=generate_history(n,rng[0],cfg,ec)
            # Use the exact observable CSV structure employed by the main pipeline.
            train_dir=Path(output_dir)/'independent_training_csv'/f'n{n}_rep{rep}'
            export_history_csv(hist,cfg,ec,train_dir)
            hist=load_history_csv(train_dir,cfg)
            sold=sum(len(s.completed_sales) for s in hist)
            returned=sum(r.mark.accepted for s in hist for r in s.completed_sales)
            for sol in (solve_saa(hist,rng[1],cfg,ec),solve_bayesian(hist,rng[2],cfg,ec)):
                out=evaluate(sol.q,test,ec)
                rows.append(dict(n_history=n,repetition=rep,n_sale_units=sold,
                                 n_accepted_return_units=returned,method=sol.method,**out,
                                 regret_vs_reference_MC=out['mean_mismatch_cost']-oracle_test))
    target=Path(output_dir);target.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(target/'sample_size_repetitions.csv',index=False)
    summary=df.groupby(['n_history','method'],as_index=False).agg(
        avg_test_cost=('mean_mismatch_cost','mean'),sd_test_cost=('mean_mismatch_cost','std'),
        avg_regret=('regret_vs_reference_MC','mean'),avg_order=('order','mean'),
        avg_n_sale_units=('n_sale_units','mean'),avg_n_accepted_returns=('n_accepted_return_units','mean'))
    summary.to_csv(target/'sample_size_summary.csv',index=False)
    print(summary.round(2).to_string(index=False))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--sample-sizes',type=int,nargs='+',default=[2,5,10,20])
    ap.add_argument('--repetitions',type=int,default=3)
    ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--output-dir',default='outputs')
    ap.add_argument('--mean-weekly-units',type=float,default=150.)
    ap.add_argument('--processing-days',type=int,default=2)
    ap.add_argument('--season-days',type=int,default=126)
    ap.add_argument('--saa-scenarios',type=int,default=20)
    ap.add_argument('--bayes-scenarios',type=int,default=20)
    ap.add_argument('--evaluation-scenarios',type=int,default=40)
    ap.add_argument('--reference-scenarios',type=int,default=60)
    a=ap.parse_args()
    run(a.sample_sizes,a.repetitions,a.seed,a.output_dir,a.mean_weekly_units,a.processing_days,
        a.season_days,a.saa_scenarios,a.bayes_scenarios,a.evaluation_scenarios,a.reference_scenarios)

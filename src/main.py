"""Reproducible demonstration (not empirical results for the paper)."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
from config import ExperimentConfig, Economics
from data_generation import generate_history, history_records, sample_true_scenario
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from solver_core import optimize
from evaluation import evaluate


def run(n_history:int=24, seed:int=20261026, output_dir:str='outputs'):
    cfg=ExperimentConfig(seed=seed)
    ec=Economics()
    outdir=Path(output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    # Independent RNG streams: history, SAA, Bayes, oracle reference, test.
    gens=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    history=generate_history(n_history,gens[0],cfg,ec)
    pd.DataFrame(history_records(history,cfg)).to_csv(outdir/'historical_sales.csv',index=False)
    pd.DataFrame([dict(season=s.season,j=j+1,demand_timestamp=time)
                  for s in history for j,time in enumerate(s.demand_times)]).to_csv(
                  outdir/'fully_observed_demand.csv',index=False)
    saa=solve_saa(history,gens[1],cfg,ec)
    bayes=solve_bayesian(history,gens[2],cfg,ec)
    reference=[sample_true_scenario(gens[3],cfg) for _ in range(cfg.reference_scenarios)]
    oracle=optimize('reference_MC_oracle',reference,ec)
    test=[sample_true_scenario(gens[4],cfg) for _ in range(cfg.evaluation_scenarios)]
    rows=[]
    for sol in (saa,bayes,oracle):
        metrics=evaluate(sol.q,test,ec)
        rows.append(dict(method=sol.method,n_history=n_history,seed=seed,
                         fitted_objective=sol.estimated_cost,**metrics))
    df=pd.DataFrame(rows)
    oracle_test=float(df.loc[df.method=='reference_MC_oracle','mean_mismatch_cost'].iloc[0])
    df['regret_vs_reference_MC_oracle']=df['mean_mismatch_cost']-oracle_test
    df.to_csv(outdir/'comparison.csv',index=False)
    pd.DataFrame(dict(Q=saa.q_grid,SAA=saa.cost_curve)).to_csv(outdir/'saa_cost_curve.csv',index=False)
    pd.DataFrame(dict(Q=bayes.q_grid,Bayesian=bayes.cost_curve)).to_csv(outdir/'bayesian_cost_curve.csv',index=False)
    print(df[['method','order','mean_mismatch_cost','fill_rate','regret_vs_reference_MC_oracle']].to_string(index=False))
    print(f'\nResults written to {outdir.resolve()}')
    return df


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n-history',type=int,default=24)
    ap.add_argument('--seed',type=int,default=20261026)
    ap.add_argument('--output-dir',default='outputs')
    args=ap.parse_args()
    run(args.n_history,args.seed,args.output_dir)
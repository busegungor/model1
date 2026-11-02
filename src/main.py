"""CSV-first DGP; train solely from exported observations, evaluate on new truth."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from config import ExperimentConfig,Economics
from data_generation import generate_history,export_history_csv,load_history_csv,sample_true_scenario
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from solver_core import optimize
from evaluation import evaluate


def run(n_history=5,seed=20261029,output_dir='outputs',input_dir=None,generate_only=False,
        season_days=126,mean_weekly_units=150,processing_days=2):
    cfg=ExperimentConfig(seed=seed,season_days=season_days,
        mean_weekly_units=mean_weekly_units,processing_days=processing_days)
    ec=Economics()
    dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    rngs=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(5)]
    if input_dir:
        history=load_history_csv(input_dir,cfg)
        print(f'Loaded {len(history)} complete historical seasons from {input_dir}')
    else:
        history=generate_history(n_history,rngs[0],cfg,ec)
        sizes=export_history_csv(history,cfg,ec,dest/'historical_data')
        # Always round-trip CSV: optimization uses observation files, not truth objects.
        history=load_history_csv(dest/'historical_data',cfg)
        print('Wrote synthetic historical CSVs:',sizes)
    if generate_only: return history
    saa=solve_saa(history,rngs[1],cfg,ec)
    bayes=solve_bayesian(history,rngs[2],cfg,ec)
    reference=[sample_true_scenario(rngs[3],cfg,i+1) for i in range(cfg.reference_scenarios)]
    oracle=optimize('reference_MC',reference,ec)
    test=[sample_true_scenario(rngs[4],cfg,i+1) for i in range(cfg.evaluation_scenarios)]
    rows=[]
    for sol in (saa,bayes,oracle):
        rows.append(dict(method=sol.method,n_history=len(history),seed=seed,
                         fitted_objective=sol.estimated_cost,**evaluate(sol.q,test,ec)))
    frame=pd.DataFrame(rows)
    target=float(frame.loc[frame.method=='reference_MC','mean_mismatch_cost'].iloc[0])
    frame['regret_vs_reference_MC']=frame.mean_mismatch_cost-target
    frame.to_csv(dest/'comparison.csv',index=False)
    for sol in (saa,bayes):
        pd.DataFrame({'Q':sol.q_grid,'cost':sol.cost_curve}).to_csv(dest/(sol.method.lower()+'_cost_curve.csv'),index=False)
    print(frame[['method','order','mean_mismatch_cost','regret_vs_reference_MC',
                 'mean_resales_actual','mean_rejected_late_attempts']].round(3).to_string(index=False))
    return frame

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--n-history',type=int,default=5)
    ap.add_argument('--seed',type=int,default=20261029)
    ap.add_argument('--output-dir',default='outputs')
    ap.add_argument('--input-dir')
    ap.add_argument('--generate-only',action='store_true')
    ap.add_argument('--season-days',type=int,default=126)
    ap.add_argument('--mean-weekly-units',type=float,default=150)
    ap.add_argument('--processing-days',type=int,default=2)
    args=ap.parse_args()
    run(args.n_history,args.seed,args.output_dir,args.input_dir,args.generate_only,
        args.season_days,args.mean_weekly_units,args.processing_days)

"""
main.py — Integration test / smoke test for the full pipeline.

Runs the entire SAA + Bayesian + benchmark pipeline end-to-end for ONE
period (W) and all 5 categories, verifying that every module hands off
correctly to the next:

    config -> data_analysis -> preprocessing -> parameter_estimation
           -> monte_carlo (SAA + Bayesian) -> saa_solver / bayesian_solver
           -> benchmark

Purpose is VERIFICATION, not full analysis. The full multi-period grid
(W/2W/3W/4W x m_train x rules x methods) belongs to the comparison step
(Step 15); parameter sweeps belong to sensitivity (Step 11). Once those
exist, this file can be extended into a full orchestrator.

Run:  python main.py
"""

import numpy as np

from config import COST_PARAMS, MC_SETTINGS
from gurobi_env import make_gurobi_env
from data_analysis import order_items_clean, returns_clean
from preprocessing import build_category_datasets, summarize_datasets
from parameter_estimation import build_category_params
from monte_carlo import (build_net_demand_matrix,
                         build_bayesian_net_demand_matrix,
                         summarize_matrix)
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from benchmark import netting_benchmark
from bayesian_models import build_posteriors


def run_integration_test(period="W", rule="S4", J=None, seed=0):
    """Run the full pipeline for one period and report per-category results."""
    if J is None:
        J = MC_SETTINGS["J_baseline"]

    print("=" * 78)
    print(f"INTEGRATION TEST — period={period}, rule={rule}, J={J}")
    print("=" * 78)

    # ── Step 1: Gurobi env (optional) ────────────────────────────────
    env = make_gurobi_env()
    print(f"\n[1] Gurobi env: {'available' if env else 'None -> quantile fallback'}")

    # ── Step 2: Preprocessing ────────────────────────────────────────
    print(f"\n[2] Preprocessing (cohort fix, dedup, unit-level timestamps)...")
    datasets = build_category_datasets(order_items_clean, returns_clean,
                                       period=period)
    print(f"    Categories built: {len(datasets)}")
    summarize_datasets(datasets)   # prints per-category panel summary

    # ── Step 3: Parameter estimation ─────────────────────────────────
    print(f"\n[3] Parameter estimation (rates, prices, cost params)...")
    params = build_category_params(datasets, order_items_clean)
    all_well_posed = all(p["well_posed"] for p in params.values())
    print(f"    All categories well-posed: {all_well_posed}")
    if not all_well_posed:
        print("    WARNING: some categories failed the well-posed check!")


 # ── Steps 4-7: per-category, ALL rules, SAA / Bayesian / benchmark ─
    print(f"\n[4-7] All sequencing rules (S1, S3, S4) — SAA vs Bayesian:")

    rules = ["S1", "S3", "S4"]
    results = {}

    for cat, ds in datasets.items():
        prm = params[cat]
        results[cat] = {}

        # Deterministic benchmark (rule-independent, SAA side)
        bench = netting_benchmark(ds["panel"], prm["r_hat"], prm["k_hat"],
                                  prm["c"], prm["s"], prm["pN"], prm["gN"],
                                  gurobi_env=env)
        results[cat]["bench"] = bench

        for rule in rules:
            N_saa = build_net_demand_matrix(ds, prm["r_hat"], prm["k_hat"],
                                            rule=rule, J=J, seed=seed)
            saa = solve_saa(N_saa, prm["c"], prm["s"], prm["pN"], prm["gN"],
                            gurobi_env=env)
            N_bay = build_bayesian_net_demand_matrix(ds, rule=rule, J=J, seed=seed)
            bay = solve_bayesian(N_bay, prm["c"], prm["s"], prm["pN"], prm["gN"],
                                 gurobi_env=env)
            results[cat][rule] = {
                "saa": saa, "bay": bay,
                "saa_meanN": float(N_saa.mean()),
                "bay_meanN": float(N_bay.mean()),
            }

    # Print per-rule table
    for rule in rules:
        print("-" * 78)
        print(f"  Rule {rule}:")
        print(f"  {'Category':<22}{'SAA_Q':>7}{'Bay_Q':>7}"
              f"{'SAA_cost':>10}{'Bay_cost':>10}{'SAA_meanN':>11}")
        for cat in datasets:
            r = results[cat][rule]
            print(f"  {cat:<22}{r['saa']['Q_opt']:>7.1f}{r['bay']['Q_opt']:>7.1f}"
                  f"{r['saa']['obj_opt']:>10.2f}{r['bay']['obj_opt']:>10.2f}"
                  f"{r['saa_meanN']:>11.2f}")

    # ── Consistency checks ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("CONSISTENCY CHECKS")
    print("=" * 78)

    checks = []

    # Check 1: all Q* finite and positive (all rules, both methods)
    all_q_valid = all(
        np.isfinite(results[cat][rule][m]["Q_opt"]) and results[cat][rule][m]["Q_opt"] > 0
        for cat in datasets for rule in rules for m in ("saa", "bay")
    )
    checks.append(("All Q* finite and positive (all rules)", all_q_valid))

    # Check 2: all costs finite and positive
    all_cost_valid = all(
        np.isfinite(results[cat][rule][m]["obj_opt"]) and results[cat][rule][m]["obj_opt"] > 0
        for cat in datasets for rule in rules for m in ("saa", "bay")
    )
    checks.append(("All costs finite and positive (all rules)", all_cost_valid))

    # Check 3: critical ratio alpha in (0,1)
    alpha_ok = all(
        0 < results[cat]["S4"]["saa"].get("alpha", -1) < 1
        for cat in datasets
    )
    checks.append(("Critical ratio alpha in (0,1)", alpha_ok))

    # Check 4: sequencing SPECTRUM  S1 <= S4 <= S3  (mean net demand)
    # S1 = aggregate netting (lowest), S3 = demand-only (highest),
    # S4 = empirical order-of-arrival (in between). Tolerance for MC noise.
    tol = 0.5
    spectrum_saa = all(
        results[cat]["S1"]["saa_meanN"] <= results[cat]["S4"]["saa_meanN"] + tol
        and results[cat]["S4"]["saa_meanN"] <= results[cat]["S3"]["saa_meanN"] + tol
        for cat in datasets
    )
    checks.append(("Sequencing spectrum S1<=S4<=S3 (SAA mean net demand)", spectrum_saa))

    spectrum_bay = all(
        results[cat]["S1"]["bay_meanN"] <= results[cat]["S4"]["bay_meanN"] + tol
        and results[cat]["S4"]["bay_meanN"] <= results[cat]["S3"]["bay_meanN"] + tol
        for cat in datasets
    )
    checks.append(("Sequencing spectrum S1<=S4<=S3 (Bayesian mean net demand)", spectrum_bay))

    # Check 5: SAA and Bayesian Q* within 2x (all rules)
    ballpark_ok = all(
        0.5 < results[cat][rule]["bay"]["Q_opt"] / results[cat][rule]["saa"]["Q_opt"] < 2.0
        for cat in datasets for rule in rules
    )
    checks.append(("SAA and Bayesian Q* within 2x (all rules)", ballpark_ok))

    for desc, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {desc}")

    all_pass = all(ok for _, ok in checks)
    print("\n" + ("ALL CHECKS PASSED — pipeline is integrated correctly."
                  if all_pass else
                  "SOME CHECKS FAILED — inspect the modules above."))
    return results


if __name__ == "__main__":
    run_integration_test(period="W", rule="S4")
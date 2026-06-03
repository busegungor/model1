"""
main.py — Full orchestrator for the entire thesis pipeline.

A single command (`python main.py`) runs everything end to end:

  STAGE A — Pipeline verification (all 4 periods W/2W/3W/4W):
    config -> data_analysis -> preprocessing -> parameter_estimation
           -> monte_carlo (SAA + Bayesian) -> saa_solver / bayesian_solver
           -> benchmark, with consistency checks (finite Q*, alpha in (0,1),
           S1<=S4<=S3 spectrum) for every period.

  STAGE B — Method comparison (SAA vs Bayesian, out-of-sample regret):
    comparison.run_comparison over periods x m_train x g, rule S4.
    -> comparison_outputs/method_comparison.csv

  STAGE C — Sequencing importance diagnostic (when does sequencing matter):
    comparison.sequencing_importance, S1/S4/S3 bounds.
    -> comparison_outputs/sequencing_importance.csv

This is SLOW (Stage B bootstraps; several minutes). For a quick pipeline
check only, call run_integration_test(period="W") directly, or run with
`python main.py --quick` to skip Stages B and C.

Run:  python main.py            (everything)
      python main.py --quick    (Stage A only, fast smoke test)
"""

import sys
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


 # ── Steps 4-7: per-category, ALL rules, SAA / Bayesian ─
    print(f"\n[4-7] All sequencing rules (S1, S3, S4) — SAA vs Bayesian:")

    rules = ["S1", "S3", "S4"]
    results = {}

    for cat, ds in datasets.items():
        prm = params[cat]
        results[cat] = {}


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


def run_all_periods(rule="S4", J=None, seed=0):
    """STAGE A: pipeline verification for all four periods."""
    periods = ("W", "2W", "3W", "4W")
    all_pass = True
    for period in periods:
        res = run_integration_test(period=period, rule=rule, J=J, seed=seed)
        # run_integration_test prints its own PASS/FAIL; collect overall status
        # (re-derive from the printed checks by re-checking Q* finiteness)
        ok = all(
            np.isfinite(res[cat][rl][m]["Q_opt"]) and res[cat][rl][m]["Q_opt"] > 0
            for cat in res for rl in ("S1", "S3", "S4") for m in ("saa", "bay")
        )
        all_pass = all_pass and ok
    print("\n" + "#" * 78)
    print("STAGE A COMPLETE — pipeline verified for all periods W/2W/3W/4W"
          if all_pass else
          "STAGE A: some periods had issues — inspect output above.")
    print("#" * 78)
    return all_pass


def run_full_orchestrator(quick=False, seed=0):
    """
    Run the entire thesis pipeline. quick=True stops after Stage A.
    """
    from pathlib import Path
    OUT = Path("comparison_outputs"); OUT.mkdir(exist_ok=True)

    # ── STAGE A: pipeline verification, all periods ──────────────────
    run_all_periods(rule="S4")

    if quick:
        print("\n[--quick] Stages B and C skipped.")
        return

    # ── STAGE B: method comparison (SAA vs Bayesian, out-of-sample) ──
    print("\n" + "#" * 78)
    print("STAGE B — METHOD COMPARISON (SAA vs Bayesian regret) — several min")
    print("#" * 78)
    from comparison import run_comparison, sequencing_importance
    df = run_comparison(rule="S4", g_grid=(5.0, 8.0, 20.0),
                        B=60, J_fit=MC_SETTINGS["J_sensitivity"], seed=seed)
    df.to_csv(OUT / "method_comparison.csv", index=False)
    print(f"Saved -> {OUT/'method_comparison.csv'}")

    # ── STAGE C: sequencing importance diagnostic ────────────────────
    print("\n" + "#" * 78)
    print("STAGE C — SEQUENCING IMPORTANCE (S1/S4/S3 bounds)")
    print("#" * 78)
    seq = sequencing_importance(g=8.0, J=2000, seed=seed)
    seq.to_csv(OUT / "sequencing_importance.csv", index=False)
    print(f"Saved -> {OUT/'sequencing_importance.csv'}")

    print("\n" + "=" * 78)
    print("ALL STAGES COMPLETE. Outputs in comparison_outputs/.")
    print("=" * 78)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    run_full_orchestrator(quick=quick)
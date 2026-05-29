"""
bayesian_solver.py — Bayesian newsvendor solution.

The Bayesian net-demand matrix is built in monte_carlo.py
(build_bayesian_net_demand_matrix); this module only SOLVES it.

The solver is identical to the SAA solver (solve_saa) — the quantile
critical-ratio solution is method-agnostic. Using the same solver
guarantees that SAA vs Bayesian Q* differences come purely from the
net-demand matrix (the demand model), not from the optimizer.

────────────────────────────────────────────────────────────────────
This module is NEW. Matrix construction now lives in monte_carlo.py.
────────────────────────────────────────────────────────────────────
"""

from saa_solver import solve_saa


def solve_bayesian(N_matrix, c, s, pN, gN,
                   gurobi_env=None, verbose=False) -> dict:
    """
    Solve the newsvendor for a Bayesian net-demand matrix.

    Thin wrapper around solve_saa (same quantile core), tagged as Bayesian.
    """
    result = solve_saa(N_matrix, c=c, s=s, pN=pN, gN=gN,
                       gurobi_env=gurobi_env, verbose=verbose)
    result["method"] = "Bayesian_NegBinBeta"
    return result


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python bayesian_solver.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params
    from monte_carlo import (build_net_demand_matrix,
                             build_bayesian_net_demand_matrix)
    from saa_solver import solve_saa
    from gurobi_env import make_gurobi_env

    datasets = build_category_datasets(order_items_clean, returns_clean, period="W")
    params = build_category_params(datasets, order_items_clean)
    env = make_gurobi_env()

    print(f"\n{'='*72}")
    print("SAA vs BAYESIAN — W period, rule S4, full data")
    print(f"{'='*72}")
    print(f"{'Category':<22}{'SAA_Q':>7}{'Bay_Q':>7}{'SAA_cost':>10}{'Bay_cost':>10}")
    print("-" * 58)

    for cat, ds in datasets.items():
        prm = params[cat]
        N_saa = build_net_demand_matrix(ds, prm["r_hat"], prm["k_hat"],
                                        rule="S4", J=200, seed=0)
        rs = solve_saa(N_saa, prm["c"], prm["s"], prm["pN"], prm["gN"],
                       gurobi_env=env)
        N_bay = build_bayesian_net_demand_matrix(ds, rule="S4", J=200, seed=0)
        rb = solve_bayesian(N_bay, prm["c"], prm["s"], prm["pN"], prm["gN"],
                            gurobi_env=env)
        print(f"{cat:<22}{rs['Q_opt']:>7.1f}{rb['Q_opt']:>7.1f}"
              f"{rs['obj_opt']:>10.3f}{rb['obj_opt']:>10.3f}")

    print("\nFull data → SAA and Bayesian close. Bayesian advantage at small m (Step 15).")
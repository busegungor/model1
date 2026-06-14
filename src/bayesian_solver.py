"""
bayesian_solver.py — Bayesian newsvendor solution.

The Bayesian net-demand matrix is built in monte_carlo.py
(build_bayesian_net_demand_matrix); this module only SOLVES it.

The solver is identical to the SAA solver (solve_saa) — the quantile
critical-ratio solution is method-agnostic. Using the same solver
guarantees that SAA vs Bayesian Q* differences come purely from the
net-demand matrix (the demand model), not from the optimizer. In
particular both methods now share the SAME PRIMITIVE critical ratio
alpha = (p+g-c)/(p+g-s) (model_body §3.6): the only object differing
between the methods is the net-demand sample N.

────────────────────────────────────────────────────────────────────
This module is NEW. Matrix construction now lives in monte_carlo.py.
v15: signature switched to the primitive (p, c, s, g) to match solve_saa.
────────────────────────────────────────────────────────────────────
"""

from saa_solver import solve_saa


def solve_bayesian(N_matrix, p, c, s, g,
                   gurobi_env=None, verbose=False) -> dict:
    """
    Solve the newsvendor for a Bayesian net-demand matrix.

    Thin wrapper around solve_saa (same primitive quantile core), tagged as
    Bayesian. Takes the primitive market constants (p, c, s, g); the return
    parameters r, k are already baked into N_matrix, not the cost ratio.
    """
    result = solve_saa(N_matrix, p=p, c=c, s=s, g=g,
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
        rs = solve_saa(N_saa, prm["p"], prm["c"], prm["s"], prm["g"],
                       gurobi_env=env)
        N_bay = build_bayesian_net_demand_matrix(ds, rule="S4", J=200, seed=0)
        rb = solve_bayesian(N_bay, prm["p"], prm["c"], prm["s"], prm["g"],
                            gurobi_env=env)
        print(f"{cat:<22}{rs['Q_opt']:>7.1f}{rb['Q_opt']:>7.1f}"
              f"{rs['obj_opt']:>10.3f}{rb['obj_opt']:>10.3f}")

    print("\nFull data → SAA and Bayesian close. Bayesian advantage at small m (Step 15).")
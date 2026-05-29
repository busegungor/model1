"""
saa_solver.py — Sample Average Approximation solver for the newsvendor.

Given a net-demand matrix N[t, j] (Step 7) and cost parameters (Step 5),
find the order quantity Q* minimizing expected cost:

    EP(Q) = cu * E[(N - Q)^+] + co * E[(Q - N)^+]

where the per-unit underage / overage costs are
    cu = pN - c + gN        (lost a net-demand sale)
    co = c  - s             (leftover unit salvaged)

Two equivalent solvers:
  - Gurobi LP  : the textbook SAA linear program (used if a license env
                 is supplied; otherwise skipped).
  - Quantile   : the closed-form newsvendor critical-ratio solution,
                 Q* = alpha-quantile of the flattened N sample, where
                 alpha = cu / (cu + co). Mathematically identical to the
                 LP (Zipkin 2000, Prop. 2.1); O(mJ log mJ), no solver.

The quantile path is the default (fast, exact, no memory blow-up).

────────────────────────────────────────────────────────────────────
Colab origin: _solve_with_gurobi / _solve_with_quantile / solve_saa,
              lines 490-569 (verbatim logic; print statements made
              optional via a verbose flag).
────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np


def _solve_with_gurobi(N_flat, c, s, pN, gN, Q_max, env) -> dict:
    """
    SAA via Gurobi LP.

    minimize  (cu/mJ) Σ z_i + (co/mJ) Σ v_i
    s.t.      z_i >= N_i - Q,   v_i >= Q - N_i,   z,v >= 0,   0 <= Q <= Q_max

    Colab origin: lines 490-518.
    """
    import gurobipy as gp
    from gurobipy import GRB

    mJ = len(N_flat)
    cu = pN - c + gN
    co = c - s

    model = gp.Model("newsvendor_saa", env=env)
    model.setParam("OutputFlag", 0)

    Q = model.addVar(lb=0.0, ub=Q_max, name="Q")
    z = model.addVars(mJ, lb=0.0, name="z")
    v = model.addVars(mJ, lb=0.0, name="v")

    for i in range(mJ):
        model.addConstr(z[i] >= N_flat[i] - Q)
        model.addConstr(v[i] >= Q - N_flat[i])

    model.setObjective(
        (cu / mJ) * gp.quicksum(z[i] for i in range(mJ))
        + (co / mJ) * gp.quicksum(v[i] for i in range(mJ)),
        GRB.MINIMIZE,
    )
    model.optimize()

    Q_opt = float(Q.X)
    obj_opt = float(model.ObjVal)
    ESN = float(np.mean(np.maximum(N_flat - Q_opt, 0)))
    model.dispose()
    return {"Q_opt": Q_opt, "obj_opt": obj_opt, "ESN": ESN,
            "solver": "Gurobi WLS academic"}


def _solve_with_quantile(N_flat, c, s, pN, gN) -> dict:
    """
    Exact SAA solution via the newsvendor critical-ratio formula.

        alpha = cu / (cu + co) = (pN - c + gN) / (pN - s + gN)
        Q*    = alpha-quantile of the empirical net-demand sample

    Mathematically identical to the LP (Zipkin 2000, Prop. 2.1).

    Colab origin: lines 523-545.
    """
    cu = pN - c + gN
    co = c - s
    alpha = cu / (cu + co)

    Q_opt = float(np.quantile(N_flat, alpha))
    ESN = float(np.mean(np.maximum(N_flat - Q_opt, 0)))
    obj_opt = float(cu * ESN + co * np.mean(np.maximum(Q_opt - N_flat, 0)))

    return {"Q_opt": Q_opt, "obj_opt": obj_opt, "ESN": ESN,
            "alpha": alpha,
            "solver": f"Quantile (alpha={alpha:.4f})"}


def solve_saa(
    N_matrix: np.ndarray,
    c: float,
    s: float,
    pN: float,
    gN: float,
    gurobi_env=None,
    Q_max: float = None,
    verbose: bool = False,
) -> dict:
    """
    Solve the SAA newsvendor for a net-demand matrix.

    Parameters
    ----------
    N_matrix : np.ndarray, shape (m, J)
        Net-demand realizations from build_net_demand_matrix.
    c, s, pN, gN : float
        Cost parameters (parameter_estimation.compute_cost_params).
    gurobi_env : gp.Env or None
        If provided, use the Gurobi LP; otherwise the quantile solver.
    Q_max : float, optional
        Upper bound for Q in the LP (default 2 * max(N)). Unused by quantile.
    verbose : bool
        Print a short summary if True.

    Returns
    -------
    dict with keys: Q_opt, obj_opt, ESN, solver (+ alpha for quantile).

    Notes
    -----
    Both solvers operate on the FLATTENED matrix (m*J scenarios), treating
    every (period, replication) pair as an equally-weighted demand scenario.
    """
    m, J = N_matrix.shape
    N_flat = N_matrix.flatten()
    if Q_max is None:
        Q_max = float(N_flat.max()) * 2.0 if N_flat.max() > 0 else 1.0

    if gurobi_env is not None:
        try:
            result = _solve_with_gurobi(N_flat, c, s, pN, gN, Q_max, gurobi_env)
        except Exception as e:
            warnings.warn(f"Gurobi failed ({e}) -> quantile fallback.")
            result = _solve_with_quantile(N_flat, c, s, pN, gN)
    else:
        result = _solve_with_quantile(N_flat, c, s, pN, gN)

    # alpha (critical ratio) is solver-independent — always report it
    if "alpha" not in result:
        cu = pN - c + gN
        co = c - s
        result["alpha"] = cu / (cu + co)

        
    if verbose:
        print(f"  Solver : {result['solver']}   m={m} J={J} mJ={m*J}")
        print(f"  Q*     = {result['Q_opt']:.4f}")
        print(f"  Cost   = {result['obj_opt']:.4f}")
        print(f"  ESN    = {result['ESN']:.4f}")



    return result


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python saa_solver.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params
    from monte_carlo import build_net_demand_matrix
    from gurobi_env import make_gurobi_env

    datasets = build_category_datasets(order_items_clean, returns_clean, period="W")
    params = build_category_params(datasets, order_items_clean)
    env = make_gurobi_env()   # None if no license → quantile path

    print(f"\n{'='*70}")
    print("SAA Q* per category (W period, rule S4)")
    print(f"{'='*70}")
    print(f"{'Category':<24} {'alpha':>7} {'Q*':>7} {'Cost':>9} {'ESN':>7}")
    print("-" * 60)

    for cat, ds in datasets.items():
        prm = params[cat]
        N = build_net_demand_matrix(
            ds, prm["r_hat"], prm["k_hat"], rule="S4", J=200, seed=0
        )
        res = solve_saa(
            N, c=prm["c"], s=prm["s"], pN=prm["pN"], gN=prm["gN"],
            gurobi_env=env,
        )
        alpha = res.get("alpha", float("nan"))
        print(f"{cat:<24} {alpha:>7.4f} {res['Q_opt']:>7.2f} "
              f"{res['obj_opt']:>9.4f} {res['ESN']:>7.4f}")
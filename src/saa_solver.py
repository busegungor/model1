"""
saa_solver.py — Sample Average Approximation solver for the newsvendor.

Given a net-demand matrix N[t, j] (Step 7) and cost parameters (Step 5),
find the order quantity Q* minimizing expected cost:

    EP(Q) = cu * E[(N - Q)^+] + co * E[(Q - N)^+]

where the per-unit underage / overage costs are the PRIMITIVE market
constants (model_body Eq. 8-10):
    cu = p + g - c          (lost a net-demand sale: foregone margin + penalty)
    co = c - s              (leftover line salvaged)
    alpha = cu / (cu + co) = (p + g - c) / (p + g - s)

PRIMITIVE CRITICAL RATIO (model_body Eq. 10). The return parameters r, k do
NOT enter alpha: net demand N already carries recapture in the quantity
dimension (Lindley running-max, Lemma 1), so pricing the marginal unit at
the primitive margin is exact under the exogenous-return premise
(Assumption 7). Scaling p, g by 1/(1-rk) — the Mostard-Teunter transformed
ratio (model_body Eq. 11, alpha_MT) — would double-count recapture; it is
kept ONLY as a comparison object (Prop 3) in _solve_with_quantile_mt and is
NOT the operating solver. In this data regime (rk ~ 0.025) the two ratios
agree to the 4th decimal and give the same integer Q* (verifies Eq. 12's
O(r) gap empirically).

Two equivalent solvers:
  - Gurobi LP  : the textbook SAA linear program (used if a license env
                 is supplied; otherwise skipped).
  - Quantile   : the closed-form newsvendor critical-ratio solution,
                 Q* = alpha-quantile of the flattened N sample, where
                 alpha = (p+g-c)/(p+g-s). Mathematically identical to the
                 LP (Zipkin 2000, Prop. 2.1); O(mJ log mJ), no solver.

The quantile path is the default (fast, exact, no memory blow-up).

────────────────────────────────────────────────────────────────────
Colab origin: _solve_with_gurobi / _solve_with_quantile / solve_saa,
              lines 490-569 (logic preserved; the critical ratio switched
              from the MT-scaled (pN,gN) form to the primitive (p,g) form
              to match model_body Eq. 10; MT form retained as a comparison).
────────────────────────────────────────────────────────────────────
"""

import warnings
import numpy as np


def _solve_with_gurobi(N_flat, p, c, s, g, Q_max, env) -> dict:
    """
    SAA via Gurobi LP, PRIMITIVE critical ratio.

    minimize  (cu/mJ) Σ z_i + (co/mJ) Σ v_i
    s.t.      z_i >= N_i - Q,   v_i >= Q - N_i,   z,v >= 0,   0 <= Q <= Q_max
    with      cu = p + g - c,   co = c - s.

    Colab origin: lines 490-518 (cu/co switched to primitive form).
    """
    import gurobipy as gp
    from gurobipy import GRB

    mJ = len(N_flat)
    cu = p + g - c
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


def _solve_with_quantile(N_flat, p, c, s, g) -> dict:
    """
    Exact SAA solution via the PRIMITIVE newsvendor critical-ratio formula.

        cu    = p + g - c
        co    = c - s
        alpha = cu / (cu + co) = (p + g - c) / (p + g - s)
        Q*    = alpha-quantile of the empirical net-demand sample

    The return parameters r, k do NOT appear: recapture lives entirely in
    the net-demand sample N (model_body Eq. 10, Lemma 1). Mathematically
    identical to the LP (Zipkin 2000, Prop. 2.1).

    Colab origin: lines 523-545 (alpha switched MT -> primitive).
    """
    cu = p + g - c
    co = c - s
    alpha = cu / (cu + co)

    Q_opt = float(np.quantile(N_flat, alpha))
    ESN = float(np.mean(np.maximum(N_flat - Q_opt, 0)))
    obj_opt = float(cu * ESN + co * np.mean(np.maximum(Q_opt - N_flat, 0)))

    return {"Q_opt": Q_opt, "obj_opt": obj_opt, "ESN": ESN,
            "alpha": alpha,
            "solver": f"Quantile (alpha={alpha:.4f})"}


def _solve_with_quantile_mt(N_flat, pN, c, s, gN) -> dict:
    """
    COMPARISON ONLY (model_body Prop 3, Eq. 11) — Mostard-Teunter transformed
    critical ratio. NOT the operating solver.

        alpha_MT = (pN - c + gN) / (pN - s + gN),
        pN = pG/(1-rk),  gN = g/(1-rk).

    This couples the resale cascade to provision; under the exogenous-return
    accounting of this paper it double-counts recapture (recapture is already
    in N). Retained so §5 can report the alpha_MT vs alpha gap and confirm it
    is O(r) (Eq. 12) — negligible in this rk~0.025 regime.

    Colab origin: the PRE-correction _solve_with_quantile (verbatim), renamed.
    """
    cu = pN - c + gN
    co = c - s
    alpha = cu / (cu + co)

    Q_opt = float(np.quantile(N_flat, alpha))
    ESN = float(np.mean(np.maximum(N_flat - Q_opt, 0)))
    obj_opt = float(cu * ESN + co * np.mean(np.maximum(Q_opt - N_flat, 0)))

    return {"Q_opt": Q_opt, "obj_opt": obj_opt, "ESN": ESN,
            "alpha": alpha,
            "solver": f"Quantile-MT (alpha_MT={alpha:.4f})"}


def solve_saa(
    N_matrix: np.ndarray,
    p: float,
    c: float,
    s: float,
    g: float,
    gurobi_env=None,
    Q_max: float = None,
    verbose: bool = False,
) -> dict:
    """
    Solve the SAA newsvendor for a net-demand matrix (PRIMITIVE ratio).

    Parameters
    ----------
    N_matrix : np.ndarray, shape (m, J)
        Net-demand realizations from build_net_demand_matrix.
    p, c, s, g : float
        PRIMITIVE market constants (selling price, unit cost, salvage,
        shortage penalty). alpha = (p+g-c)/(p+g-s). Return params r, k are
        NOT passed: they live in N, not in the cost coefficients.
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

    SIGNATURE CHANGE (v15): was solve_saa(N, c, s, pN, gN); now takes the
    primitive (p, c, s, g). Callers that previously passed pN=prm['pN'],
    gN=prm['gN'] must pass p=prm['p'], g=prm['g'].
    """
    m, J = N_matrix.shape
    N_flat = N_matrix.flatten()
    if Q_max is None:
        Q_max = float(N_flat.max()) * 2.0 if N_flat.max() > 0 else 1.0

    if gurobi_env is not None:
        try:
            result = _solve_with_gurobi(N_flat, p, c, s, g, Q_max, gurobi_env)
        except Exception as e:
            warnings.warn(f"Gurobi failed ({e}) -> quantile fallback.")
            result = _solve_with_quantile(N_flat, p, c, s, g)
    else:
        result = _solve_with_quantile(N_flat, p, c, s, g)

    # alpha (critical ratio) is solver-independent — always report it
    if "alpha" not in result:
        cu = p + g - c
        co = c - s
        result["alpha"] = cu / (cu + co)


    if verbose:
        print(f"  Solver : {result['solver']}   m={m} J={J} mJ={m*J}")
        print(f"  Q*     = {result['Q_opt']:.4f}")
        print(f"  Cost   = {result['obj_opt']:.4f}")
        print(f"  ESN    = {result['ESN']:.4f}")



    return result


def compare_critical_ratios(
    N_matrix: np.ndarray,
    p: float, c: float, s: float, g: float,
    pN: float, gN: float,
) -> dict:
    """
    COMPARISON (model_body Prop 3 / Eq. 12): primitive alpha vs MT alpha_MT
    on the SAME net-demand sample. Reports both Q* and cost and their gaps,
    to confirm the O(r) closeness empirically in §5.

    Not used in the main pipeline; called by the sequencing/robustness report.
    """
    prim = _solve_with_quantile(N_matrix.flatten(), p, c, s, g)
    mt = _solve_with_quantile_mt(N_matrix.flatten(), pN, c, s, gN)
    return {
        "alpha": prim["alpha"], "alpha_MT": mt["alpha"],
        "d_alpha": mt["alpha"] - prim["alpha"],
        "Q_prim": prim["Q_opt"], "Q_MT": mt["Q_opt"],
        "dQ": mt["Q_opt"] - prim["Q_opt"],
        "cost_prim": prim["obj_opt"], "cost_MT": mt["obj_opt"],
    }


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python saa_solver.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params
    from monte_carlo import build_net_demand_matrix
    try:
        from gurobi_env import make_gurobi_env
        env = make_gurobi_env()   # None if no license → quantile path
    except Exception:
        env = None

    datasets = build_category_datasets(order_items_clean, returns_clean, period="2W")
    params = build_category_params(datasets, order_items_clean)

    print(f"\n{'='*70}")
    print("SAA Q* per category (2W period, rule S4) — PRIMITIVE alpha")
    print(f"{'='*70}")
    print(f"{'Category':<24} {'alpha':>7} {'Q*':>7} {'Cost':>9} {'ESN':>7}")
    print("-" * 60)

    for cat, ds in datasets.items():
        prm = params[cat]
        N = build_net_demand_matrix(
            ds, prm["r_hat"], prm["k_hat"], rule="S4", J=200, seed=0
        )
        res = solve_saa(
            N, p=prm["p"], c=prm["c"], s=prm["s"], g=prm["g"],
            gurobi_env=env,
        )
        alpha = res.get("alpha", float("nan"))
        print(f"{cat:<24} {alpha:>7.4f} {res['Q_opt']:>7.2f} "
              f"{res['obj_opt']:>9.4f} {res['ESN']:>7.4f}")
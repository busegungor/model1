"""
parameter_estimation.py — Per-category parameter estimation and cost setup.

Computes everything the SAA / Bayesian solvers need, separating clearly:
  - ESTIMATED FROM DATA:   r_hat, k_hat (frequentist MLE, from the panel)
                           p, c        (category mean price / cost)
  - DERIVED:               s  = s_pct * c
  - CONSTANT (config):     s_pct, g, d
  - MT-COMPARISON ONLY:    pG, pN, gN  (Mostard-Teunter transformed economics,
                           model_body Eq. 11 / Prop 3). These are NOT used by
                           the primitive solver; retained so §5 can report the
                           alpha_MT vs alpha gap (Eq. 12).

NOTE on the critical ratio (v15). The operating solver uses the PRIMITIVE
ratio alpha = (p+g-c)/(p+g-s) (model_body Eq. 10); return params r, k enter
ONLY through the net-demand sample N, never through the cost coefficients.
The earlier code scaled p, g by 1/(1-rk) (pN, gN) and used alpha_MT — that
double-counts recapture (N already carries it) and is now confined to the
comparison path. See saa_solver._solve_with_quantile_mt.

NOTE on Bayesian: r_hat and k_hat here are the FREQUENTIST point estimates
used by SAA. The Bayesian solver (Step 14) replaces them with Beta
posteriors; their posterior means converge to these r_hat/k_hat as m→∞.
This module is the SAA-side parameter source AND a sanity reference for
the Bayesian posteriors.

────────────────────────────────────────────────────────────────────
Cost-parameter formulas (model_body §3; Colab lines 338-360):
  r_hat = Σ u_t / Σ G_t        (line-level; Path L)
  k_hat = Σ k_obs_t / Σ u_t
  s     = s_pct · c
  PRIMITIVE (operating):  cu = p+g-c,  co = c-s,  alpha = (p+g-c)/(p+g-s)
  Well-posed: p + g > c > s >= 0
  MT-COMPARISON (Eq. 11, not operating):
    pG = (1-r)·p + r·k·(p-d) + r·(1-k)·(s-d)
    pN = pG / (1 - r·k)
    gN = g  / (1 - r·k)
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from config import COST_PARAMS, ORDER_STATUS_FILTER


def estimate_rates(panel: pd.DataFrame) -> dict:
    """
    Frequentist MLE of return and resalability probabilities from a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Per-category panel from preprocessing.build_category_datasets,
        with columns [G_t, u_t, k_obs_t] counted at the LINE level (Path L).

    Returns
    -------
    dict with keys: r_hat, k_hat, total_G, total_u, total_k

    Colab origin: lines 338-339
        r_hat = total_u / total_G
        k_hat = total_k / total_u
    """
    total_G = int(panel["G_t"].sum())
    total_u = int(panel["u_t"].sum())
    total_k = int(panel["k_obs_t"].sum())

    r_hat = total_u / total_G if total_G > 0 else 0.0
    k_hat = total_k / total_u if total_u > 0 else 0.0

    return {
        "r_hat": r_hat,
        "k_hat": k_hat,
        "total_G": total_G,
        "total_u": total_u,
        "total_k": total_k,
    }


def estimate_price_cost(order_items_clean: pd.DataFrame, category: str) -> dict:
    """
    Category-level selling price p and unit cost c.

    Uses the ROW-MEAN of unit_price / unit_cost over fulfilled order_items
    in the category — i.e. each sold order_item contributes equally,
    matching the original Colab computation. (unit_price is the catalog
    price column; item_price is the realized transaction price. The model
    uses catalog price p.)

    Parameters
    ----------
    order_items_clean : pd.DataFrame
        From data_analysis.py. One row per order_item.
    category : str

    Returns
    -------
    dict with keys: p, c

    Colab origin: per-category p, c via mean of unit_price / unit_cost.
    """
    items = order_items_clean.loc[
        (order_items_clean["order_status"].isin(ORDER_STATUS_FILTER))
        & (order_items_clean["category_name"] == category),
        ["unit_price", "unit_cost"],
    ]
    p = float(items["unit_price"].mean())
    c = float(items["unit_cost"].mean())
    return {"p": p, "c": c}


def compute_cost_params(
    p: float,
    c: float,
    r: float,
    k: float,
    s_pct: float = None,
    g: float = None,
    d: float = None,
) -> dict:
    """
    Compute the newsvendor cost parameters.

    PRIMITIVE (operating): the solver needs only p, c, s, g — alpha is
    (p+g-c)/(p+g-s) and contains no r, k (model_body Eq. 10).

    MT-COMPARISON (not operating): pG, pN, gN (model_body Eq. 11) are also
    returned so saa_solver.compare_critical_ratios can report the alpha_MT
    gap (Eq. 12). They are NOT consumed by solve_saa.

    Parameters
    ----------
    p, c : float
        Selling price and unit cost (from estimate_price_cost).
    r, k : float
        Return probability and resalability (from estimate_rates, OR a
        sensitivity grid value, OR a Bayesian posterior draw). Used ONLY for
        the MT-comparison terms; the primitive ratio ignores them.
    s_pct, g, d : float, optional
        Cost constants. Default to config.COST_PARAMS if not given.

    Returns
    -------
    dict with keys: p, c, s, g, d, r, k, pG, pN, gN, well_posed

    Notes
    -----
    Guards against r*k → 1 (division blow-up) in the MT terms. If r*k >=
    1 - 1e-9 the net-demand transformation is undefined; we clip the
    denominator. In practice r*k is ~0.02-0.04 here, so this is purely
    defensive (matters only for an extreme Bayesian draw / sensitivity grid).

    well_posed uses the PRIMITIVE condition p + g > c > s >= 0 (the operating
    ratio's requirement), not the MT condition.

    Colab origin: lines 355-360 (pN/gN now comparison-only; well_posed
    switched to the primitive condition).
    """
    if s_pct is None:
        s_pct = COST_PARAMS["s_pct"]
    if g is None:
        g = COST_PARAMS["g"]
    if d is None:
        d = COST_PARAMS["d"]

    # Derived salvage value
    s = s_pct * c

    # ── PRIMITIVE well-posedness (operating ratio): p + g > c > s >= 0 ──
    well_posed = (p + g > c) and (c > s) and (s >= 0)

    # ── MT-COMPARISON terms (model_body Eq. 11; NOT used by solve_saa) ──
    # Net-demand scaling denominator, guarded against r*k → 1.
    denom = 1.0 - r * k
    if denom < 1e-9:
        denom = 1e-9  # defensive: avoid division blow-up

    # Expected gross revenue per unit of gross demand (model_body Eq. 11)
    pG = (1 - r) * p + r * k * (p - d) + r * (1 - k) * (s - d)

    # Per net-demand-unit revenue and shortage cost (MT transform, Eq. 11)
    pN = pG / denom
    gN = g / denom

    return {
        "p": p, "c": c, "s": s, "g": g, "d": d, "r": r, "k": k,
        "pG": pG, "pN": pN, "gN": gN,
        "well_posed": well_posed,
    }


def build_category_params(
    datasets: dict,
    order_items_clean: pd.DataFrame,
    s_pct: float = None,
    g: float = None,
    d: float = None,
) -> dict:
    """
    Assemble the full parameter set for every category.

    Combines estimate_rates + estimate_price_cost + compute_cost_params.

    Parameters
    ----------
    datasets : dict
        Output of preprocessing.build_category_datasets.
    order_items_clean : pd.DataFrame
        For price/cost lookup.
    s_pct, g, d : float, optional
        Override config defaults if given.

    Returns
    -------
    dict
        {category: {r_hat, k_hat, p, c, s, g, d, pG, pN, gN, well_posed,
                    total_G, total_u, total_k}}
    """
    params = {}
    for cat, d_cat in datasets.items():
        rates = estimate_rates(d_cat["panel"])
        pc = estimate_price_cost(order_items_clean, cat)
        cost = compute_cost_params(
            p=pc["p"], c=pc["c"],
            r=rates["r_hat"], k=rates["k_hat"],
            s_pct=s_pct, g=g, d=d,
        )
        # Merge all into one record
        params[cat] = {
            **rates,         # r_hat, k_hat, total_G/u/k
            **cost,          # p, c, s, g, d, r, k, pG, pN, gN, well_posed
        }
    return params


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python parameter_estimation.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets

    for per in ["W", "2W", "3W", "4W"]:
        datasets = build_category_datasets(order_items_clean, returns_clean, period=per)
        params = build_category_params(datasets, order_items_clean)

        print(f"\n{'='*78}")
        print(f"PER-CATEGORY PARAMETERS — Period: {per}")
        print(f"{'='*78}")
        rows = {}
        for cat, prm in params.items():
            # primitive critical ratio for reference
            alpha = (prm["p"] + prm["g"] - prm["c"]) / (prm["p"] + prm["g"] - prm["s"])
            rows[cat] = {
                "r_hat": round(prm["r_hat"], 4),
                "k_hat": round(prm["k_hat"], 4),
                "p": round(prm["p"], 2),
                "c": round(prm["c"], 2),
                "s": round(prm["s"], 2),
                "alpha": round(alpha, 4),
                "ok": prm["well_posed"],
            }
        print(pd.DataFrame(rows).T)
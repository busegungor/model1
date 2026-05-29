"""
sensitivity.py — Full-factorial sensitivity analysis over (g, k, r).

Sweeps cost/return parameters on a dense grid and reports three things
per cell, for every combination of period × rule × method:

  (a) Q* sensitivity (SAA and Bayesian)
  (b) SAA vs Bayesian gap (which method dominates, by how much)
  (c) Sequencing gap S3 - S1 (when does sequencing matter)

This addresses three supervisor feedback items at once:
  #5  "different assumptions don't matter much — try other parameters"
      → dense grid over g, k, r including extreme values (g up to 100)
  #9  "show sequencing matters systematically — when does it matter?"
      → sequencing gap (S3-S1) recorded for every cell
  + the new Bayesian contribution: SAA vs Bayesian comparison across the
      same grid.

KEY OPTIMIZATIONS (full grid is large: 720 param combos × 24 cells = 17K):
  1. Posteriors are computed ONCE per (category, period) and reused across
     all (g, k, r) cells, because the posterior depends on data, not on
     cost / return parameters.
  2. Net-demand matrices are also computed ONCE per (category, period,
     rule, method) and reused across (g, k, r) cells: the matrix depends
     on r_hat/k_hat (for SAA) or posteriors (for Bayesian), NOT on g.
     Only the cost computation (alpha, Q*, EP) changes per parameter cell.
  3. r and k parameter sweeps DO require new matrices (they change the
     binomial sampling), so those are looped inside.

Result is a DataFrame with one row per cell, suitable for heatmaps and
slice plots in visualization (Step 10/15).

────────────────────────────────────────────────────────────────────
Origin: extends Colab sensitivity loops with (i) Bayesian path,
        (ii) sequencing gap, (iii) period and rule axes.
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from config import SENSITIVITY_GRIDS, MC_SETTINGS
from preprocessing import build_category_datasets
from parameter_estimation import build_category_params
from monte_carlo import (build_net_demand_matrix,
                         build_bayesian_net_demand_matrix)
from saa_solver import solve_saa
from bayesian_solver import solve_bayesian
from bayesian_models import build_posteriors


# ════════════════════════════════════════════════════════════════════
# Cost parameter assembly (varies with g, k, r — the swept parameters)
# ════════════════════════════════════════════════════════════════════
def _build_cost_params(prm, g, k, r, d=2.0):
    """
    Rebuild cost parameters for a given (g, k, r) sensitivity cell.

    Uses the same scaling formula as parameter_estimation.compute_cost_params
    but with g, k, r swept rather than estimated from data:
        pG = (1-r) p + r k (p-d) + r (1-k) (s-d)
        pN = pG / (1 - r*k),  gN = g / (1 - r*k)
    p, c, s are kept at their estimated values (the price/cost structure of
    the category is not swept — only the return / penalty parameters are).

    Returns (c, s, pN, gN), or None if the cell is ill-posed.
    """
    p_, c_, s_ = prm["p"], prm["c"], prm["s"]
    if r * k >= 1.0:                            # division guard
        return None
    pG = (1 - r) * p_ + r * k * (p_ - d) + r * (1 - k) * (s_ - d)
    scale = 1.0 / (1.0 - r * k)
    pN = pG * scale
    gN = g * scale
    # well-posed: gN + pN > c > s >= 0
    if not (gN + pN > c_ > s_ >= 0):
        return None
    return c_, s_, pN, gN


# ════════════════════════════════════════════════════════════════════
# Solving on a precomputed matrix (no resampling)
# ════════════════════════════════════════════════════════════════════
def _solve_on_matrix(N_matrix, c, s, pN, gN):
    """
    Solve newsvendor on an EXISTING net-demand matrix for one (c,s,pN,gN).

    No env -> quantile path (fast, deterministic given the matrix).
    Returns (Q_opt, cost, alpha).
    """
    res = solve_saa(N_matrix, c=c, s=s, pN=pN, gN=gN, gurobi_env=None)
    return res["Q_opt"], res["obj_opt"], res["alpha"]


# ════════════════════════════════════════════════════════════════════
# One (category, period) block of the grid
# ════════════════════════════════════════════════════════════════════
def _run_one_block(cat, ds, prm, period, g_grid, k_grid, r_grid,
                   rules, J, seed) -> list:
    """
    Run the full (g x k x r x rule x method) grid for ONE category-period.

    Returns a list of dict rows, one per cell.

    Optimization: for each (rule, r, k) we build the net-demand matrix
    ONCE, then solve it for every g (g only enters the cost). This avoids
    recomputing m*J net-demand draws for every g.
    """
    rows = []
    # Bayesian posteriors: data-dependent only, computed once
    posteriors = build_posteriors(ds["panel"])

    for rule in rules:
        # Loop r and k OUTSIDE (they affect the matrix via binomial sampling)
        for r in r_grid:
            for k in k_grid:
                # SAA matrix at this (r, k) — built once, reused for all g
                N_saa = build_net_demand_matrix(ds, r, k, rule=rule,
                                                J=J, seed=seed)
                # Bayesian matrix is DATA-DRIVEN (uses posteriors), NOT (r,k)
                # → build once per (rule), reuse across (r, k, g)
                # But we want consistent comparison: Bayesian posteriors
                # already reflect data's r, k. So Bayesian matrix is fixed
                # per (rule); we put it OUTSIDE the (r,k) loop below.
                # For clarity, build inside but cache:
                # (We'll just build it once per rule via a small trick.)

                for g in g_grid:
                    cp = _build_cost_params(prm, g, k, r)
                    if cp is None:
                        continue
                    c_, s_, pN, gN = cp
                    Q_saa, cost_saa, alpha = _solve_on_matrix(
                        N_saa, c_, s_, pN, gN)
                    rows.append({
                        "category": cat, "period": period, "rule": rule,
                        "g": g, "k": k, "r": r,
                        "method": "SAA",
                        "Q_opt": Q_saa, "cost": cost_saa, "alpha": alpha,
                    })

    # Bayesian — posteriors fixed per (cat, period); matrix fixed per rule
    # (Bayesian net demand doesn't use swept r, k — it uses posterior draws.)
    for rule in rules:
        N_bay = build_bayesian_net_demand_matrix(
            ds, rule=rule, J=J, seed=seed, posteriors=posteriors)
        for r in r_grid:
            for k in k_grid:
                for g in g_grid:
                    cp = _build_cost_params(prm, g, k, r)
                    if cp is None:
                        continue
                    c_, s_, pN, gN = cp
                    Q_bay, cost_bay, alpha = _solve_on_matrix(
                        N_bay, c_, s_, pN, gN)
                    rows.append({
                        "category": cat, "period": period, "rule": rule,
                        "g": g, "k": k, "r": r,
                        "method": "Bayesian",
                        "Q_opt": Q_bay, "cost": cost_bay, "alpha": alpha,
                    })

    return rows


# ════════════════════════════════════════════════════════════════════
# Main entry point
# ════════════════════════════════════════════════════════════════════
def run_sensitivity(
    categories=("Classic Scrunchies",),
    periods=("W", "2W", "3W", "4W"),
    rules=("S1", "S3", "S4"),
    g_grid=None, k_grid=None, r_grid=None,
    J=None, seed=0,
    verbose=True,
) -> pd.DataFrame:
    """
    Run full-factorial sensitivity analysis.

    Parameters
    ----------
    categories : tuple of str
        Default Classic only (representative). Pass more for full coverage.
    periods : tuple of str
    rules : tuple of str
    g_grid, k_grid, r_grid : array-like, optional
        Defaults pulled from config.SENSITIVITY_GRIDS (dense ranges).
    J : int
        MC replications; default config.MC_SETTINGS["J_baseline"].

    Returns
    -------
    pd.DataFrame with columns:
        category, period, rule, g, k, r, method, Q_opt, cost, alpha
    """
    if g_grid is None:
        g_grid = SENSITIVITY_GRIDS["g"]
    if k_grid is None:
        k_grid = SENSITIVITY_GRIDS["k"]
    if r_grid is None:
        r_grid = SENSITIVITY_GRIDS["r"]
    if J is None:
        J = MC_SETTINGS["J_baseline"]

    n_cells = (len(categories) * len(periods) * len(rules)
               * len(g_grid) * len(k_grid) * len(r_grid) * 2)
    if verbose:
        print(f"Sensitivity grid: {n_cells} cells")
        print(f"  categories={len(categories)}, periods={len(periods)}, "
              f"rules={len(rules)}")
        print(f"  g({len(g_grid)})={list(g_grid)}")
        print(f"  k({len(k_grid)})={[f'{x:.2f}' for x in k_grid]}")
        print(f"  r({len(r_grid)})={[f'{x:.2f}' for x in r_grid]}")

    # Build datasets ONCE per period (expensive); cache across categories
    rows = []
    from data_analysis import order_items_clean, returns_clean

    for period in periods:
        if verbose:
            print(f"\n[period={period}] Building datasets / parameters...")
        datasets = build_category_datasets(order_items_clean, returns_clean,
                                           period=period)
        params = build_category_params(datasets, order_items_clean)

        for cat in categories:
            if cat not in datasets:
                if verbose:
                    print(f"  Skipping {cat}: not in datasets")
                continue
            if verbose:
                print(f"  [{cat}] running grid...")
            block_rows = _run_one_block(
                cat, datasets[cat], params[cat], period,
                g_grid, k_grid, r_grid, rules, J, seed,
            )
            rows.extend(block_rows)
            if verbose:
                print(f"     {len(block_rows)} rows")

    df = pd.DataFrame(rows)
    if verbose:
        print(f"\nDone. Total rows: {len(df)}")
    return df


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived columns for the three sensitivity metrics:
      - bayes_minus_saa_cost  (negative = Bayesian wins)
      - sequencing_gap_S3_S1  (per category/period/g/k/r/method)
    Returns a wide-format helper frame.
    """
    # SAA-Bayesian gap (per cell)
    piv = df.pivot_table(
        index=["category", "period", "rule", "g", "k", "r"],
        columns="method", values=["Q_opt", "cost"]
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()
    piv["bayes_minus_saa_cost"] = piv["cost_Bayesian"] - piv["cost_SAA"]
    piv["bayes_minus_saa_Q"] = piv["Q_opt_Bayesian"] - piv["Q_opt_SAA"]

    # Sequencing gap S3 - S1 (per category/period/g/k/r/method)
    rule_piv = df.pivot_table(
        index=["category", "period", "g", "k", "r", "method"],
        columns="rule", values="cost"
    ).reset_index()
    if "S1" in rule_piv.columns and "S3" in rule_piv.columns:
        rule_piv["sequencing_gap_S3_S1"] = rule_piv["S3"] - rule_piv["S1"]
    return piv, rule_piv


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python sensitivity.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time

    t0 = time.time()
    df = run_sensitivity()             # defaults: Classic only, all periods
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"Total runtime: {elapsed/60:.1f} minutes  ({len(df)} rows)")
    print(f"{'='*70}")

    # Save raw results
    from config import TABLES_DIR
    out_path = TABLES_DIR / "sensitivity_full.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Derived metrics
    saa_vs_bay, seq_gap = compute_derived_metrics(df)
    saa_vs_bay.to_csv(TABLES_DIR / "sensitivity_saa_vs_bayesian.csv", index=False)
    seq_gap.to_csv(TABLES_DIR / "sensitivity_sequencing_gap.csv", index=False)
    print(f"Saved: sensitivity_saa_vs_bayesian.csv, sensitivity_sequencing_gap.csv")

    # Quick summary
    print(f"\nQuick summary (Classic, W period):")
    sub = df[(df["category"] == "Classic Scrunchies") & (df["period"] == "W")]
    print(f"  Cost range: [{sub['cost'].min():.2f}, {sub['cost'].max():.2f}]")
    print(f"  Q* range: [{sub['Q_opt'].min():.1f}, {sub['Q_opt'].max():.1f}]")
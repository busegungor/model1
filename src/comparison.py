"""
comparison.py — Method comparison: SAA vs Bayesian (Ban, Gao & Taigel 2019 style).

THIS IS A METHOD COMPARISON, NOT A SENSITIVITY ANALYSIS. The distinction:
  - sensitivity.py : ONE method (SAA), sweep a parameter, see how Q* moves.
  - comparison.py  : TWO methods, common metric (out-of-sample relative
                     regret), see which one decides better and WHEN.

Because it is a comparison, r and k are NOT swept on a grid: each method
learns them FROM THE DATA in its own way (SAA: point estimate r_hat,k_hat;
Bayesian: posterior). Sweeping r,k from a grid would force SAA to use the
grid value while Bayesian uses its posterior -> the two methods would sit at
different r,k and the comparison would be unfair. So r,k come from data; we
only sweep NEUTRAL axes that both methods see identically:

  m_train : sample size (bootstrap subsample) -- the core small-data axis
  g       : shortage penalty -- fixed at a FEW representative policy values
            {5, 8, 20} (low / baseline / high service), NOT swept. This
            mirrors Ban, Gao & Taigel (2019) Table 7, which evaluates the
            methods at a few fixed critical-fractile settings rather than
            sweeping them; g alone has an obvious comparative-static effect
            (higher g -> higher Q*), so it is a scenario knob, not an object
            of study. Cost-layer only, so cheap.
  period  : W / 2W / 3W / 4W

Evaluation (no train/test): each method is fit on a bootstrap subsample of
size m_train and its order Q is scored against a full-data EMPIRICAL
net-demand TRUTH pool. Metric = relative regret = cost(Q)/cost(oracle) - 1.
Truth is empirical (not NegBin-Beta) so SAA is asymptotically exact and any
Bayesian small-m advantage is earned. Bootstrap = sample-size generator
(truth = empirical CDF; resampling emulates drawing m samples, Efron 1979).

m_train="all" is DETERMINISTIC (full panel, no bootstrap) so SAA's asymptote
is the genuine full-data fit (regret -> 0 up to MC noise).
"""

import numpy as np
import pandas as pd

from config import MC_SETTINGS
from preprocessing import build_category_datasets
from parameter_estimation import build_category_params, estimate_rates
from monte_carlo import (build_net_demand_matrix,
                         build_bayesian_net_demand_matrix)
from bayesian_models import build_posteriors


# ── cost helpers ────────────────────────────────────────────────────
def _alpha(c, s, pN, gN):
    cu = pN - c + gN; co = c - s
    return cu / (cu + co)


def _cost(Q, f, c, s, pN, gN):
    cu = pN - c + gN; co = c - s
    return float(cu * np.mean(np.maximum(f - Q, 0.0))
                 + co * np.mean(np.maximum(Q - f, 0.0)))


def build_truth(dataset, r_hat, k_hat, rule, J_truth=3000, seed=12345):
    """Full-data empirical net-demand pool (the target SAA converges to)."""
    return build_net_demand_matrix(dataset, r_hat, k_hat, rule=rule,
                                   J=J_truth, seed=seed).flatten()


def _subsample(dataset, pos):
    panel = dataset["panel"]; sp = panel.iloc[pos]
    at = dataset.get("arrival_times", {}); rt = dataset.get("return_arrival_times", {})
    keys = set(sp.index)
    return {"panel": sp,
            "arrival_times": {p: at[p] for p in keys if p in at},
            "return_arrival_times": {p: rt[p] for p in keys if p in rt}}


def _period_grid(m_max, base=(5, 10, 15, 20, 30, 50, 75)):
    """Per-period m_train grid: base values below 0.9*m_max, then 'all'."""
    grid = [m for m in base if m < 0.9 * m_max]
    grid.append(None)            # 'all' = deterministic full-data point
    return grid


# ════════════════════════════════════════════════════════════════════
# Core: SAA vs Bayesian regret for ONE (category, period), over m_train x g
# ════════════════════════════════════════════════════════════════════
def compare_one(dataset, prm, rule="S4", m_grid=None, g_grid=(5.0, 8.0, 20.0),
                B=60, J_fit=100, J_truth=3000, learn_returns=False, seed=0):
    """
    Returns df_raw: one row per (m_train, g, rep, method) with
    Q, cost, cost_oracle, rel_regret.

    r,k come FROM DATA (not swept):
      SAA      : point estimates from the subsample (learn_returns=True) or
                 full-data r_hat,k_hat held fixed (learn_returns=False, the
                 clean isolation of demand-distribution learning).
      Bayesian : posterior from the subsample's demand; r,k posteriors from
                 full data (learn_returns=False) or subsample (True).
    g is a POLICY parameter, swept on a grid (cost layer only -> matrices
    built once per (m_train,rep), reused across all g).
    The TRUTH is built with the FULL-DATA r_hat,k_hat (the real return
    structure), independent of g (g doesn't change the demand law).
    """
    panel = dataset["panel"]; m_max = len(panel)
    if m_grid is None:
        m_grid = _period_grid(m_max)

    c, s, p, d = prm["c"], prm["s"], prm["p"], prm["d"]
    r_full, k_full = prm["r_hat"], prm["k_hat"]
    rk = r_full * k_full
    # cost layer pieces that don't depend on g (phi = rk idealization)
    pG = (1 - r_full) * p + rk * (p - d) + r_full * (1 - k_full) * (s - d)
    pN = pG / (1 - rk); gN_unit = 1.0 / (1 - rk)     # gN = g * gN_unit

    # TRUTH (full-data return structure, real rule)
    truth = build_truth(dataset, r_full, k_full, rule, J_truth=J_truth)

    post_full = build_posteriors(panel)
    rng = np.random.default_rng(seed)
    rows = []

    for m_train in m_grid:
        if m_train is None:
            mt, B_eff, det, lbl = m_max, 1, True, "all"
        else:
            mt, B_eff, det, lbl = int(m_train), B, False, int(m_train)

        for b in range(B_eff):
            pos = np.arange(m_max) if det else rng.integers(0, m_max, size=mt)
            sub = _subsample(dataset, pos)

            # r,k FROM DATA (not from any grid)
            if learn_returns:
                rr = estimate_rates(sub["panel"])
                r_s = rr.get("r_hat", r_full); k_s = rr.get("k_hat", k_full)
                if not np.isfinite(r_s): r_s = 0.0
                if not np.isfinite(k_s): k_s = k_full
            else:
                r_s, k_s = r_full, k_full

            # fit matrices ONCE (g doesn't affect them)
            N_saa = build_net_demand_matrix(
                sub, r_s, k_s, rule=rule, J=J_fit,
                seed=int(rng.integers(1, 2**31 - 1)))
            post_sub = build_posteriors(sub["panel"])
            posteriors = post_sub if learn_returns else {
                "demand": post_sub["demand"], "r": post_full["r"], "k": post_full["k"]}
            N_bay = build_bayesian_net_demand_matrix(
                sub, rule=rule, J=J_fit,
                seed=int(rng.integers(1, 2**31 - 1)), posteriors=posteriors)
            f_saa = N_saa.flatten(); f_bay = N_bay.flatten()

            # sweep g cheaply (cost layer only)
            for g in g_grid:
                gN = g * gN_unit
                if not (pN + gN > c > s >= 0):
                    continue
                a = _alpha(c, s, pN, gN)
                co = _cost(np.quantile(truth, a), truth, c, s, pN, gN)
                Qs = float(np.quantile(f_saa, a)); Qb = float(np.quantile(f_bay, a))
                cs_ = _cost(Qs, truth, c, s, pN, gN)
                cb_ = _cost(Qb, truth, c, s, pN, gN)
                rows.append({"m_train": lbl, "g": g, "rep": b, "method": "SAA",
                             "Q": Qs, "cost": cs_, "cost_oracle": co,
                             "rel_regret": cs_ / co - 1})
                rows.append({"m_train": lbl, "g": g, "rep": b, "method": "Bayesian",
                             "Q": Qb, "cost": cb_, "cost_oracle": co,
                             "rel_regret": cb_ / co - 1})
    return pd.DataFrame(rows)


def summarize(df_raw):
    """Mean cost, regret & Q per (m_train, g, method) with standard errors."""
    g = (df_raw.groupby(["m_train", "g", "method"])
         .agg(cost_mean=("cost", "mean"),
              Q_mean=("Q", "mean"),
              Q_se=("Q", lambda x: x.std() / np.sqrt(max(len(x), 1))),
              regret_mean=("rel_regret", "mean"),
              regret_se=("rel_regret", lambda x: x.std() / np.sqrt(max(len(x), 1))),
              n=("cost", "count")).reset_index())
    return g


# ════════════════════════════════════════════════════════════════════
# Driver: all categories x periods
# ════════════════════════════════════════════════════════════════════
def run_comparison(periods=("W", "2W", "3W", "4W"), rule="S4",
                   g_grid=(5.0, 8.0, 20.0),
                   B=60, J_fit=100, J_truth=3000, learn_returns=False,
                   seed=0, verbose=True):
    """
    SAA vs Bayesian out-of-sample regret across periods, m_train, and g.

    Uses rule="S4" — the EMPIRICAL order-of-arrival netting, i.e. the rule
    that actually occurs. The method comparison answers "which method decides
    better", and the real decision is made under S4, so the comparison is run
    under S4 only. Whether sequencing matters at all (S1 vs S3 vs S4) is a
    SEPARATE question, answered by sequencing_importance() below — that is a
    property of the data, not a method-vs-method decision.
    """
    from data_analysis import order_items_clean, returns_clean
    all_rows = []
    for period in periods:
        datasets = build_category_datasets(order_items_clean, returns_clean,
                                            period=period)
        params = build_category_params(datasets, order_items_clean)
        for cat, ds in datasets.items():
            if verbose:
                print(f"[{period}|{cat}|{rule}] m_max={len(ds['panel'])}")
            df_raw = compare_one(ds, params[cat], rule=rule, g_grid=g_grid,
                                 B=B, J_fit=J_fit, J_truth=J_truth,
                                 learn_returns=learn_returns, seed=seed)
            s = summarize(df_raw)
            s.insert(0, "period", period)
            s.insert(1, "category", cat)
            s.insert(2, "rule", rule)
            all_rows.append(s)
    return pd.concat(all_rows, ignore_index=True)


# ════════════════════════════════════════════════════════════════════
# Sequencing importance — WHEN does sequencing matter (if at all)?
# ════════════════════════════════════════════════════════════════════
def sequencing_importance(periods=("W", "2W", "3W", "4W"),
                          g=8.0, J=2000, seed=0, verbose=True):
    """
    NOT a method comparison and NOT a decision — a DATA DIAGNOSTIC.

    The real model always uses S4 (empirical order-of-arrival). S1 (best
    case: all resalable returns offset demand) and S3 (worst case: none do)
    are the BOUNDS of the sequencing spectrum, never an operating choice.
    This function reports, per (period, category, method), the cost under
    each bound and where S4 falls, to answer "when does it matter, if at
    all": if cost_S3 - cost_S1 is small, sequencing is irrelevant.

    All three rules use the SAME upstream draws (same seed) and the SAME
    cost layer, so differences are purely sequencing. Computed for BOTH
    methods — SAA (point-estimate r,k, empirical demand) and Bayesian
    (posterior r,k, NegBin-Beta demand) — so the claim "sequencing is
    irrelevant in this regime" is shown for each method rather than assumed
    to transfer from one to the other. Note the two methods estimate r,k
    differently (point estimate vs posterior mean), so their rk — and hence
    the band width — can differ slightly even on full data.

    Reports per method:
      cost_S1, cost_S4, cost_S3
      gap_pct  = (cost_S3 - cost_S1)/cost_S1 * 100   (max impact)
      S4_pos   = (cost_S4 - cost_S1)/(cost_S3 - cost_S1)  (0=best, 1=worst)
    """
    from data_analysis import order_items_clean, returns_clean
    rows = []
    for period in periods:
        datasets = build_category_datasets(order_items_clean, returns_clean,
                                            period=period)
        params = build_category_params(datasets, order_items_clean)
        for cat, ds in datasets.items():
            prm = params[cat]
            c, s_, p, d = prm["c"], prm["s"], prm["p"], prm["d"]
            r, k = prm["r_hat"], prm["k_hat"]
            rk = r * k
            pG = (1 - r) * p + rk * (p - d) + r * (1 - k) * (s_ - d)
            pN = pG / (1 - rk); gN = g / (1 - rk)
            a = _alpha(c, s_, pN, gN)
            post = build_posteriors(ds["panel"])

            for method in ("SAA", "Bayesian"):
                costs = {}
                for rule in ("S1", "S4", "S3"):
                    # SAME seed across rules -> identical draws -> pure sequencing
                    if method == "SAA":
                        f = build_net_demand_matrix(
                            ds, r, k, rule=rule, J=J, seed=seed).flatten()
                    else:
                        f = build_bayesian_net_demand_matrix(
                            ds, rule=rule, J=J, seed=seed,
                            posteriors=post).flatten()
                    Q = float(np.quantile(f, a))
                    costs[rule] = _cost(Q, f, c, s_, pN, gN)
                denom = costs["S3"] - costs["S1"]
                rows.append({
                    "period": period, "category": cat, "method": method,
                    "rk": round(rk, 4),
                    "cost_S1": round(costs["S1"], 4),
                    "cost_S4": round(costs["S4"], 4),
                    "cost_S3": round(costs["S3"], 4),
                    "gap_pct": round(100 * denom / costs["S1"], 3),
                    "S4_pos": round((costs["S4"] - costs["S1"]) / denom, 3)
                              if abs(denom) > 1e-9 else np.nan,
                })
                if verbose:
                    print(f"[{period}|{cat}|{method}] gap={rows[-1]['gap_pct']:.2f}%  "
                          f"S4_pos={rows[-1]['S4_pos']}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from pathlib import Path
    OUT = Path("comparison_outputs"); OUT.mkdir(exist_ok=True)

    # ── 1. Method comparison: SAA vs Bayesian, S4 only (the real rule) ──
    df = run_comparison(rule="S4", g_grid=(5.0, 8.0, 20.0),
                        B=60, J_fit=MC_SETTINGS["J_sensitivity"])
    df.to_csv(OUT / "method_comparison.csv", index=False)
    print(f"\nSaved -> {OUT/'method_comparison.csv'}")
    v = df[(df.period == "W") & (df.g == 8.0)]
    if len(v):
        cat0 = v.category.iloc[0]
        piv = v[v.category == cat0].pivot(index="m_train", columns="method",
                                          values="regret_mean")
        print(f"\nMethod comparison — {cat0}, W, S4, g=8: regret vs m_train")
        print(piv.to_string())

    # ── 2. Sequencing importance diagnostic: S1/S4/S3 bounds (when matters)
    print("\n" + "=" * 60)
    print("SEQUENCING IMPORTANCE (data diagnostic, not a decision)")
    print("=" * 60)
    seq = sequencing_importance(g=8.0, J=2000)
    seq.to_csv(OUT / "sequencing_importance.csv", index=False)
    print("\nMax sequencing impact (gap_pct = (cost_S3-cost_S1)/cost_S1):")
    print(seq.groupby("period")["gap_pct"].agg(["mean", "max"]).round(2).to_string())
    print(f"\nSaved -> {OUT/'sequencing_importance.csv'}")
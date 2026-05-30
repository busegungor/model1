"""
comparison.py — Step 15: SAA vs Bayesian as a function of data size.

Two studies, both evaluated against a TRUTH distribution (no train/test
split; evaluation is against the data-generating truth, in the style of
Besbes & Mouchtaki 2023, Levi et al. 2015, Berk et al. 2007).

A) REGIME MAP (no bootstrap, descriptive): SAA vs Bayesian across period
   definitions W/2W/3W/4W at the natural sample size m=101/51/34/26. Reports
   Q* and the realized recapture rate phi = 1 - E[N]/E[G]. (In-sample cost
   columns are method-specific and NOT a quality comparison.)

B) SAMPLE-SIZE STUDY (bootstrap-as-truth): run per (period, sequence rule).
   Within a period the net-demand distribution is fixed and the data size
   m_train is varied by bootstrap subsampling that period's own historical
   periods, up to its own m_max. SAA (empirical) and Bayesian (NegBin-Beta
   posterior) are fit on each resample and their order Q is scored AGAINST
   that period's full-data empirical net-demand TRUTH.

   Reported per (period, category, rule, m_train, method):
     cost        = expected newsvendor cost of the fitted Q under the truth
                   pool  (the quality metric, comparable WITHIN a category)
     cost_oracle = cost of the truth-optimal Q (best achievable)
     rel_regret  = cost / cost_oracle - 1  (cost normalized by the oracle)
   The m_train="all" point is DETERMINISTIC (full panel, no bootstrap), so
   SAA's asymptote is the genuine full-data fit (cost -> cost_oracle).

DESIGN
------
- TRUTH = full-data empirical net-demand pool (NOT NegBin-Beta), so SAA is
  asymptotically exact and any Bayesian small-m advantage is earned.
- c,s,pN,gN and r_hat,k_hat held at full-data values; only the demand
  distribution is learned from the subsample (learn_returns=True to also
  learn r,k).
- Bootstrap is a sample-size generator, not a held-out evaluator
  (truth = full-data empirical CDF; resampling emulates drawing m samples —
  Efron 1979). No train/test.
- phi = r*k idealization retained (sequencing_recapture_consistency.md).

Q* is the alpha-quantile of the flattened net-demand sample (the exact
SAA / Bayesian quantile core); matches solve_saa / solve_bayesian.
"""

import numpy as np
import pandas as pd

from config import MC_SETTINGS
from preprocessing import build_category_datasets
from parameter_estimation import build_category_params, estimate_rates
from monte_carlo import (build_net_demand_matrix,
                         build_bayesian_net_demand_matrix)
from bayesian_models import build_posteriors


# ════════════════════════════════════════════════════════════════════
# Cost helpers
# ════════════════════════════════════════════════════════════════════
def _alpha(c, s, pN, gN):
    cu = pN - c + gN
    co = c - s
    return cu / (cu + co)


def _cost(Q, N_flat, c, s, pN, gN):
    cu = pN - c + gN
    co = c - s
    return float(cu * np.mean(np.maximum(N_flat - Q, 0.0))
                 + co * np.mean(np.maximum(Q - N_flat, 0.0)))


def build_truth(dataset, r_hat, k_hat, rule, J_truth=3000, seed=12345):
    """Full-data empirical net-demand pool for this rule (SAA's target)."""
    N = build_net_demand_matrix(dataset, r_hat, k_hat,
                                rule=rule, J=J_truth, seed=seed)
    return N.flatten()


def _subsample(dataset, pos):
    panel = dataset["panel"]
    sub_panel = panel.iloc[pos]
    at = dataset.get("arrival_times", {})
    rt = dataset.get("return_arrival_times", {})
    keys = set(sub_panel.index)
    sub_at = {p: at[p] for p in keys if p in at}
    sub_rt = {p: rt[p] for p in keys if p in rt}
    return {"panel": sub_panel, "arrival_times": sub_at,
            "return_arrival_times": sub_rt}


def _period_grid(m_max, base=(5, 10, 15, 20, 30, 50, 75)):
    grid = [m for m in base if m < 0.9 * m_max]
    grid.append(None)              # 'all' = deterministic full-data point
    return grid


# ════════════════════════════════════════════════════════════════════
# STUDY B — sample-size curve for ONE (category, period, rule)
# ════════════════════════════════════════════════════════════════════
def sample_size_curve(dataset, prm, rule="S4", m_grid=None,
                      B=80, J_fit=100, J_truth=3000,
                      learn_returns=False, seed=0):
    """
    Returns (df_raw, info). df_raw has one row per
    (m_train, rep, method) with columns Q, cost, cost_oracle, rel_regret.
    """
    panel = dataset["panel"]
    m_max = len(panel)
    if m_grid is None:
        m_grid = _period_grid(m_max)

    c, s, pN, gN = prm["c"], prm["s"], prm["pN"], prm["gN"]
    r_full, k_full = prm["r_hat"], prm["k_hat"]
    alpha = _alpha(c, s, pN, gN)

    truth = build_truth(dataset, r_full, k_full, rule, J_truth=J_truth)
    Q_oracle = float(np.quantile(truth, alpha))
    cost_oracle = _cost(Q_oracle, truth, c, s, pN, gN)

    post_full = build_posteriors(panel)
    rng = np.random.default_rng(seed)
    rows = []

    for m_train in m_grid:
        if m_train is None:
            mt, B_eff, det, label = m_max, 1, True, "all"
        else:
            mt, B_eff, det, label = int(m_train), B, False, int(m_train)

        for b in range(B_eff):
            pos = np.arange(m_max) if det else rng.integers(0, m_max, size=mt)
            sub = _subsample(dataset, pos)

            # SAA
            if learn_returns:
                rr = estimate_rates(sub["panel"])
                r_s = rr.get("r_hat", r_full); k_s = rr.get("k_hat", k_full)
                if not np.isfinite(r_s): r_s = 0.0
                if not np.isfinite(k_s): k_s = k_full
            else:
                r_s, k_s = r_full, k_full
            N_saa = build_net_demand_matrix(
                sub, r_s, k_s, rule=rule, J=J_fit,
                seed=int(rng.integers(1, 2**31 - 1)))
            Q_saa = float(np.quantile(N_saa.flatten(), alpha))
            cost_saa = _cost(Q_saa, truth, c, s, pN, gN)

            # Bayesian
            post_sub = build_posteriors(sub["panel"])
            posteriors = post_sub if learn_returns else {
                "demand": post_sub["demand"],
                "r": post_full["r"], "k": post_full["k"]}
            N_bay = build_bayesian_net_demand_matrix(
                sub, rule=rule, J=J_fit,
                seed=int(rng.integers(1, 2**31 - 1)),
                posteriors=posteriors)
            Q_bay = float(np.quantile(N_bay.flatten(), alpha))
            cost_bay = _cost(Q_bay, truth, c, s, pN, gN)

            rows.append({"m_train": label, "rep": b, "method": "SAA",
                         "Q": Q_saa, "cost": cost_saa,
                         "cost_oracle": cost_oracle,
                         "rel_regret": cost_saa / cost_oracle - 1.0})
            rows.append({"m_train": label, "rep": b, "method": "Bayesian",
                         "Q": Q_bay, "cost": cost_bay,
                         "cost_oracle": cost_oracle,
                         "rel_regret": cost_bay / cost_oracle - 1.0})

    info = {"alpha": alpha, "Q_oracle": Q_oracle,
            "cost_oracle": cost_oracle, "m_max": m_max}
    return pd.DataFrame(rows), info


def summarize_curve(df_raw):
    """Mean cost, mean regret, std/se per (m_train, method)."""
    g = (df_raw.groupby(["m_train", "method"])
         .agg(cost_mean=("cost", "mean"),
              cost_se=("cost", lambda x: x.std() / np.sqrt(max(len(x), 1))),
              regret_mean=("rel_regret", "mean"),
              regret_se=("rel_regret", lambda x: x.std() / np.sqrt(max(len(x), 1))),
              n=("cost", "count"))
         .reset_index())
    return g


# ════════════════════════════════════════════════════════════════════
# STUDY A — regime map across periods (no bootstrap)
# ════════════════════════════════════════════════════════════════════
def regime_map(order_items_clean, returns_clean,
               periods=("W", "2W", "3W", "4W"),
               rules=("S1", "S4", "S3"), J=200, seed=0):
    rows = []
    for period in periods:
        datasets = build_category_datasets(order_items_clean, returns_clean,
                                            period=period)
        params = build_category_params(datasets, order_items_clean)
        for cat, ds in datasets.items():
            prm = params[cat]
            c, s, pN, gN = prm["c"], prm["s"], prm["pN"], prm["gN"]
            alpha = _alpha(c, s, pN, gN)
            m = len(ds["panel"]); G_mean = ds["panel"]["G_t"].mean()
            for rule in rules:
                N_saa = build_net_demand_matrix(
                    ds, prm["r_hat"], prm["k_hat"], rule=rule, J=J, seed=seed)
                Q_saa = float(np.quantile(N_saa.flatten(), alpha))
                phi = float(1.0 - N_saa.mean() / G_mean) if G_mean > 0 else np.nan
                N_bay = build_bayesian_net_demand_matrix(ds, rule=rule, J=J, seed=seed)
                Q_bay = float(np.quantile(N_bay.flatten(), alpha))
                rows.append({"period": period, "m": m, "category": cat,
                             "rule": rule, "alpha": round(alpha, 4),
                             "Q_SAA": round(Q_saa, 2), "Q_Bay": round(Q_bay, 2),
                             "Q_gap": round(Q_bay - Q_saa, 2),
                             "phi_SAA": round(phi, 4),
                             "rk": round(prm["r_hat"] * prm["k_hat"], 4)})
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from pathlib import Path
    from data_analysis import order_items_clean, returns_clean

    OUT = Path("comparison_outputs"); OUT.mkdir(exist_ok=True)
    RULES = ("S1", "S4", "S3")

    print("=" * 76)
    print("STUDY A — REGIME MAP (period x category x rule)")
    print("=" * 76)
    df_regime = regime_map(order_items_clean, returns_clean,
                           rules=RULES, J=MC_SETTINGS["J_baseline"])
    df_regime.to_csv(OUT / "regime_map.csv", index=False)
    print(df_regime[df_regime["rule"] == "S4"][
        ["period", "category", "m", "Q_SAA", "Q_Bay", "Q_gap"]].to_string(index=False))
    print(f"\nSaved -> {OUT/'regime_map.csv'}")

    print("\n" + "=" * 76)
    print("STUDY B — SAMPLE-SIZE STUDY (cost vs m_train; SAA vs Bayesian)")
    print("per period x sequence rule; cost = expected cost under truth pool")
    print("=" * 76)

    all_curves = []
    for period in ("W", "2W", "3W", "4W"):
        datasets = build_category_datasets(order_items_clean, returns_clean,
                                           period=period)
        params = build_category_params(datasets, order_items_clean)
        for rule in RULES:
            print(f"\n############  PERIOD {period}  |  RULE {rule}  ############")
            for cat, ds in datasets.items():
                df_raw, info = sample_size_curve(
                    ds, params[cat], rule=rule,
                    B=80, J_fit=MC_SETTINGS["J_sensitivity"], J_truth=3000,
                    learn_returns=False, seed=0)
                s = summarize_curve(df_raw)
                s.insert(0, "period", period)
                s.insert(1, "category", cat)
                s.insert(2, "rule", rule)
                all_curves.append(s)
                print(f"\n{cat}  (m_max={info['m_max']}, alpha={info['alpha']:.4f}, "
                      f"Q_oracle={info['Q_oracle']:.2f}, cost_oracle={info['cost_oracle']:.4f})")
                # cost table: SAA vs Bayesian side by side
                piv = s.pivot(index="m_train", columns="method", values="cost_mean")
                piv.columns = [f"cost_{c}" for c in piv.columns]
                pivr = s.pivot(index="m_train", columns="method", values="regret_mean")
                pivr.columns = [f"regret_{c}" for c in pivr.columns]
                out = piv.join(pivr)
                print(out.to_string())

    curve_df = pd.concat(all_curves, ignore_index=True)
    curve_df.to_csv(OUT / "sample_size_by_period_rule.csv", index=False)
    print(f"\nSaved -> {OUT/'sample_size_by_period_rule.csv'}")
    print("Columns: period, category, rule, m_train, method,")
    print("         cost_mean, cost_se, regret_mean, regret_se, n")
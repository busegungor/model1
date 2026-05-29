"""
preprocessing.py — Build model-ready category datasets.

Takes the cleaned dataframes from data_analysis.py and produces, per
category and per period definition:
  - a cohort-fixed panel of (G_t, u_t, k_obs_t)
  - intraperiod arrival timestamps (unit-level) for demand and returns

KEY DESIGN DECISIONS:
  1. Data source: data_analysis.py's order_items_clean / returns_clean
     (from main_clean.csv → 376 returns, the CORRECT cleaned set). The
     old Colab path read raw CSVs (800 returns) → inflated r_hat ~0.084;
     cleaned data gives r_hat ~0.045-0.055.
  2. Status filter: Completed + Returned both count as fulfilled demand.
  3. Cohort fix: returns attributed to SALE period (order_date), not
     return period. Preserves U_t | G_t ~ Binomial(G_t, r), so u_t <= G_t.
  4. Resalable: return_reason in RESALABLE_REASONS.
  5. MULTIPLE-RETURN DEDUP (Decision: Seçenek 1, resalable-priority):
     Some order_items appear with several return records (synthetic-data
     artifact — e.g. order_item 2387 has 3 returns, two of them defective,
     which is physically impossible to resell-then-return). The single-
     period model assumes "at most one return-and-resale cycle per unit"
     (paper Section 2.5), so we keep ONE return per order_item. We prefer
     the RESALABLE record (it carries the information relevant to k), and
     among ties take the earliest return_date. 376 -> 361 returns.
  6. UNIT-LEVEL TIMESTAMPS (addresses supervisor's question:
     "Customer demands: do they all take one unit or could it also be
     more?"). Quantity ranges 1-4 (mean 2.52, 76% of order_items have
     quantity > 1). An order_item with quantity q expands to q identical
     arrival fractions, so the netting logic operates on individual units.

────────────────────────────────────────────────────────────────────
Colab origin: load_real_data(), c_alıs_ır_son_hali_s_u_anlık.py
              lines ~150-310 (refactored to consume cleaned dataframes
              instead of re-reading raw CSVs; dedup added).
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from config import RESALABLE_REASONS, ORDER_STATUS_FILTER


def _period_fraction(dt_series: pd.Series, period: str) -> np.ndarray:
    """
    Normalized position of each timestamp within its period bucket, in [0,1).

    For "W" this is essentially day-of-week / 7. For multi-week periods we
    use (days since bucket start) / (bucket length in days).

    Colab origin: arrival_frac / return_frac computation (~lines 200, 290)
    """
    period_days = {"W": 7, "2W": 14, "3W": 21, "4W": 28}
    if period not in period_days:
        raise ValueError(f"Unknown period: {period}")
    n_days = period_days[period]

    period_obj = dt_series.dt.to_period(period)
    period_start = period_obj.dt.start_time
    days_into = (dt_series - period_start).dt.total_seconds() / (24 * 3600)
    frac = (days_into / n_days).to_numpy()
    return np.clip(frac, 0.0, 1.0 - 1e-9)


def _dedup_returns(ret: pd.DataFrame) -> pd.DataFrame:
    """
    Keep ONE return record per order_item (single-cycle assumption).

    DECISION (Seçenek 1, resalable-priority):
    Some order_items have multiple return records — a synthetic-data
    artifact, not a genuine resale-then-return cycle (e.g. defective items
    cannot be resold and then returned again). The single-period model
    permits at most one return-and-resale cycle per unit, so we collapse
    multiple returns of the same order_item to a single record.

    Selection rule:
      1. Prefer a RESALABLE return reason (resalable returns carry the
         information that matters for k; if a unit was ever resalable we
         treat it as such).
      2. Among ties, take the earliest return_date.

    Colab origin: NOT in the original Colab (the raw-CSV path silently
    summed multiple returns); added here after inspecting main_clean.csv.
    """
    out = (
        ret
        .sort_values(["is_resalable", "return_date"], ascending=[False, True])
        .drop_duplicates("order_item_id", keep="first")
    )
    return out


def build_category_datasets(
    order_items_clean: pd.DataFrame,
    returns_clean: pd.DataFrame,
    period: str = "W",
) -> dict:
    """
    Build per-category, cohort-fixed datasets for a given period definition.

    Parameters
    ----------
    order_items_clean : pd.DataFrame
        One row per order_item. Needs: order_id, order_item_id, order_date,
        order_status, product_id, quantity, category_name.
    returns_clean : pd.DataFrame
        Return rows. Needs: order_id, order_item_id, order_date (= SALE
        date), return_date, quantity, category_name, return_reason.
    period : str
        Cohort/period definition: "W", "2W", "3W", "4W".

    Returns
    -------
    dict
        {
          category_name: {
            "panel": DataFrame indexed by sale_period with [G_t, u_t, k_obs_t],
            "arrival_times": {sale_period: np.ndarray of demand fractions},
            "return_arrival_times": {sale_period: np.ndarray of return fracs},
          },
          ...
        }

    Notes
    -----
    - Returns are deduped to one per order_item (see _dedup_returns).
    - u_t, k_obs_t clipped so k_obs_t <= u_t <= G_t.
    - arrival_times are UNIT-LEVEL (quantity-expanded); for each period
      len(arrival_times[p]) == G_t[p].
    - return_arrival_times cover RESALABLE returns only (only resalable
      returns can offset demand in S4 order-of-arrival netting).
    """
    # ── 1. Fulfilled demand (Completed + Returned) ──────────────────
    items = order_items_clean.loc[
        order_items_clean["order_status"].isin(ORDER_STATUS_FILTER),
        ["order_item_id", "order_id", "order_date", "product_id",
         "quantity", "category_name"],
    ].copy()
    items["order_date"] = pd.to_datetime(items["order_date"], errors="coerce")
    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce").fillna(0).astype(int)
    items = items.dropna(subset=["order_date", "category_name"])

    items["sale_period"] = items["order_date"].dt.to_period(period)
    items["arrival_frac"] = _period_fraction(items["order_date"], period)

    # ── 2. Returns: clean, classify, DEDUP, cohort-fix ──────────────
    ret = returns_clean.loc[
        returns_clean["order_status"].isin(ORDER_STATUS_FILTER),
        ["order_item_id", "order_id", "order_date", "return_date",
         "quantity", "category_name", "return_reason"],
    ].copy()
    ret["order_date"] = pd.to_datetime(ret["order_date"], errors="coerce")
    ret["return_date"] = pd.to_datetime(ret["return_date"], errors="coerce")
    ret["quantity"] = pd.to_numeric(ret["quantity"], errors="coerce").fillna(0).astype(int)
    ret = ret.dropna(subset=["order_date", "category_name"])

    ret["is_resalable"] = ret["return_reason"].isin(RESALABLE_REASONS)

    # DEDUP: one return per order_item (single-cycle assumption)
    ret = _dedup_returns(ret)

    # COHORT FIX: attribute return to the SALE period (order_date),
    # not the return period. Keeps U_t | G_t ~ Binomial(G_t, r).
    ret["sale_period"] = ret["order_date"].dt.to_period(period)

    # Intraperiod timing of the return uses return_date (when in the cycle
    # it physically arrives); the COHORT (which period) is fixed by sale.
    ret["return_frac"] = _period_fraction(ret["return_date"], period)

    # ── 3. Per-category assembly ────────────────────────────────────
    datasets = {}
    categories = sorted(items["category_name"].unique())

    for cat in categories:
        items_c = items[items["category_name"] == cat]
        ret_c = ret[ret["category_name"] == cat]

        # ─ 3a. Period aggregates ─────────────────────────────────
        G = items_c.groupby("sale_period")["quantity"].sum()
        u = ret_c.groupby("sale_period")["quantity"].sum()
        k = ret_c[ret_c["is_resalable"]].groupby("sale_period")["quantity"].sum()

        all_periods = pd.period_range(
            start=G.index.min(), end=G.index.max(), freq=period
        )
        panel = pd.DataFrame(index=all_periods)
        panel.index.name = "sale_period"
        panel["G_t"] = G.reindex(all_periods).fillna(0).astype(int)
        panel["u_t"] = u.reindex(all_periods).fillna(0).astype(int)
        panel["k_obs_t"] = k.reindex(all_periods).fillna(0).astype(int)

        # ─ 3b. Clip to maintain k_obs_t <= u_t <= G_t ────────────
        panel["u_t"] = np.minimum(panel["u_t"], panel["G_t"])
        panel["k_obs_t"] = np.minimum(panel["k_obs_t"], panel["u_t"])

        # ─ 3c. Unit-level arrival timestamps ─────────────────────
        # Demand: quantity-expanded so one entry per UNIT (addresses the
        # supervisor's "could a customer take more than one unit?" — yes,
        # quantity is 1-4, so we expand).
        arrival_times = {}
        for p, grp in items_c.groupby("sale_period"):
            fracs = np.repeat(grp["arrival_frac"].values, grp["quantity"].values)
            arrival_times[p] = np.sort(fracs)

        # Resalable returns only, also unit-level.
        return_arrival_times = {}
        res_c = ret_c[ret_c["is_resalable"]]
        for p, grp in res_c.groupby("sale_period"):
            fracs = np.repeat(grp["return_frac"].values, grp["quantity"].values)
            return_arrival_times[p] = np.sort(fracs)

        datasets[cat] = {
            "panel": panel,
            "arrival_times": arrival_times,
            "return_arrival_times": return_arrival_times,
        }

    return datasets


def summarize_datasets(datasets: dict) -> pd.DataFrame:
    """
    Per-category summary of aggregates and implied frequentist rates.
    Sanity-check against the (re-generated) paper Table 4.
    """
    rows = {}
    for cat, d in datasets.items():
        panel = d["panel"]
        total_G = int(panel["G_t"].sum())
        total_u = int(panel["u_t"].sum())
        total_k = int(panel["k_obs_t"].sum())
        rows[cat] = {
            "n_periods": len(panel),
            "total_G": total_G,
            "total_u": total_u,
            "total_k": total_k,
            "r_hat": total_u / total_G if total_G > 0 else np.nan,
            "k_hat": total_k / total_u if total_u > 0 else np.nan,
        }
    return pd.DataFrame(rows).T


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python preprocessing.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean

    for per in ["W", "2W", "3W","4W"]:
        print(f"\n{'='*60}\nPERIOD: {per}\n{'='*60}")
        ds = build_category_datasets(order_items_clean, returns_clean, period=per)
        print(summarize_datasets(ds).round(4))
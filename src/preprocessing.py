"""
preprocessing.py — Build model-ready category datasets.

Takes the cleaned dataframes from data_analysis.py and produces, per
category and per period definition:
  - a cohort-fixed panel of (G_t, u_t, k_obs_t)
  - intraperiod arrival timestamps (line-level) for demand and returns

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
  6. LINE-LEVEL ACCOUNTING (Path L — supersedes the earlier unit-expand
     design). The atomic ordering AND return unit is the ORDER LINE
     (order_item), not the physical piece. Rationale: the data records a
     return per order_item (one return_id per line, no return-quantity
     column), so a returned line is observed as fully returned. Treating
     the line as the atomic Bernoulli(r) trial is what the data actually
     measures and keeps U_t | G_t ~ Binomial(G_t, r) exact (verified:
     return rate is homogeneous across line sizes q=1..4, ~4-5%, so line
     size carries no return-relevant information). An order may contain
     several lines; G_t counts LINES, not pieces. (This reverses the prior
     quantity-expand approach, which assumed within-line unit independence
     and produced over-dispersion phi~3.0 against the line-clustered data.)
  7. DAY-BASED COHORT INDEX (supersedes pd.to_period/period_range). pandas
     does not support multi-week period frequencies reliably: to_period("2W")
     silently produces WEEKLY buckets, while period_range(freq="2W") produces
     true 2-week buckets, so a reindex between them drops >50% of rows for
     2W/3W/4W. We instead assign each row an INTEGER cohort id =
     floor((order_date - origin) / period_days), with a single global origin
     (earliest order_date). The same formula builds the contiguous cohort
     range, so grouping and reindexing are aligned by construction and every
     row survives under every period definition. arrival_frac / return_frac
     (intraperiod timing) are unchanged — they were already day-based.

────────────────────────────────────────────────────────────────────
Colab origin: load_real_data(), c_alıs_ır_son_hali_s_u_anlık.py
              lines ~150-310 (refactored to consume cleaned dataframes
              instead of re-reading raw CSVs; dedup added; Path-L
              line-level aggregation replaces unit-expand; day-based
              cohort index replaces to_period/period_range).
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd

from config import RESALABLE_REASONS, ORDER_STATUS_FILTER


# Period length in days. Single source of truth for both the cohort index
# and the intraperiod fraction.
PERIOD_DAYS = {"W": 7, "2W": 14, "3W": 21, "4W": 28}


def _cohort_index(dt_series: pd.Series, period: str, origin: pd.Timestamp) -> np.ndarray:
    """
    Integer cohort id for each timestamp: floor((date - origin)/period_days).

    Replaces pd.to_period(period), which is unreliable for multi-week
    frequencies (to_period("2W") silently yields weekly buckets). Using a
    day-based integer index makes grouping and the full cohort range share
    one definition, so reindex never drops rows.

    Colab origin: replaces the to_period() cohorting (~lines 180, 250).
    """
    if period not in PERIOD_DAYS:
        raise ValueError(f"Unknown period: {period}")
    n_days = PERIOD_DAYS[period]
    days = (dt_series - origin).dt.total_seconds() / (24 * 3600)
    return np.floor(days / n_days).astype("int64")


def _period_fraction(dt_series: pd.Series, period: str, origin: pd.Timestamp) -> np.ndarray:
    """
    Normalized position of each timestamp within its cohort bucket, in [0,1).

    Position within the same day-based bucket used by _cohort_index: the
    fractional part of (days since origin)/period_days. For "W" this is
    essentially day-of-week / 7; for multi-week periods it is the position
    within the n-day bucket.

    Colab origin: arrival_frac / return_frac computation (~lines 200, 290),
    re-anchored to the day-based cohort origin for internal consistency.
    """
    if period not in PERIOD_DAYS:
        raise ValueError(f"Unknown period: {period}")
    n_days = PERIOD_DAYS[period]
    days = (dt_series - origin).dt.total_seconds() / (24 * 3600)
    frac = (days / n_days) % 1.0
    return np.clip(frac.to_numpy(), 0.0, 1.0 - 1e-9)


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
            "panel": DataFrame indexed by cohort (integer id) with
                     [G_t, u_t, k_obs_t],
            "arrival_times": {cohort: np.ndarray of demand fractions},
            "return_arrival_times": {cohort: np.ndarray of return fracs},
          },
          ...
        }

    Notes
    -----
    - Returns are deduped to one per order_item (see _dedup_returns).
    - G_t, u_t, k_obs_t count ORDER LINES (order_items), not pieces
      (Path L, decision #6). k_obs_t <= u_t <= G_t.
    - Cohorts are day-based integer ids from a single global origin
      (decision #7), so total_G is invariant to the period definition.
    - arrival_times are LINE-LEVEL (one fraction per order_item line); for
      each cohort len(arrival_times[c]) == G_t[c].
    - return_arrival_times cover RESALABLE returns only (only resalable
      returns can offset demand in S4 order-of-arrival netting).
    """
    # ── 0. Global cohort origin (earliest fulfilled sale date) ──────
    # Single origin shared by demand AND returns and across categories, so
    # all panels live on the same cohort grid (sample-size axis comparable).
    items = order_items_clean.loc[
        order_items_clean["order_status"].isin(ORDER_STATUS_FILTER),
        ["order_item_id", "order_id", "order_date", "product_id",
         "quantity", "category_name"],
    ].copy()
    items["order_date"] = pd.to_datetime(items["order_date"], errors="coerce")
    items["quantity"] = pd.to_numeric(items["quantity"], errors="coerce").fillna(0).astype(int)
    items = items.dropna(subset=["order_date", "category_name"])

    origin = items["order_date"].min().normalize()

    # ── 1. Fulfilled demand (Completed + Returned) ──────────────────
    items["cohort"] = _cohort_index(items["order_date"], period, origin)
    items["arrival_frac"] = _period_fraction(items["order_date"], period, origin)

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

    # COHORT FIX: attribute return to the SALE cohort (order_date),
    # not the return period. Keeps U_t | G_t ~ Binomial(G_t, r).
    ret["cohort"] = _cohort_index(ret["order_date"], period, origin)

    # Intraperiod timing of the return uses return_date (when in the cycle
    # it physically arrives); the COHORT (which period) is fixed by sale.
    ret["return_frac"] = _period_fraction(ret["return_date"], period, origin)

    # ── 3. Per-category assembly ────────────────────────────────────
    datasets = {}
    categories = sorted(items["category_name"].unique())

    for cat in categories:
        items_c = items[items["category_name"] == cat]
        ret_c = ret[ret["category_name"] == cat]

        # ─ 3a. Period aggregates (LINE counts, Path L) ───────────
        # G_t = number of order lines sold in the cohort
        # u_t = number of returned lines (one return per order_item)
        # k_obs_t = number of resalable returned lines
        G = items_c.groupby("cohort").size()
        u = ret_c.groupby("cohort").size()
        k = ret_c[ret_c["is_resalable"]].groupby("cohort").size()

        # Contiguous cohort range from the SAME day-based index (decision
        # #7): grouping and reindex are aligned, so no row is dropped.
        all_cohorts = pd.RangeIndex(int(G.index.min()), int(G.index.max()) + 1)
        panel = pd.DataFrame(index=all_cohorts)
        panel.index.name = "cohort"
        panel["G_t"] = G.reindex(all_cohorts).fillna(0).astype(int)
        panel["u_t"] = u.reindex(all_cohorts).fillna(0).astype(int)
        panel["k_obs_t"] = k.reindex(all_cohorts).fillna(0).astype(int)

        # ─ 3b. Clip to maintain k_obs_t <= u_t <= G_t ────────────
        panel["u_t"] = np.minimum(panel["u_t"], panel["G_t"])
        panel["k_obs_t"] = np.minimum(panel["k_obs_t"], panel["u_t"])

        # ─ 3c. Line-level arrival timestamps ─────────────────────
        # Demand: ONE arrival fraction per order line (Path L). The line is
        # the atomic ordering/return unit, so the netting logic operates on
        # individual lines, NOT quantity-expanded pieces. (Reverses the
        # earlier np.repeat(..., quantity) expansion.)
        arrival_times = {}
        for c, grp in items_c.groupby("cohort"):
            fracs = grp["arrival_frac"].values
            arrival_times[c] = np.sort(fracs)

        # Resalable returns only, also line-level (one fraction per line).
        return_arrival_times = {}
        res_c = ret_c[ret_c["is_resalable"]]
        for c, grp in res_c.groupby("cohort"):
            fracs = grp["return_frac"].values
            return_arrival_times[c] = np.sort(fracs)

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
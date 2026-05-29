"""
Cohort Panel Construction (DÜZELTİLMİŞ)

Önceki versiyondaki hata: G_t ve u_t'nin cohort_start sütunları farklı
yöntemlerle hesaplanıyordu, merge eşleşmiyordu.

Düzeltme: Her iki tarafta da AYNI yöntem kullanılır:
  pd.Grouper(freq=PERIOD) — sales VE returns için aynı
"""

import pandas as pd
import numpy as np
import os
from data_analysis import order_items_clean, returns_clean


# ════════════════════════════════════════════════════════════
# Settings
# ════════════════════════════════════════════════════════════
PERIOD = "2W"
RESTOCKABLE_REASONS = {"Wrong Item", "Customer Preference"}


# ════════════════════════════════════════════════════════════
# 1. Sales side: G_t
# ════════════════════════════════════════════════════════════
sales = order_items_clean.loc[
    order_items_clean["order_status"].isin(["Completed", "Returned"]),
    ["order_item_id", "product_id", "order_date", "quantity"]
].copy()

sales["order_date"] = pd.to_datetime(sales["order_date"], errors="coerce")
sales = sales.dropna(subset=["order_date"])

G_t = (
    sales
    .groupby([
        "product_id",
        pd.Grouper(key="order_date", freq=PERIOD)
    ])["quantity"]
    .sum()
    .reset_index()
    .rename(columns={"order_date": "cohort_start", "quantity": "G_t"})
)


# ════════════════════════════════════════════════════════════
# 2. Returns side — AYNI Grouper kullan (sale period için)
# ════════════════════════════════════════════════════════════
returns = returns_clean[
    ["return_id", "order_item_id", "product_id", "return_reason",
     "order_date", "return_date", "quantity"]
].copy()

returns["order_date"] = pd.to_datetime(returns["order_date"], errors="coerce")
returns["return_date"] = pd.to_datetime(returns["return_date"], errors="coerce")
returns = returns.dropna(subset=["order_date"])

returns["is_restockable"] = returns["return_reason"].isin(RESTOCKABLE_REASONS)

# Cohort fix: returns sale period'a atfedilsin
# pd.Grouper(key='order_date') sale period kullanır
u_t = (
    returns
    .groupby([
        "product_id",
        pd.Grouper(key="order_date", freq=PERIOD)
    ])["quantity"]
    .sum()
    .reset_index()
    .rename(columns={"order_date": "cohort_start", "quantity": "u_t"})
)

k_t = (
    returns[returns["is_restockable"]]
    .groupby([
        "product_id",
        pd.Grouper(key="order_date", freq=PERIOD)
    ])["quantity"]
    .sum()
    .reset_index()
    .rename(columns={"order_date": "cohort_start", "quantity": "k_t"})
)


# ════════════════════════════════════════════════════════════
# 3. DEBUG: Tipleri ve örnek değerleri karşılaştır
# ════════════════════════════════════════════════════════════
print("=" * 60)
print("DEBUG — Tip ve örnek karşılaştırma")
print("=" * 60)
print(f"G_t cohort_start dtype : {G_t['cohort_start'].dtype}")
print(f"u_t cohort_start dtype : {u_t['cohort_start'].dtype}")
print(f"\nG_t ilk 3 cohort_start: {G_t['cohort_start'].head(3).tolist()}")
print(f"u_t ilk 3 cohort_start: {u_t['cohort_start'].head(3).tolist()}")
print(f"\nG_t toplam satır: {len(G_t)}, u_t: {len(u_t)}, k_t: {len(k_t)}")
print(f"u_t toplam quantity: {u_t['u_t'].sum()}")
print(f"k_t toplam quantity: {k_t['k_t'].sum()}")


# ════════════════════════════════════════════════════════════
# 4. Tam grid + boş cohort'ları doldur
# ════════════════════════════════════════════════════════════
all_products = G_t["product_id"].unique()
all_periods = pd.date_range(
    start=G_t["cohort_start"].min(),
    end=G_t["cohort_start"].max(),
    freq=PERIOD
)
full_index = pd.MultiIndex.from_product(
    [all_products, all_periods],
    names=["product_id", "cohort_start"]
)

G_t_full = (
    G_t
    .set_index(["product_id", "cohort_start"])
    .reindex(full_index, fill_value=0)
    .reset_index()
)

# Merge
cohort_panel = G_t_full.merge(u_t, on=["product_id", "cohort_start"], how="left")
cohort_panel = cohort_panel.merge(k_t, on=["product_id", "cohort_start"], how="left")
cohort_panel[["u_t", "k_t"]] = cohort_panel[["u_t", "k_t"]].fillna(0).astype(int)

# Clip u_t to G_t (replacement loops cause synthetic violations)
n_clipped_u = (cohort_panel["u_t"] > cohort_panel["G_t"]).sum()
cohort_panel["u_t"] = np.minimum(cohort_panel["u_t"], cohort_panel["G_t"])
# After clipping u_t, k_t may also exceed (rare)
n_clipped_k = (cohort_panel["k_t"] > cohort_panel["u_t"]).sum()
cohort_panel["k_t"] = np.minimum(cohort_panel["k_t"], cohort_panel["u_t"])

print(f"\nClipping uygulandı:")
print(f"  u_t clip edilen cohort  : {n_clipped_u}")
print(f"  k_t clip edilen cohort  : {n_clipped_k}")

# ════════════════════════════════════════════════════════════
# 5. Sanity checks
# ════════════════════════════════════════════════════════════
violations_u = (cohort_panel["u_t"] > cohort_panel["G_t"]).sum()
violations_k = (cohort_panel["k_t"] > cohort_panel["u_t"]).sum()
zero_G_pos_u = ((cohort_panel["G_t"] == 0) & (cohort_panel["u_t"] > 0)).sum()


# ════════════════════════════════════════════════════════════
# 6. Cohort panel özeti
# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"COHORT PANEL — Period: {PERIOD}")
print("=" * 60)
print(f"\nGrid yapısı:")
print(f"  Ürün sayısı           : {len(all_products)}")
print(f"  Cohort sayısı/ürün    : {len(all_periods)}")
print(f"  Toplam satır          : {len(cohort_panel)}")

print(f"\nSanity checks:")
print(f"  u_t > G_t ihlal       : {violations_u}")
print(f"  k_t > u_t ihlal       : {violations_k}")
print(f"  G_t=0 ama u_t>0       : {zero_G_pos_u}")

print(f"\nGenel istatistikler (cohort başına):")
stats = cohort_panel[["G_t", "u_t", "k_t"]].describe()
print(stats.round(2))

total_G = cohort_panel["G_t"].sum()
total_u = cohort_panel["u_t"].sum()
total_k = cohort_panel["k_t"].sum()
r_hat = total_u / total_G if total_G > 0 else 0
k_hat = total_k / total_u if total_u > 0 else 0

print(f"\nPooled Frequentist tahminler:")
print(f"  Toplam G_t            : {total_G:,}")
print(f"  Toplam u_t            : {total_u:,}")
print(f"  Toplam k_t            : {total_k:,}")
print(f"  r̂                     : {r_hat:.4f}")
print(f"  k̂                     : {k_hat:.4f}")
print(f"  r̂ × k̂                 : {r_hat * k_hat:.4f}")


# ════════════════════════════════════════════════════════════
# 7. Per-product breakdown
# ════════════════════════════════════════════════════════════
print(f"\nİlk 10 ürün özeti:")
per_product = (
    cohort_panel
    .groupby("product_id")
    .agg(
        total_G=("G_t", "sum"),
        total_u=("u_t", "sum"),
        total_k=("k_t", "sum"),
        n_cohorts=("G_t", "count"),
        nonzero_cohorts=("G_t", lambda x: (x > 0).sum())
    )
    .head(10)
)
per_product["r_hat"] = per_product["total_u"] / per_product["total_G"]
per_product["k_hat"] = per_product["total_k"] / per_product["total_u"].replace(0, np.nan)
print(per_product.round(4))


# ════════════════════════════════════════════════════════════
# 8. Kaydet
# ════════════════════════════════════════════════════════════
os.makedirs("data", exist_ok=True)
output_path = f"data/cohort_panel_{PERIOD}.csv"
cohort_panel.to_csv(output_path, index=False)
print(f"\nCohort panel kaydedildi: {output_path}")
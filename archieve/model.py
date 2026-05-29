"""
Demand Distribution Analysis (Manuel AIC bazlı)

distfit yerine scipy ile direkt manuel AIC karşılaştırması.
Test edilen dağılımlar: Poisson, NegBin, Geometric, Binomial.

Üç sorgu yapılır:
  1. Per-product fit (her ürün için en iyi dağılım)
  2. Pooled fit (tüm ürünleri tek havuzda — aile seçimi için)
  3. Period karşılaştırması (W, 2W, 3W, 4W)
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from data_analysis import order_items_clean, returns_clean


# ════════════════════════════════════════════════════════════
# 1. Demand data
# ════════════════════════════════════════════════════════════
fulfilled_orders = order_items_clean.loc[
    order_items_clean["order_status"].isin(["Completed", "Returned"]),
    ["product_id", "order_date", "quantity"]
].copy()



fulfilled_orders["order_date"] = pd.to_datetime(
    fulfilled_orders["order_date"], errors="coerce"
)
fulfilled_orders["quantity"] = pd.to_numeric(
    fulfilled_orders["quantity"], errors="coerce"
).fillna(0)
fulfilled_orders = fulfilled_orders.dropna(
    subset=["product_id", "order_date"]
)


# ════════════════════════════════════════════════════════════
# 2. Settings
# ════════════════════════════════════════════════════════════
candidates = ["W", "2W", "3W", "4W"]
min_obs = 20

all_results = {}
best_model_rows = []
pooled_results = {}


# ════════════════════════════════════════════════════════════
# 3. AIC fit fonksiyonu (manuel, scipy ile)
# ════════════════════════════════════════════════════════════
def fit_aic(data):
    """
    Discrete dağılımlar için AIC bazlı fit.
    Test edilen: Poisson, NegBin, Geometric, Binomial.
    
    Returns: dict with best, aic_table, params
    """
    if len(data) < 10:
        return None
    
    n = len(data)
    mean, var = data.mean(), data.var()
    results = {}

    # Poisson — λ = mean
    try:
        ll = stats.poisson.logpmf(data, mean).sum()
        if np.isfinite(ll):
            results['poisson'] = {
                'aic': 2*1 - 2*ll,
                'params': {'lambda': mean},
                'k': 1
            }
    except: pass

    # NegBin — sadece overdispersed (var > mean)
    try:
        if var > mean and mean > 0:
            p = mean / var
            r = mean * p / (1 - p)
            ll = stats.nbinom.logpmf(data, r, p).sum()
            if np.isfinite(ll):
                results['nbinom'] = {
                    'aic': 2*2 - 2*ll,
                    'params': {'r': r, 'p': p},
                    'k': 2
                }
    except: pass

    # Geometric — p = 1 / (1 + mean), 1-shifted
   # Geometric — 0-shifted, NegBin(r=1, p) olarak parametrize
    # p = 1 / (1 + mean), support {0, 1, 2, ...}
    try:
        if mean > 0:
            p_geo = 1 / (1 + mean)
            # nbinom(1, p) = 0-shifted geometric (scipy)
            ll = stats.nbinom.logpmf(data, 1, p_geo).sum()
            if np.isfinite(ll):
                results['geometric'] = {
                    'aic': 2*1 - 2*ll,
                    'params': {'p': p_geo},
                    'k': 1
                }
    except: pass

    # Binomial — n = max(data), p = mean / n
    try:
        n_param = int(data.max())
        if n_param > 0:
            p = mean / n_param
            ll = stats.binom.logpmf(data, n_param, p).sum()
            if np.isfinite(ll):
                results['binom'] = {
                    'aic': 2*2 - 2*ll,
                    'params': {'n': n_param, 'p': p},
                    'k': 2
                }
    except: pass

    if not results:
        return None
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['aic'])
    best_name = sorted_results[0][0]
    
    return {
        'best': best_name,
        'best_aic': sorted_results[0][1]['aic'],
        'best_params': sorted_results[0][1]['params'],
        'aic_table': sorted_results,
        'delta_runner_up': sorted_results[1][1]['aic'] - sorted_results[0][1]['aic'] if len(sorted_results) > 1 else None
    }


# ════════════════════════════════════════════════════════════
# 4. Period × ürün × dağılım fit
# ════════════════════════════════════════════════════════════
for freq in candidates:

    print(f"\n{'=' * 60}")
    print(f"PERIOD: {freq}")
    print(f"{'=' * 60}")

    # Period-level gross demand
    demand_panel = (
        fulfilled_orders
        .groupby([
            "product_id",
            pd.Grouper(key="order_date", freq=freq)
        ])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={
            "order_date": "cohort_start",
            "quantity": "G_t"
        })
    )

    # ─── Sıfır demand period'ları doldur ─────────────────────
    all_products = demand_panel["product_id"].unique()
    all_periods = pd.date_range(
        start=demand_panel["cohort_start"].min(),
        end=demand_panel["cohort_start"].max(),
        freq=freq
    )
    full_index = pd.MultiIndex.from_product(
        [all_products, all_periods],
        names=["product_id", "cohort_start"]
    )
    demand_panel = (
        demand_panel
        .set_index(["product_id", "cohort_start"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # ─── PER-PRODUCT FIT ─────────────────────────────────────
    fit_per_product = {}
    skipped = 0

    for pid, group in demand_panel.groupby("product_id"):

        data = group["G_t"].values.astype(int)

        if len(data) < min_obs:
            skipped += 1
            continue

        result = fit_aic(data)
        if result is None:
            skipped += 1
            continue

        fit_per_product[pid] = {
            "n_obs": len(data),
            "data": data,
            "result": result
        }

        best_model_rows.append({
            "freq": freq,
            "product_id": pid,
            "n_obs": len(data),
            "best_distribution": result['best'],
            "best_aic": result['best_aic'],
            "delta_runner_up": result['delta_runner_up'],
            "params": result['best_params']
        })

    # ─── POOLED FIT (tüm ürünler tek havuz) ──────────────────
    pooled_data = demand_panel["G_t"].values.astype(int)
    pooled_result = fit_aic(pooled_data)

    pooled_results[freq] = {
        "n_obs": len(pooled_data),
        "mean": pooled_data.mean(),
        "var": pooled_data.var(),
        "vmr": pooled_data.var() / pooled_data.mean() if pooled_data.mean() > 0 else np.nan,
        "result": pooled_result
    }

    all_results[freq] = {
        "demand_panel": demand_panel,
        "fit_per_product": fit_per_product,
        "skipped_products": skipped,
        "pooled": pooled_results[freq]
    }

    print(f"Per-product fit edilen : {len(fit_per_product)}")
    print(f"Per-product skip       : {skipped}")
    print(f"Pooled n_obs           : {len(pooled_data)}")
    print(f"Pooled mean / var / VMR: {pooled_data.mean():.2f} / {pooled_data.var():.2f} / {pooled_data.var()/pooled_data.mean():.2f}")


# ════════════════════════════════════════════════════════════
# 5. PER-PRODUCT KARŞILAŞTIRMA
# ════════════════════════════════════════════════════════════
best_models_df = pd.DataFrame(best_model_rows)

print(f"\n{'='*60}")
print("PER-PRODUCT — Period bazında en iyi dağılım dağılımı")
print(f"{'='*60}")
for freq in candidates:
    period_df = best_models_df[best_models_df['freq'] == freq]
    if len(period_df) == 0:
        continue
    counter = Counter(period_df['best_distribution'])
    avg_delta = period_df['delta_runner_up'].mean()
    print(f"\n{freq}: ({len(period_df)} ürün)")
    print(f"  Ortalama ΔAIC (runner-up'a fark): {avg_delta:.1f}")
    for dist, count in counter.most_common():
        share = count / len(period_df) * 100
        print(f"    {dist:15s} : {count:3d} ürün ({share:.1f}%)")


# ════════════════════════════════════════════════════════════
# 6. POOLED FIT KARŞILAŞTIRMA (asıl önemli kısım)
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("POOLED FIT — Tüm period'larda detaylı AIC sıralaması")
print(f"{'='*60}")

for freq in candidates:
    p = pooled_results[freq]
    if p['result'] is None:
        continue
    print(f"\n--- {freq} (n={p['n_obs']}, mean={p['mean']:.2f}, var={p['var']:.2f}, VMR={p['vmr']:.2f}) ---")
    for name, info in p['result']['aic_table']:
        params_str = ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in info['params'].items())
        print(f"  {name:15s} AIC: {info['aic']:>10.1f}  ({params_str})")


# ════════════════════════════════════════════════════════════
# 7. POOLED ÖZET TABLOSU
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("POOLED FIT — Özet")
print(f"{'='*60}")

pooled_summary = pd.DataFrame({
    freq: {
        "n_obs": pooled_results[freq]["n_obs"],
        "mean": round(pooled_results[freq]["mean"], 2),
        "var": round(pooled_results[freq]["var"], 2),
        "VMR": round(pooled_results[freq]["vmr"], 2),
        "best_dist": pooled_results[freq]["result"]["best"] if pooled_results[freq]["result"] else None,
        "delta_runner_up": round(pooled_results[freq]["result"]["delta_runner_up"], 1) if pooled_results[freq]["result"] else None,
    }
    for freq in candidates
}).T
print(pooled_summary)


# ════════════════════════════════════════════════════════════
# 8. ÜRÜN HOMOJENLİK KONTROLÜ
# ════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("ÜRÜN HOMOJENLİK KONTROLÜ")
print(f"{'='*60}")

product_stats = fulfilled_orders.groupby('product_id')['quantity'].agg(['mean', 'std', 'sum'])
product_stats['cv'] = product_stats['std'] / product_stats['mean']

print("\nÜrünler arası dağılım:")
print(product_stats.describe())

mean_range = product_stats['mean'].max() - product_stats['mean'].min()
cv_range = product_stats['cv'].max() - product_stats['cv'].min()
mean_pct = mean_range / product_stats['mean'].mean() * 100
cv_pct = cv_range / product_stats['cv'].mean() * 100

print(f"\nYorum:")
print(f"  Mean range  : {mean_range:.2f} (%{mean_pct:.1f})")
print(f"  CV range    : {cv_range:.2f} (%{cv_pct:.1f})")
print(f"  Eğer mean range küçük (<%20) → ürünler homojen → pooled fit güvenli")
print(f"  Eğer CV range küçük (<%20)   → varyasyon benzer → tek aile mantıklı")


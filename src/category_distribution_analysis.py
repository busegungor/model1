"""
Category-Level Demand Distribution Analysis (AIC bazlı)

Paperin kategori-seviye SAA yapısıyla uyumlu olacak şekilde demand
dağılımı analizi. Her kategori için bağımsız olarak haftalık aggregate
demand'in en iyi fit ettiği dağılım belirleniyor.

Test edilen dağılımlar: Poisson, NegBin, Geometric (0-shifted).

Üç çıktı üretilir:
  1. Per-category fit (her kategori için en iyi dağılım)
  2. Period karşılaştırması (W, 2W, 3W, 4W)
  3. Kategori homojenlik kontrolü

data_analysis.py'den 'order_items_clean' import ediliyor — bu DataFrame
zaten 'category_name' sütununu içeriyor.
"""

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from data_analysis import order_items_clean


# ════════════════════════════════════════════════════════════
# 1. Demand data
# ════════════════════════════════════════════════════════════
# order_items_clean zaten data_analysis.py'de drop_duplicates('order_item_id')
# yapılmış. order_status filtresi uygulayalım (paperdaki "delivered orders"
# tanımıyla uyumlu — sadece Completed ve Returned status'ler fulfilled sayılır).
fulfilled_orders = order_items_clean.loc[
    order_items_clean["order_status"].isin(["Completed", "Returned"]),
    ["product_id", "category_name", "order_date", "quantity"]
].copy()

fulfilled_orders["order_date"] = pd.to_datetime(
    fulfilled_orders["order_date"], errors="coerce"
)
fulfilled_orders["quantity"] = pd.to_numeric(
    fulfilled_orders["quantity"], errors="coerce"
).fillna(0)
fulfilled_orders = fulfilled_orders.dropna(
    subset=["product_id", "order_date", "category_name"]
)

print(f"Total fulfilled order_items   : {len(fulfilled_orders)}")
print(f"Number of categories          : {fulfilled_orders['category_name'].nunique()}")
print(f"Categories                    : {sorted(fulfilled_orders['category_name'].unique())}")
print(f"\nKategori başına satır sayısı:")
print(fulfilled_orders['category_name'].value_counts())


# ════════════════════════════════════════════════════════════
# 2. Settings
# ════════════════════════════════════════════════════════════
candidates = ["W", "2W", "3W", "4W"]
min_obs = 20

all_results = {}
best_model_rows = []


# ════════════════════════════════════════════════════════════
# 3. AIC fit fonksiyonu
# ════════════════════════════════════════════════════════════
def fit_aic(data):
    """
    Discrete dağılımlar için AIC bazlı fit.
    Test edilen: Poisson, NegBin, Geometric (0-shifted).
    
    Binomial çıkarıldı: demand için natural upper bound olmadığı için
    n = max(data) keyfi bir fit; metodolojik olarak savunulamaz.
    
    Geometric, scipy'nin nbinom(r=1, p) parametrizasyonuyla 0-shifted
    olarak hesaplanıyor — diğer dağılımlarla aynı support'ta olduğu
    için AIC karşılaştırması artık geçerli.
    
    Returns: dict with best, aic_table, params
    """
    if len(data) < 10:
        return None
    
    mean, var = data.mean(), data.var()
    results = {}

    # Poisson — λ = mean (MLE)
    try:
        ll = stats.poisson.logpmf(data, mean).sum()
        if np.isfinite(ll):
            results['poisson'] = {
                'aic': 2 * 1 - 2 * ll,
                'params': {'lambda': mean},
                'k': 1
            }
    except Exception:
        pass

    # NegBin — moment matching (var > mean gerekli)
    try:
        if var > mean and mean > 0:
            p = mean / var
            r = mean * p / (1 - p)
            ll = stats.nbinom.logpmf(data, r, p).sum()
            if np.isfinite(ll):
                results['nbinom'] = {
                    'aic': 2 * 2 - 2 * ll,
                    'params': {'r': r, 'p': p},
                    'k': 2
                }
    except Exception:
        pass

    # Geometric — 0-shifted, NegBin(r=1, p) olarak parametrize
    # support {0, 1, 2, ...}, diğer dağılımlarla comparable
    try:
        if mean > 0:
            p_geo = 1 / (1 + mean)
            ll = stats.nbinom.logpmf(data, 1, p_geo).sum()
            if np.isfinite(ll):
                results['geometric'] = {
                    'aic': 2 * 1 - 2 * ll,
                    'params': {'p': p_geo},
                    'k': 1
                }
    except Exception:
        pass

    if not results:
        return None
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['aic'])
    best_name = sorted_results[0][0]
    
    return {
        'best': best_name,
        'best_aic': sorted_results[0][1]['aic'],
        'best_params': sorted_results[0][1]['params'],
        'aic_table': sorted_results,
        'delta_runner_up': (
            sorted_results[1][1]['aic'] - sorted_results[0][1]['aic']
            if len(sorted_results) > 1 else None
        )
    }


# ════════════════════════════════════════════════════════════
# 4. Period × kategori × dağılım fit
# ════════════════════════════════════════════════════════════
for freq in candidates:

    print(f"\n{'=' * 60}")
    print(f"PERIOD: {freq}")
    print(f"{'=' * 60}")

    # Kategori-seviye aggregate demand panel
    # Her kategori için her period'da TOPLAM satılan birim sayısı
    demand_panel = (
        fulfilled_orders
        .groupby([
            "category_name",
            pd.Grouper(key="order_date", freq=freq)
        ])["quantity"]
        .sum()
        .reset_index()
        .rename(columns={
            "order_date": "cohort_start",
            "quantity": "G_t"
        })
    )

    # ─── Sıfır demand period'ları doldur (kategori bazında) ─────
    all_categories = demand_panel["category_name"].unique()
    all_periods = pd.date_range(
        start=demand_panel["cohort_start"].min(),
        end=demand_panel["cohort_start"].max(),
        freq=freq
    )
    full_index = pd.MultiIndex.from_product(
        [all_categories, all_periods],
        names=["category_name", "cohort_start"]
    )
    demand_panel = (
        demand_panel
        .set_index(["category_name", "cohort_start"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    # ─── PER-CATEGORY FIT ───────────────────────────────────
    fit_per_category = {}
    skipped = 0

    for cat, group in demand_panel.groupby("category_name"):

        data = group["G_t"].values.astype(int)

        if len(data) < min_obs:
            skipped += 1
            print(f"  Skip {cat}: only {len(data)} obs (< {min_obs})")
            continue

        result = fit_aic(data)
        if result is None:
            skipped += 1
            continue

        fit_per_category[cat] = {
            "n_obs": len(data),
            "mean": data.mean(),
            "var": data.var(),
            "vmr": data.var() / data.mean() if data.mean() > 0 else np.nan,
            "data": data,
            "result": result
        }

        best_model_rows.append({
            "freq": freq,
            "category_name": cat,
            "n_obs": len(data),
            "mean": data.mean(),
            "var": data.var(),
            "vmr": data.var() / data.mean() if data.mean() > 0 else np.nan,
            "best_distribution": result['best'],
            "best_aic": result['best_aic'],
            "delta_runner_up": result['delta_runner_up'],
            "params": result['best_params']
        })

    all_results[freq] = {
        "demand_panel": demand_panel,
        "fit_per_category": fit_per_category,
        "skipped": skipped
    }

    print(f"\nCategories fit edilen : {len(fit_per_category)}")
    print(f"Categories skip       : {skipped}")


# ════════════════════════════════════════════════════════════
# 5. PER-CATEGORY DETAYLI SONUÇ
# ════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("PER-CATEGORY — Period bazında detaylı AIC sıralaması")
print(f"{'=' * 60}")

for freq in candidates:
    print(f"\n{'─' * 60}")
    print(f"PERIOD: {freq}")
    print(f"{'─' * 60}")
    fits = all_results[freq]["fit_per_category"]
    for cat in sorted(fits.keys()):
        f = fits[cat]
        print(f"\n  {cat}")
        print(f"    n_obs={f['n_obs']}, mean={f['mean']:.2f}, "
              f"var={f['var']:.2f}, VMR={f['vmr']:.2f}")
        for name, info in f['result']['aic_table']:
            params_str = ', '.join(
                f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in info['params'].items()
            )
            marker = " ← best" if name == f['result']['best'] else ""
            print(f"      {name:12s} AIC: {info['aic']:>9.1f}  "
                  f"({params_str}){marker}")


# ════════════════════════════════════════════════════════════
# 6. PER-CATEGORY ÖZET TABLOSU
# ════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("PER-CATEGORY — Özet")
print(f"{'=' * 60}")

best_models_df = pd.DataFrame(best_model_rows)

# Pivot: kategori x period için best distribution
print("\nBest distribution by category × period:")
pivot_best = best_models_df.pivot(
    index="category_name", columns="freq", values="best_distribution"
)
print(pivot_best[candidates])

# Pivot: kategori x period için VMR
print("\nVMR by category × period (Poisson=1):")
pivot_vmr = best_models_df.pivot(
    index="category_name", columns="freq", values="vmr"
).round(2)
print(pivot_vmr[candidates])

# Pivot: kategori x period için ΔAIC (runner-up'a fark)
print("\nΔAIC to runner-up by category × period:")
pivot_delta = best_models_df.pivot(
    index="category_name", columns="freq", values="delta_runner_up"
).round(1)
print(pivot_delta[candidates])


# ════════════════════════════════════════════════════════════
# 7. PERIOD ÖZETİ (kaç kategori hangi dağılımı tercih ediyor)
# ════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("PERIOD ÖZETİ — En çok seçilen dağılım")
print(f"{'=' * 60}")

for freq in candidates:
    period_df = best_models_df[best_models_df['freq'] == freq]
    if len(period_df) == 0:
        continue
    counter = Counter(period_df['best_distribution'])
    avg_delta = period_df['delta_runner_up'].mean()
    avg_vmr = period_df['vmr'].mean()
    print(f"\n{freq}: ({len(period_df)} kategori, "
          f"ortalama VMR={avg_vmr:.2f}, ortalama ΔAIC={avg_delta:.1f})")
    for dist, count in counter.most_common():
        share = count / len(period_df) * 100
        print(f"    {dist:15s} : {count} kategori ({share:.0f}%)")


# ════════════════════════════════════════════════════════════
# 8. KATEGORİ HOMOJENLİK KONTROLÜ
# ════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("KATEGORİ HOMOJENLİK KONTROLÜ (weekly baseline)")
print(f"{'=' * 60}")

# Haftalık demand kategori bazında
weekly_demand = (
    fulfilled_orders
    .groupby([
        "category_name",
        pd.Grouper(key="order_date", freq="W")
    ])["quantity"]
    .sum()
    .reset_index()
)

category_stats = weekly_demand.groupby('category_name')['quantity'].agg(
    ['mean', 'std', 'sum']
)
category_stats['cv'] = category_stats['std'] / category_stats['mean']
category_stats['vmr'] = (category_stats['std'] ** 2) / category_stats['mean']

print("\nKategoriler arası dağılım (haftalık):")
print(category_stats.round(2))

mean_range = category_stats['mean'].max() - category_stats['mean'].min()
cv_range = category_stats['cv'].max() - category_stats['cv'].min()
mean_pct = mean_range / category_stats['mean'].mean() * 100
cv_pct = cv_range / category_stats['cv'].mean() * 100

print(f"\nYorum:")
print(f"  Mean range  : {mean_range:.2f} (%{mean_pct:.1f})")
print(f"  CV range    : {cv_range:.2f} (%{cv_pct:.1f})")
print(f"  Mean range büyük → kategoriler farklı scale'lerde "
      f"→ her kategori bağımsız modellenmeli (pool ETME)")
print(f"  CV range küçükse → varyasyon yapısı benzer "
      f"→ aynı aile (Poisson-gamma) tüm kategorilerde uygun olabilir")
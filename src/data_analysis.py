import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fitter import Fitter, get_common_distributions, get_distributions
from collections import Counter
from scipy import stats


pd.set_option('display.max_columns', None)
pd.set_option("display.width", 500)
df = pd.read_csv("/Users/busegungor/model1/data/master_data.csv", index_col=0, parse_dates=["order_date", "return_date"])
df_columns = ['order_item_id', 'order_id', 'order_date', 'order_status', 'product_id', 'quantity', 'item_price', 'product_name', 'category_id', 'category_name', 'unit_cost', 'unit_price', 'return_id', 'return_date', 'return_reason']
df = df[df_columns]


def df_check(dataframe, head = 5):
    print("Shape")
    print(dataframe.shape)
    print("Types")
    print(dataframe.dtypes)
    print("Columns")
    print(dataframe.columns)
    print("NA")
    print(dataframe.isnull().sum())


main = df[~(df["return_date"] < df["order_date"])].copy()

### Filling most needed columns

# return_reason: null = wasn't returned. Fill with "No Return" for grouping.
main["return_reason"] = main["return_reason"].fillna("No Return")


# tüm flag'leri tek bir mask'te topla
flags = (
    ((main['order_status']=='Returned') & (main['return_id'].isna())) |
    ((main['order_status']=='Cancelled') & (main['return_id'].notna()))
)


# temiz veri: flag'li satırları çıkar
clean = main[~flags].copy()

# clean.to_csv('/Users/busegungor/model1/data/main_clean.csv', index=False)



duplicates = clean[clean["order_item_id"].duplicated(keep=False)].sort_values("order_item_id")
# duplicates.to_csv('/Users/busegungor/model1/data/duplicates.csv', index=False)


# ════════════════════════════════════════════════════════════
# DESCRIPTIVE ANALYSIS — RAPOR İÇİN
# ════════════════════════════════════════════════════════════
# ─── Order vs Return level ayrımı ─────────────────────

order_items_clean = clean.drop_duplicates('order_item_id')
# Hem return_id var, hem de order_status iade ile uyumlu olsun
returns_clean = clean[clean['return_id'].notna() & clean['order_status'].isin(['Completed', 'Returned'])].copy()


# ════════════════════════════════════════════════════════════
# DESCRIPTIVE RAPOR — yalnızca `python data_analysis.py` ile çalışır,
# başka modüller import ettiğinde çalışmaz (import'lar sessiz kalır).
# ════════════════════════════════════════════════════════════
def descriptive_report():

    print("Her kombinasyonun sayısı:")
    print(main.groupby(['order_status', main['return_id'].notna()]).size())

    # return_id float64, NaN korunmuş → isna/notna doğru çalışır
    print("Completed, iade yok (normal satış):", len(main[(main['order_status']=='Completed') & (main['return_id'].isna())]))
    print("Completed, iade var (teslim sonrası iade):", len(main[(main['order_status']=='Completed') & (main['return_id'].notna())]))
    print("Cancelled, iade yok (kargolanmadan iptal):", len(main[(main['order_status']=='Cancelled') & (main['return_id'].isna())]))
    print("Cancelled, iade var (iptal + iade kaydı, incele):", len(main[(main['order_status']=='Cancelled') & (main['return_id'].notna())]))
    print("Returned, iade yok (VERİ KALİTESİ sorunu):", len(main[(main['order_status']=='Returned') & (main['return_id'].isna())]))
    print("Returned, iade var (normal):", len(main[(main['order_status']=='Returned') & (main['return_id'].notna())]))

    print(f"Toplam satır       : {len(main)}")
    print(f"Flag'li satır      : {flags.sum()}")
    print(f"Anlamlı (temiz)    : {(~flags).sum()}")
    print(f"Temiz oran         : {(~flags).mean():.1%}")

    print(f"Toplam return kaydı: {main['return_id'].notna().sum()}")
    # çıkarılan flagler
    print(f"  Cancelled + return_id (incele): {len(main[(main['order_status']=='Cancelled') & (main['return_id'].notna())])}")

    print(f"Kaydedildi: {len(clean)} satır")


    # ─── 1. MULTI-PRODUCT ORDER ANALİZİ ─────────────────────────
    print("="*60)
    print("1. SEPET DAVRANIŞI (Multi-product orders)")
    print("="*60)

    # Her order da kaç farklı ürün var?
    order_product_count = order_items_clean.groupby('order_id')['product_id'].nunique()

    print(f"Toplam unique order        : {len(order_product_count)}")
    print(f"Tek ürünlü order           : {(order_product_count == 1).sum()} "
          f"({(order_product_count == 1).mean():.1%})")
    print(f"Çoklu ürünlü order         : {(order_product_count > 1).sum()} "
          f"({(order_product_count > 1).mean():.1%})")

    print(f"\nFarklı ürün sayısı dağılımı:")
    print(order_product_count.value_counts().sort_index())

    print(f"\nQuantity dağılımı (order_item başına):")
    print(order_items_clean['quantity'].value_counts().sort_index())


    # ─── 4. SATIŞ HACMİ ZAMANSAL ────────────────────────────────
    print("\n" + "="*60)
    print("4. SATIŞ HACMİ (zamansal)")
    print("="*60)

    oi = order_items_clean.copy()
    oi['week'] = oi['order_date'].dt.to_period('W')
    oi['month'] = oi['order_date'].dt.to_period('M')

    # Aylık
    monthly = oi.groupby('month').agg(
        n_orders=('order_id', 'nunique'),
        units=('quantity', 'sum')
    )
    monthly['avg_units_per_order'] = (monthly['units'] / monthly['n_orders']).round(2) # Her sipariş ortalama kaç ünite içeriyor?
    print("\nAylık özet:")
    print(monthly)

    print(f"\nGenel ortalama:")
    print(f"  Toplam ay sayısı     : {len(monthly)}")
    print(f"  Ay başına order      : {monthly['n_orders'].mean():.0f}")
    print(f"  Ay başına ünite      : {monthly['units'].mean():.0f}")
    print(f"  En yoğun ay          : {monthly['units'].idxmax()} ({monthly['units'].max()} ünite)")
    print(f"  En sakin ay          : {monthly['units'].idxmin()} ({monthly['units'].min()} ünite)")

    # Haftalık kısa özet (model haftalık çalışacak diye)
    weekly = oi.groupby('week').agg(
        n_orders=('order_id', 'nunique'),
        units=('quantity', 'sum')
    )
    print(f"\nHaftalık özet:")
    print(f"  Toplam hafta sayısı  : {len(weekly)}")
    print(f"  Hafta başına ünite   : {weekly['units'].mean():.1f}")
    print(f"  Hafta başına std     : {weekly['units'].std():.1f}")


    # ─── 2. ÜRÜN BAZINDA AYLIK DEMAND ÖZETİ ─────────────────────
    print("\n" + "="*60)
    print("2. ÜRÜN BAZINDA AYLIK DEMAND")
    print("="*60)


    monthly_demand = oi.groupby(['product_id', 'month'])['quantity'].sum().reset_index()
    demand_stats = monthly_demand.groupby('product_id')['quantity'].agg([
        'mean', 'std', 'min', 'max'
    ]).round(2)
    demand_stats['cv'] = (demand_stats['std'] / demand_stats['mean']).round(2)


    # The standard deviation measures how far the average value lies from the mean,
    # whereas the coefficient of variation measures the ratio of the standard deviation to the mean.
    # The coefficient of variation is used to compare two or more data sets.
    # The lower the CV, the better the risk-return tradeoff.
    # CV neden kritik? Newsvendor'da safety stock kararını etkiler: Düşük CV → stabil ürün, az safety stock, Yüksek CV → değişken ürün, çok safety stock



    print(f"Genel demand stats (ürün ortalamaları):")
    print(f"  Ortalama mean    : {demand_stats['mean'].mean():.1f} adet/ay")
    print(f"  Ortalama CV      : {demand_stats['cv'].mean():.2f}")
    print(f"  En değişken ürün : {demand_stats['cv'].idxmax()} (CV={demand_stats['cv'].max():.2f})")
    print(f"  En stabil ürün   : {demand_stats['cv'].idxmin()} (CV={demand_stats['cv'].min():.2f})")

    # /Users/busegungor/model1/image/Screenshot 2026-05-06 at 11.33.30.png
    print(f"\nEn çok satan 5 ürün (toplam ünite bazında):")
    print(oi.groupby('product_id')['quantity'].sum().nlargest(5))
    print(f"\nEn az satan 5 ürün (toplam ünite bazında):")
    print(oi.groupby('product_id')['quantity'].sum().nsmallest(5))


    # ─── 3. RETURN ZAMANSAL ─────────────────────────────────────
    print("\n" + "="*60)
    print("3. RETURN ZAMANSAL DAĞILIMI")
    print("="*60)

    ret = returns_clean.copy()
    ret['return_week'] = ret['return_date'].dt.to_period('W')
    ret['return_month'] = ret['return_date'].dt.to_period('M')
    ret['lag_days'] = (ret['return_date'] - ret['order_date']).dt.days

    # Aylık return
    monthly_ret = ret.groupby('return_month').agg(
        n_events=('return_id', 'count'),
        units=('quantity', 'sum')
    )
    print("\nAylık return:")
    print(monthly_ret)


    # ─── Haftalık (model paralel) ───────────────────────────────
    weekly_ret = ret.groupby('return_week').agg(
        n_events=('return_id', 'count'),
        units=('quantity', 'sum')
    )
    print(f"\nHaftalık return özeti:")
    print(f"  Toplam hafta : {len(weekly_ret)}")
    print(f"  Hafta başına ortalama event : {weekly_ret['n_events'].mean():.1f}")
    print(f"  Hafta başına ortalama adet  : {weekly_ret['units'].mean():.1f}")
    print(f"  Hafta std (event)           : {weekly_ret['n_events'].std():.1f}")

    # Reason × ay
    print("\nReason × ay (event sayısı):")
    print(ret.groupby(['return_month', 'return_reason']).size().unstack(fill_value=0))

    # Return lag
    print("\nReturn lag (sipariş→iade gün):")
    print(ret['lag_days'].describe().round(1))

    print("\nReason'a göre lag:")
    print(ret.groupby('return_reason')['lag_days'].agg(['mean', 'median', 'std']).round(1))

    # ─── Reason × hafta haritası gerekirse ──────────────────────
    # 100+ hafta var, satır olarak yazdırmak okunamaz
    # Sadece reason bazlı haftalık ortalama göster:
    weekly_by_reason = ret.groupby(['return_week', 'return_reason']).size().unstack(fill_value=0)
    print("\nReason bazlı haftalık ortalama event:")
    print(weekly_by_reason.mean().round(1))
    print("\nReason bazlı haftalık std (volatilite):")
    print(weekly_by_reason.std().round(1))

    # Lag'i haftaya çevir
    ret['lag_weeks'] = ret['lag_days'] / 7

    print("\nReturn lag (hafta cinsinden):")
    print(ret['lag_weeks'].describe().round(1))

    print("\nReason'a göre lag (hafta):")
    print(ret.groupby('return_reason')['lag_weeks'].agg(['mean', 'median', 'std']).round(1))

    # Dağılımı yüzdeliklerle anlat (modelde lag-buffer hesabı için)
    print("\nLag yüzdelikleri (hafta):")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = ret['lag_weeks'].quantile(p/100)
        print(f"  %{p}: {val:.1f} hafta")

    # ─── 5. RATES (return ve refund oranları) ───────────────────
    print("\n" + "="*60)
    print("5. RATE'LER")
    print("="*60)

    # Item-level
    n_order_items = len(order_items_clean)
    n_returned_items = ret['order_item_id'].nunique()
    item_return_rate = n_returned_items / n_order_items

    # Event-level
    n_return_events = len(ret)
    event_return_rate = n_return_events / n_order_items

    # Unit-level (modelin r̂)
    total_units_sold = order_items_clean['quantity'].sum()
    total_units_returned = ret['quantity'].sum()
    unit_return_rate = total_units_returned / total_units_sold

    # Revenue (gross)
    order_items_clean['gross_revenue'] = order_items_clean['item_price'] * order_items_clean['quantity']
    total_revenue = order_items_clean['gross_revenue'].sum()

    # Refund'u biz türetiyoruz: dönen adet × satış fiyatı
    ret['implied_refund'] = ret['quantity'] * ret['item_price']
    total_refund = ret['implied_refund'].sum()
    refund_rate = total_refund / total_revenue

    print(f"Item-level return rate     : {item_return_rate:.2%}")
    print(f"Event-level return rate    : {event_return_rate:.2%}")
    print(f"Unit-level return rate     : {unit_return_rate:.2%} (modelin r̂'sine yakın)")
    print(f"\nGross revenue              : ${total_revenue:,.2f}")
    print(f"Implied refund (qty×price) : ${total_refund:,.2f}")
    print(f"Refund rate                : {refund_rate:.2%}")
    print(f"Net revenue                : ${total_revenue - total_refund:,.2f}")

    # Reason kırılımı (refund implied)
    reason_summary = ret.groupby('return_reason').agg(
        events=('return_id', 'count'),
        units=('quantity', 'sum'),
        refund=('implied_refund', 'sum')
    )
    reason_summary['event_share'] = (reason_summary['events'] / reason_summary['events'].sum()).round(3)
    print(f"\nReason kırılımı:")
    print(reason_summary)


    # ─── 6. MALİYETLER ──────────────────────────────────────────

    print("\n" + "="*60)
    print("6. PARASAL ÖZET")
    print("="*60)

    oi['gross_revenue'] = oi['item_price'] * oi['quantity']
    oi['cogs'] = oi['unit_cost'] * oi['quantity']

    ret['implied_refund'] = ret['quantity'] * ret['item_price']
    ret['cogs_at_risk'] = ret['quantity'] * ret['unit_cost']

    RESTOCKABLE = {'Wrong Item', 'Customer Preference'}
    ret['restockable'] = ret['return_reason'].isin(RESTOCKABLE)
    ret['cogs_written_off'] = ret['cogs_at_risk'] * (~ret['restockable'])

    print(f"Toplam gross revenue       : ${oi['gross_revenue'].sum():,.2f}")
    print(f"Toplam COGS                : ${oi['cogs'].sum():,.2f}")
    print(f"Toplam implied refund      : ${ret['implied_refund'].sum():,.2f}")
    print(f"Toplam yazılan COGS        : ${ret['cogs_written_off'].sum():,.2f}")
    print(f"\nOrtalama satış fiyatı       : ${oi['item_price'].mean():.2f}")
    print(f"Ortalama birim maliyet     : ${oi['unit_cost'].mean():.2f}")


if __name__ == "__main__":
    descriptive_report()
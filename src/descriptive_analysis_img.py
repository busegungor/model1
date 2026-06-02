"""
Descriptive Analysis - Görseller (güncellenmiş)

Önkoşul:
  - clean, order_items_clean, returns_clean dataframe'leri hazır
  - returns_clean['lag_weeks'] ve 'restockable' kolonları hazır
    (eğer yoksa script altında oluşturuyoruz)

Üretilen 6 görsel:
  1. Aylık satış trendi
  2. Haftalık satış volatilitesi
  3. Aylık satış vs return karşılaştırması
  4. Return reason dağılımı (event ve adet)
  5. Return lag (hafta) — reason bazında boxplot
  6. Order başına ürün sayısı + Quantity dağılımı (combined)
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from data_analysis import clean, order_items_clean, returns_clean
from pathlib import Path


BASE = Path(__file__).parent / 'visuals'
# ════════════════════════════════════════════════════════════
# STİL
# ════════════════════════════════════════════════════════════
plt.rcParams.update({
    'figure.dpi': 100,
    'savefig.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.titleweight': 'bold',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})

COLOR_SALES = '#2E86AB'
COLOR_RETURN = '#E63946'
COLOR_NEUTRAL = '#6C757D'
COLOR_RESTOCK = '#52B788'
COLOR_WRITEOFF = '#E76F51'

REASON_COLORS = {
    'Customer Preference': '#52B788',
    'Wrong Item':          '#74C0FC',
    'Damaged in Shipping': '#F4A261',
    'Defective':           '#E76F51',
}


# ════════════════════════════════════════════════════════════
# Hazırlık (eğer kolonlar yoksa)
# ════════════════════════════════════════════════════════════
RESTOCKABLE = {'Wrong Item', 'Customer Preference'}

oi = order_items_clean.copy()
oi['order_month'] = oi['order_date'].dt.to_period('M')
oi['order_week'] = oi['order_date'].dt.to_period('W')

ret = returns_clean.copy()
if 'lag_weeks' not in ret.columns:
    ret['lag_weeks'] = (ret['return_date'] - ret['order_date']).dt.days / 7
if 'restockable' not in ret.columns:
    ret['restockable'] = ret['return_reason'].isin(RESTOCKABLE)
ret['return_month'] = ret['return_date'].dt.to_period('M')


# ════════════════════════════════════════════════════════════
# 1. AYLIK SATIŞ TRENDİ
# ════════════════════════════════════════════════════════════
monthly_sales = oi.groupby('order_month').agg(
    n_orders=('order_id', 'nunique'),
    units=('quantity', 'sum')
).reset_index()
monthly_sales['date'] = monthly_sales['order_month'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(monthly_sales['date'], monthly_sales['units'],
        color=COLOR_SALES, linewidth=2, marker='o', markersize=5)
ax.fill_between(monthly_sales['date'], monthly_sales['units'],
                alpha=0.15, color=COLOR_SALES)
mean_units = monthly_sales['units'].mean()
ax.axhline(mean_units, color=COLOR_NEUTRAL, linestyle=':',
           linewidth=1.5, label=f'Ortalama: {mean_units:.0f} adet')
ax.set_title('Aylık Toplam Satış (Adet)', loc='left')
ax.set_xlabel('Ay')
ax.set_ylabel('Adet')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
ax.legend(loc='lower right', frameon=False)
plt.tight_layout()
plt.savefig(BASE / '01_monthly_sales.png', bbox_inches='tight')
plt.close()


# ════════════════════════════════════════════════════════════
# 2. HAFTALIK SATIŞ VOLATİLİTESİ
# ════════════════════════════════════════════════════════════
weekly_sales = oi.groupby('order_week')['quantity'].sum().reset_index()
weekly_sales['date'] = weekly_sales['order_week'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(weekly_sales['date'], weekly_sales['quantity'],
        color=COLOR_SALES, linewidth=1, alpha=0.6, label='Haftalık')
# 4 haftalık hareketli ortalama
weekly_sales['ma4'] = weekly_sales['quantity'].rolling(4, min_periods=1).mean()
ax.plot(weekly_sales['date'], weekly_sales['ma4'],
        color='#1A4D6E', linewidth=2, label='4-haftalık ortalama')

ax.set_title('Haftalık Satış Volatilitesi', loc='left')
ax.set_xlabel('Hafta')
ax.set_ylabel('Adet')
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
ax.legend(loc='upper right', frameon=False)
plt.tight_layout()
plt.savefig(BASE / '02_weekly_sales_volatility.png', bbox_inches='tight')
plt.close()


# ════════════════════════════════════════════════════════════
# 3. AYLIK SATIŞ VS RETURN
# ════════════════════════════════════════════════════════════
monthly_ret = ret.groupby('return_month')['quantity'].sum().reset_index()
monthly_ret.columns = ['return_month', 'units_returned']
monthly_ret['date'] = monthly_ret['return_month'].dt.to_timestamp()

fig, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.plot(monthly_sales['date'], monthly_sales['units'],
         color=COLOR_SALES, linewidth=2, marker='o', markersize=4, label='Satış')
ax1.set_xlabel('Ay')
ax1.set_ylabel('Satış (adet)', color=COLOR_SALES)
ax1.tick_params(axis='y', labelcolor=COLOR_SALES)

ax2 = ax1.twinx()
ax2.plot(monthly_ret['date'], monthly_ret['units_returned'],
         color=COLOR_RETURN, linewidth=2, marker='s', markersize=4, label='Return')
ax2.set_ylabel('Return (adet)', color=COLOR_RETURN)
ax2.tick_params(axis='y', labelcolor=COLOR_RETURN)
ax2.grid(False)
ax2.spines['top'].set_visible(False)

ax1.set_title('Aylık Satış vs Return', loc='left')
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(BASE / '03_sales_vs_returns.png', bbox_inches='tight')
plt.close()


# ════════════════════════════════════════════════════════════
# 4. RETURN REASON DAĞILIMI (event + adet)
# ════════════════════════════════════════════════════════════
reason_summary = ret.groupby('return_reason').agg(
    events=('return_id', 'count'),
    units=('quantity', 'sum')
).sort_values('events', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Event sayısı
colors = [REASON_COLORS.get(r, COLOR_NEUTRAL) for r in reason_summary.index]
bars1 = axes[0].barh(reason_summary.index, reason_summary['events'],
                     color=colors, edgecolor='white', linewidth=1.5)
total_e = reason_summary['events'].sum()
for bar, val in zip(bars1, reason_summary['events']):
    pct = val / total_e * 100
    axes[0].text(val + total_e*0.01, bar.get_y() + bar.get_height()/2,
                 f'{val} ({pct:.1f}%)', va='center', fontsize=9)
axes[0].set_title('Return Reason — Event Sayısı', loc='left')
axes[0].set_xlim(0, reason_summary['events'].max() * 1.18)

# Adet
bars2 = axes[1].barh(reason_summary.index, reason_summary['units'],
                     color=colors, edgecolor='white', linewidth=1.5)
total_u = reason_summary['units'].sum()
for bar, val in zip(bars2, reason_summary['units']):
    pct = val / total_u * 100
    axes[1].text(val + total_u*0.01, bar.get_y() + bar.get_height()/2,
                 f'{val} ({pct:.1f}%)', va='center', fontsize=9)
axes[1].set_title('Return Reason — Toplam Adet', loc='left')
axes[1].set_xlim(0, reason_summary['units'].max() * 1.18)

plt.tight_layout()
plt.savefig(BASE / '04_return_reasons.png', bbox_inches='tight')
plt.close()


# ════════════════════════════════════════════════════════════
# 5. RETURN LAG (HAFTA) - REASON BAZINDA BOXPLOT
# ════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 4.5))

reasons_order = sorted(ret['return_reason'].unique())
data = [ret[ret['return_reason']==r]['lag_weeks'].values for r in reasons_order]
colors = [REASON_COLORS.get(r, COLOR_NEUTRAL) for r in reasons_order]

bp = ax.boxplot(data, labels=reasons_order, patch_artist=True,
                medianprops=dict(color='black', linewidth=1.5),
                flierprops=dict(marker='o', markersize=3, alpha=0.4))
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_title('Return Lag (hafta) — Reason Bazında', loc='left')
ax.set_ylabel('Hafta (sipariş → iade)')
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(BASE / '05_return_lag_by_reason.png', bbox_inches='tight')
plt.close()


# ════════════════════════════════════════════════════════════
# 6. ORDER YAPISI (combined: order başına ürün + quantity)
# ════════════════════════════════════════════════════════════
order_product_count = order_items_clean.groupby('order_id')['product_id'].nunique()
opc_dist = order_product_count.value_counts().sort_index()
qty_counts = order_items_clean['quantity'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Order başına ürün
bars_o = axes[0].bar(opc_dist.index, opc_dist.values,
                     color=COLOR_NEUTRAL, edgecolor='white',
                     linewidth=1.5, width=0.7)
total_o = opc_dist.sum()
for bar, val in zip(bars_o, opc_dist.values):
    pct = val / total_o * 100
    axes[0].text(bar.get_x() + bar.get_width()/2, val + total_o*0.01,
                 f'{pct:.1f}%', ha='center', fontsize=9)
multi_pct = (order_product_count > 1).mean() * 100
axes[0].set_title(f'Order Başına Farklı Ürün (Multi: {multi_pct:.1f}%)', loc='left')
axes[0].set_xlabel('Farklı ürün sayısı')
axes[0].set_ylabel('Order')
axes[0].set_xticks(opc_dist.index)

# Quantity dağılımı
bars_q = axes[1].bar(qty_counts.index, qty_counts.values,
                     color=COLOR_SALES, edgecolor='white',
                     linewidth=1.5, width=0.7)
total_q = qty_counts.sum()
for bar, val in zip(bars_q, qty_counts.values):
    pct = val / total_q * 100
    axes[1].text(bar.get_x() + bar.get_width()/2, val + total_q*0.01,
                 f'{pct:.1f}%', ha='center', fontsize=9)
axes[1].set_title('Order Item Başına Quantity', loc='left')
axes[1].set_xlabel('Quantity')
axes[1].set_ylabel('Order item')
axes[1].set_xticks(qty_counts.index)

plt.tight_layout()
plt.savefig(BASE / '06_order_structure.png', bbox_inches='tight')
plt.close()


print("Tüm görseller kaydedildi:")
for fname in [
    '01_monthly_sales.png',
    '02_weekly_sales_volatility.png',
    '03_sales_vs_returns.png',
    '04_return_reasons.png',
    '05_return_lag_by_reason.png',
    '06_order_structure.png',
]:
    print(f"  {fname}")
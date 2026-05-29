"""Quick sanity check on sensitivity results."""
import pandas as pd
from config import TABLES_DIR

df = pd.read_csv(TABLES_DIR / "sensitivity_full.csv")
print(f"Toplam satır: {len(df)}")
print(f"Kategoriler: {df['category'].unique()}")
print(f"Period'lar: {df['period'].unique()}")
print(f"Rule'lar: {df['rule'].unique()}")
print(f"Method'lar: {df['method'].unique()}")
print(f"g değerleri: {sorted(df['g'].unique())}")
print(f"k değerleri: {sorted(df['k'].unique())}")
print(f"r değerleri: {sorted(df['r'].unique())}")

print("\n" + "="*70)
print("KONTROL 1 — Cost g ile artıyor mu? (baseline k=0.5, r=0.1, W, S4, SAA)")
print("="*70)
sub = df[(df['period']=='W') & (df['rule']=='S4') & (df['method']=='SAA')
         & (df['k']==0.5) & (df['r']==0.1)].sort_values('g')
print(sub[['g','Q_opt','cost']].to_string(index=False))
print("→ Q* ve cost g arttıkça artmalı (yüksek penalty → yüksek stok → yüksek maliyet)")

print("\n" + "="*70)
print("KONTROL 2 — SAA vs Bayesian farkı parametreye duyarlı mı?")
print("="*70)
piv = pd.read_csv(TABLES_DIR / "sensitivity_saa_vs_bayesian.csv")
sub = piv[(piv['period']=='W') & (piv['rule']=='S4') & (piv['k']==0.5) & (piv['r']==0.1)].sort_values('g')
print(sub[['g','cost_SAA','cost_Bayesian','bayes_minus_saa_cost']].to_string(index=False))
print("→ bayes_minus_saa_cost negatifse Bayesian kazanıyor, pozitifse SAA")

print("\n" + "="*70)
print("KONTROL 3 — Sequencing gap (S3-S1) g ile büyüyor mu?")
print("="*70)
seq = pd.read_csv(TABLES_DIR / "sensitivity_sequencing_gap.csv")
sub = seq[(seq['period']=='W') & (seq['method']=='SAA')
          & (seq['k']==0.5) & (seq['r']==0.1)].sort_values('g')
print(sub[['g','S1','S3','sequencing_gap_S3_S1']].to_string(index=False))
print("→ supervisor #9: sequencing ne zaman önemli? Gap büyüyorsa cevap budur.")

print("\n" + "="*70)
print("KONTROL 4 — Ekstrem parametrelerde fark açılıyor mu?")
print("="*70)
# Baseline cell vs ekstrem cell
baseline = df[(df['period']=='W') & (df['rule']=='S4') & (df['k']==0.5)
              & (df['r']==0.1) & (df['g']==8.0)]
extreme = df[(df['period']=='W') & (df['rule']=='S4') & (df['k']==0.95)
             & (df['r']==0.3) & (df['g']==100.0)]
print("Baseline (g=8, k=0.5, r=0.1):")
print(baseline[['method','Q_opt','cost']].to_string(index=False))
print("\nEkstrem (g=100, k=0.95, r=0.3):")
print(extreme[['method','Q_opt','cost']].to_string(index=False))
print("→ Ekstremde Q* ve cost ÇOK daha yüksek olmalı, SAA-Bayes farkı belki de açılır")
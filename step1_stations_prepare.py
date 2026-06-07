"""
1.3–1.5 QADAMLAR: Stansiya ma'lumotlarini tayyorlash
═════════════════════════════════════════════════════

1.3 Boysun → boysun_ready.csv
1.4 Denov  → denov_ready.csv
1.5 Regional (Boysun + Denov o'rtachasi) → regional_trend.csv

Baza davri: 1933–1960 (anomaliya hisoblash uchun)

pip install numpy pandas matplotlib openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR
# ══════════════════════════════════════════════════════════════

BOYSUN_FILE = 'Бойсун МС  8.xlsx'
BOYSUN_TEMP_SHEET = 'Ҳарорат'

DENOV_FILE = 'Denov_Mingchuqur.xlsx'
DENOV_SHEET = 'Денов'

BASE_START = 1933
BASE_END = 1960

# ══════════════════════════════════════════════════════════════
# KOD
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("  1.3–1.5: STANSIYA MA'LUMOTLARINI TAYYORLASH")
print(f"  Baza davri: {BASE_START}–{BASE_END}")
print("=" * 70)
print()

# ──────────────────────────────────────────
# 1.3 BOYSUN
# ──────────────────────────────────────────
print("━" * 70)
print("  1.3 BOYSUN MS — HARORAT")
print("━" * 70)
print()

df_raw = pd.read_excel(BOYSUN_FILE, sheet_name=BOYSUN_TEMP_SHEET)
cols = list(df_raw.columns)
print(f"  Fayl: {BOYSUN_FILE} / '{BOYSUN_TEMP_SHEET}'")
print(f"  Ustunlar: {cols}")
print()

years_b = pd.to_numeric(df_raw[cols[0]], errors='coerce')

if len(cols) > 13:
    month_data = df_raw[cols[1:13]].apply(pd.to_numeric, errors='coerce')
    T_annual_b = month_data.mean(axis=1)
    print("  Format: OYLIK → yillik o'rtacha hisoblandi")
elif len(cols) >= 2:
    T_annual_b = pd.to_numeric(df_raw[cols[1]], errors='coerce')
    print("  Format: YILLIK")

df_boysun = pd.DataFrame({'year': years_b, 'T_annual': T_annual_b})
df_boysun = df_boysun.dropna(subset=['year', 'T_annual']).reset_index(drop=True)
df_boysun['year'] = df_boysun['year'].astype(int)
df_boysun = df_boysun.sort_values('year').reset_index(drop=True)

base_mask_b = (df_boysun['year'] >= BASE_START) & (df_boysun['year'] <= BASE_END)
T_base_b = df_boysun.loc[base_mask_b, 'T_annual'].mean()
df_boysun['T_anomaly'] = df_boysun['T_annual'] - T_base_b

print(f"  Davr: {df_boysun['year'].min()}–{df_boysun['year'].max()} ({len(df_boysun)} yil)")
print(f"  T_base ({BASE_START}-{BASE_END}): {T_base_b:.2f}°C (n={base_mask_b.sum()})")
print(f"  T_anomaly: {df_boysun['T_anomaly'].min():.2f} dan {df_boysun['T_anomaly'].max():.2f}°C")

df_boysun.to_csv('boysun_ready.csv', index=False)
print(f"  💾 boysun_ready.csv")
print()

# ──────────────────────────────────────────
# 1.4 DENOV
# ──────────────────────────────────────────
print("━" * 70)
print("  1.4 DENOV — HARORAT")
print("━" * 70)
print()

df_raw = pd.read_excel(DENOV_FILE, sheet_name=DENOV_SHEET)
cols = list(df_raw.columns)
print(f"  Fayl: {DENOV_FILE} / '{DENOV_SHEET}'")
print(f"  Ustunlar: {cols}")

years_d = pd.to_numeric(df_raw[cols[0]], errors='coerce')
T_annual_d = pd.to_numeric(df_raw[cols[1]], errors='coerce')

df_denov = pd.DataFrame({'year': years_d, 'T_annual': T_annual_d})
df_denov = df_denov.dropna(subset=['year', 'T_annual']).reset_index(drop=True)
df_denov['year'] = df_denov['year'].astype(int)
df_denov = df_denov.sort_values('year').reset_index(drop=True)

base_mask_d = (df_denov['year'] >= BASE_START) & (df_denov['year'] <= BASE_END)
T_base_d = df_denov.loc[base_mask_d, 'T_annual'].mean()
df_denov['T_anomaly'] = df_denov['T_annual'] - T_base_d

print(f"  Davr: {df_denov['year'].min()}–{df_denov['year'].max()} ({len(df_denov)} yil)")
print(f"  T_base ({BASE_START}-{BASE_END}): {T_base_d:.2f}°C (n={base_mask_d.sum()})")
print(f"  T_anomaly: {df_denov['T_anomaly'].min():.2f} dan {df_denov['T_anomaly'].max():.2f}°C")

df_denov.to_csv('denov_ready.csv', index=False)
print(f"  💾 denov_ready.csv")
print()

# ──────────────────────────────────────────
# 1.5 REGIONAL TREND
# ──────────────────────────────────────────
print("━" * 70)
print("  1.5 REGIONAL TREND (Boysun + Denov)")
print("━" * 70)
print()

all_years = sorted(set(df_boysun['year']).union(set(df_denov['year'])))
df_regional = pd.DataFrame({'year': all_years})

boysun_anom = df_boysun.set_index('year')['T_anomaly']
denov_anom = df_denov.set_index('year')['T_anomaly']

df_regional['T_anomaly_boysun'] = df_regional['year'].map(boysun_anom)
df_regional['T_anomaly_denov'] = df_regional['year'].map(denov_anom)
df_regional['T_anomaly_regional'] = df_regional[['T_anomaly_boysun', 'T_anomaly_denov']].mean(axis=1)
df_regional['n_stations'] = df_regional[['T_anomaly_boysun', 'T_anomaly_denov']].notna().sum(axis=1)

print(f"  Davr: {df_regional['year'].min()}–{df_regional['year'].max()} ({len(df_regional)} yil)")
print(f"  Ikkala stansiya: {(df_regional['n_stations'] == 2).sum()} yil")
print(f"  Bitta stansiya:  {(df_regional['n_stations'] == 1).sum()} yil")
print(f"  Regional anomaliya: {df_regional['T_anomaly_regional'].min():.2f} dan "
      f"{df_regional['T_anomaly_regional'].max():.2f}°C")

df_regional.to_csv('regional_trend.csv', index=False)
print(f"  💾 regional_trend.csv")
print()

# ──────────────────────────────────────────
# GRAFIK
# ──────────────────────────────────────────
print("━" * 70)
print("  GRAFIK")
print("━" * 70)
print()

plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'figure.dpi': 150})
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig.subplots_adjust(hspace=0.15)

# a) Boysun
ax = axes[0]
colors_b = ['#E07060' if a >= 0 else '#6099C0' for a in df_boysun['T_anomaly']]
ax.bar(df_boysun['year'], df_boysun['T_anomaly'], color=colors_b, alpha=0.7, width=0.8)
ax.axhline(0, color='#888', lw=0.5)
ax.set_ylabel('ΔT (°C)', fontsize=11, fontweight='bold')
ax.set_title(f'Boysun MS (baza: {BASE_START}-{BASE_END})', fontsize=11, fontweight='bold')
ax.text(0.02, 0.95, 'a)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', color='#CC0000')

# b) Denov
ax = axes[1]
colors_d = ['#E07060' if a >= 0 else '#6099C0' for a in df_denov['T_anomaly']]
ax.bar(df_denov['year'], df_denov['T_anomaly'], color=colors_d, alpha=0.7, width=0.8)
ax.axhline(0, color='#888', lw=0.5)
ax.set_ylabel('ΔT (°C)', fontsize=11, fontweight='bold')
ax.set_title(f'Denov (baza: {BASE_START}-{BASE_END})', fontsize=11, fontweight='bold')
ax.text(0.02, 0.95, 'b)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', color='#CC0000')

# c) Regional
ax = axes[2]
reg = df_regional.dropna(subset=['T_anomaly_regional'])
colors_r = ['#E07060' if a >= 0 else '#6099C0' for a in reg['T_anomaly_regional']]
ax.bar(reg['year'], reg['T_anomaly_regional'], color=colors_r, alpha=0.7, width=0.8)
ma = reg.set_index('year')['T_anomaly_regional'].rolling(11, center=True, min_periods=6).mean()
ax.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0, label="11 yillik o'rtacha")
ax.axhline(0, color='#888', lw=0.5)
ax.set_xlabel('Yil', fontsize=11, fontweight='bold')
ax.set_ylabel('ΔT (°C)', fontsize=11, fontweight='bold')
ax.set_title(f'Regional (Boysun + Denov, baza: {BASE_START}-{BASE_END})', fontsize=11, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.text(0.02, 0.95, 'c)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', color='#CC0000')

plt.tight_layout()
plt.savefig('stations_ready_check.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  💾 stations_ready_check.png")
print()
print("=" * 70)
print("  ✨ 1.3–1.5 TAYYOR!")
print("=" * 70)
print()
print("  Yaratilgan fayllar:")
print("    • boysun_ready.csv")
print("    • denov_ready.csv")
print("    • regional_trend.csv")

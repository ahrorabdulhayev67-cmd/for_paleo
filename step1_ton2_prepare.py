"""
1.1-QADAM: TON-2 ma'lumotlarini tayyorlash
════════════════════════════════════════════

Kirish:  TON2_iso_original.csv (depth_mm, age_calBP, d13CcarbVPDB, d18OcarbVPDB)
Chiqish: ton2_ready.csv (age_ka, d18O, d13C — tozalangan, 500 yillik bin)

Jarayon:
  1. Yuklash va ustunlarni standartlashtirish
  2. NaN va outlier tozalash (3σ)
  3. Yoshga qarab saralash
  4. 500 yillik binning
  5. Saqlash

pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR
# ══════════════════════════════════════════════════════════════

INPUT_FILE = 'TON2_iso_original.csv'
OUTPUT_FILE = 'ton2_ready.csv'
BIN_SIZE = 500  # yil

# ══════════════════════════════════════════════════════════════
# KOD
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("  1.1-QADAM: TON-2 MA'LUMOTLARINI TAYYORLASH")
print("=" * 70)
print()

# ──────────────────────────────────────────
# 1. YUKLASH
# ──────────────────────────────────────────
print("━" * 70)
print("  1. YUKLASH")
print("━" * 70)
print()

df_raw = pd.read_csv(INPUT_FILE)
print(f"  Fayl: {INPUT_FILE}")
print(f"  Qatorlar: {len(df_raw)}")
print(f"  Ustunlar: {list(df_raw.columns)}")
print()

# Ustunlarni standartlashtirish
df = pd.DataFrame()
df['age_BP'] = pd.to_numeric(df_raw['age_calBP'], errors='coerce')
df['age_ka'] = df['age_BP'] / 1000

# δ¹⁸O topish
d18O_col = [c for c in df_raw.columns if 'd18o' in c.lower()][0]
df['d18O'] = pd.to_numeric(df_raw[d18O_col], errors='coerce')

# δ¹³C topish
d13C_col = [c for c in df_raw.columns if 'd13c' in c.lower()][0]
df['d13C'] = pd.to_numeric(df_raw[d13C_col], errors='coerce')

print(f"  δ¹⁸O ustuni: '{d18O_col}'")
print(f"  δ¹³C ustuni: '{d13C_col}'")
print(f"  Davr: {df['age_ka'].min():.1f} - {df['age_ka'].max():.1f} ka BP")
print(f"  δ¹⁸O: {df['d18O'].min():.2f} dan {df['d18O'].max():.2f}‰")
print(f"  δ¹³C: {df['d13C'].min():.2f} dan {df['d13C'].max():.2f}‰")
print()

# ──────────────────────────────────────────
# 2. NaN VA OUTLIER TOZALASH
# ──────────────────────────────────────────
print("━" * 70)
print("  2. TOZALASH")
print("━" * 70)
print()

n_start = len(df)

# NaN olib tashlash
df = df.dropna(subset=['age_ka', 'd18O'])
n_after_nan = len(df)
print(f"  NaN olib tashlandi: {n_start - n_after_nan} qator")

# Outlier (3σ) — δ¹⁸O
d18O_mean = df['d18O'].mean()
d18O_std = df['d18O'].std()
mask_d18O = np.abs(df['d18O'] - d18O_mean) < 3 * d18O_std

# Outlier (3σ) — δ¹³C
d13C_mean = df['d13C'].mean()
d13C_std = df['d13C'].std()
mask_d13C = np.abs(df['d13C'] - d13C_mean) < 3 * d13C_std

df = df[mask_d18O & mask_d13C].copy()
n_after_outlier = len(df)
print(f"  Outlier olib tashlandi (3σ): {n_after_nan - n_after_outlier} qator")
print(f"     δ¹⁸O: μ={d18O_mean:.2f}, σ={d18O_std:.2f}, chegara=[{d18O_mean-3*d18O_std:.2f}, {d18O_mean+3*d18O_std:.2f}]")
print(f"     δ¹³C: μ={d13C_mean:.2f}, σ={d13C_std:.2f}, chegara=[{d13C_mean-3*d13C_std:.2f}, {d13C_mean+3*d13C_std:.2f}]")
print(f"  Qoldi: {len(df)} nuqta")
print()

# ──────────────────────────────────────────
# 3. SARALASH (yosh bo'yicha)
# ──────────────────────────────────────────
print("━" * 70)
print("  3. SARALASH")
print("━" * 70)
print()

df = df.sort_values('age_ka').reset_index(drop=True)
print(f"  ✅ Yoshga qarab saralandi (kichikdan kattaga)")
print(f"  Birinchi: {df['age_ka'].iloc[0]:.2f} ka BP")
print(f"  Oxirgi:   {df['age_ka'].iloc[-1]:.2f} ka BP")
print()

# ──────────────────────────────────────────
# 4. 500 YILLIK BINNING
# ──────────────────────────────────────────
print("━" * 70)
print(f"  4. {BIN_SIZE} YILLIK BINNING")
print("━" * 70)
print()

age_min = df['age_ka'].min()
age_max = df['age_ka'].max()
bin_width_ka = BIN_SIZE / 1000  # 0.5 ka

bin_edges = np.arange(
    np.floor(age_min / bin_width_ka) * bin_width_ka,
    age_max + bin_width_ka,
    bin_width_ka
)

age_binned = []
d18O_binned = []
d18O_std_binned = []
d13C_binned = []
d13C_std_binned = []
n_points_binned = []

for i in range(len(bin_edges) - 1):
    mask = (df['age_ka'] >= bin_edges[i]) & (df['age_ka'] < bin_edges[i + 1])
    n = mask.sum()
    
    if n > 0:
        age_binned.append(df.loc[mask, 'age_ka'].mean())
        d18O_binned.append(df.loc[mask, 'd18O'].mean())
        d18O_std_binned.append(df.loc[mask, 'd18O'].std() if n > 1 else np.nan)
        d13C_binned.append(df.loc[mask, 'd13C'].mean())
        d13C_std_binned.append(df.loc[mask, 'd13C'].std() if n > 1 else np.nan)
        n_points_binned.append(n)

# DataFrame yaratish
df_binned = pd.DataFrame({
    'age_ka': age_binned,
    'age_BP': [a * 1000 for a in age_binned],
    'd18O': d18O_binned,
    'd18O_std': d18O_std_binned,
    'd13C': d13C_binned,
    'd13C_std': d13C_std_binned,
    'n_points': n_points_binned,
})

print(f"  Bin o'lchami: {BIN_SIZE} yil")
print(f"  Natija: {len(df_binned)} bin")
print(f"  Davr: {df_binned['age_ka'].min():.2f} - {df_binned['age_ka'].max():.2f} ka BP")
print(f"  O'rtacha nuqtalar/bin: {np.mean(n_points_binned):.1f}")
print(f"  Min/Max nuqtalar: {min(n_points_binned)} / {max(n_points_binned)}")
print()

# Statistika jadvali
print(f"  {'Parametr':<12} {'Min':<10} {'Max':<10} {'O\\'rtacha':<10} {'Std':<10}")
print(f"  {'─'*12} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
print(f"  {'δ¹⁸O':<12} {df_binned['d18O'].min():<10.2f} {df_binned['d18O'].max():<10.2f} "
      f"{df_binned['d18O'].mean():<10.2f} {df_binned['d18O'].std():<10.2f}")
print(f"  {'δ¹³C':<12} {df_binned['d13C'].min():<10.2f} {df_binned['d13C'].max():<10.2f} "
      f"{df_binned['d13C'].mean():<10.2f} {df_binned['d13C'].std():<10.2f}")
print()

# ──────────────────────────────────────────
# 5. SAQLASH
# ──────────────────────────────────────────
print("━" * 70)
print("  5. SAQLASH")
print("━" * 70)
print()

df_binned.to_csv(OUTPUT_FILE, index=False)
print(f"  💾 {OUTPUT_FILE}")
print(f"     {len(df_binned)} qator × {len(df_binned.columns)} ustun")
print(f"     Ustunlar: {list(df_binned.columns)}")
print()

# ──────────────────────────────────────────
# 6. TEKSHIRUV GRAFIGI
# ──────────────────────────────────────────
print("━" * 70)
print("  6. TEKSHIRUV GRAFIGI")
print("━" * 70)
print()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
fig.subplots_adjust(hspace=0.05)

# δ¹⁸O
ax1.plot(df['age_ka'], df['d18O'], '.', color='#CCCCCC', ms=2, alpha=0.5, label='Xom nuqtalar')
ax1.plot(df_binned['age_ka'], df_binned['d18O'], '-', color='#2C3E50', lw=1.5, label=f'{BIN_SIZE} yillik bin')
ax1.fill_between(df_binned['age_ka'],
                 df_binned['d18O'] - df_binned['d18O_std'].fillna(0),
                 df_binned['d18O'] + df_binned['d18O_std'].fillna(0),
                 alpha=0.2, color='#2C3E50')
ax1.invert_yaxis()
ax1.set_ylabel('δ¹⁸O (‰ VPDB)', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.text(0.02, 0.95, 'a) δ¹⁸O', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

# δ¹³C
ax2.plot(df['age_ka'], df['d13C'], '.', color='#D4E6F1', ms=2, alpha=0.5, label='Xom nuqtalar')
ax2.plot(df_binned['age_ka'], df_binned['d13C'], '-', color='#2874A6', lw=1.5, label=f'{BIN_SIZE} yillik bin')
ax2.fill_between(df_binned['age_ka'],
                 df_binned['d13C'] - df_binned['d13C_std'].fillna(0),
                 df_binned['d13C'] + df_binned['d13C_std'].fillna(0),
                 alpha=0.2, color='#2874A6')
ax2.set_xlabel('Yosh (ka BP)', fontsize=11, fontweight='bold')
ax2.set_ylabel('δ¹³C (‰ VPDB)', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.text(0.02, 0.95, 'b) δ¹³C', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

ax2.set_xlim(0, df_binned['age_ka'].max() + 2)

plt.tight_layout()
plt.savefig('ton2_ready_check.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  💾 ton2_ready_check.png")
print()
print("=" * 70)
print("  ✨ 1.1-QADAM TAYYOR!")
print("  Keyingi qadam: 1.2 — Transfer funksiya (ΔT hisoblash)")
print("=" * 70)

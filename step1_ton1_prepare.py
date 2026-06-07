"""
1.2-QADAM: TON-1 ma'lumotlarini tayyorlash
════════════════════════════════════════════

Kirish:  TON1_trace_original.csv (depth_mm, age_calBP, Sr/Ca, S/Ca)
Chiqish: ton1_ready.csv (age_ka, SrCa, SCa — tozalangan, 500 yillik bin)

Jarayon:
  1. Yuklash va ustunlarni standartlashtirish
  2. NaN va outlier tozalash (IQR usuli)
  3. Yoshga qarab saralash
  4. 500 yillik binning
  5. Saqlash

⚠️ MUHIM: INPUT_FILE nomini o'zingizning faylingizga o'zgartiring!

pip install numpy pandas matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR — SHU YERDA O'ZGARTIRING
# ══════════════════════════════════════════════════════════════

INPUT_FILE = 'TON1_trace_original.csv'   # ← O'ZGARTIRING!
OUTPUT_FILE = 'ton1_ready.csv'
BIN_SIZE = 500  # yil

# ══════════════════════════════════════════════════════════════
# KOD
# ══════════════════════════════════════════════════════════════

print("=" * 70)
print("  1.2-QADAM: TON-1 MA'LUMOTLARINI TAYYORLASH")
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
df['depth_mm'] = pd.to_numeric(df_raw['depth_mm'], errors='coerce')
df['age_BP'] = pd.to_numeric(df_raw['age_calBP'], errors='coerce')
df['age_ka'] = df['age_BP'] / 1000

# Sr/Ca va S/Ca topish
srca_col = [c for c in df_raw.columns if 'sr' in c.lower() and 'ca' in c.lower()]
sca_col = [c for c in df_raw.columns if 's/ca' in c.lower() or (c.lower().startswith('s') and 'ca' in c.lower() and 'sr' not in c.lower())]

if srca_col:
    df['SrCa'] = pd.to_numeric(df_raw[srca_col[0]], errors='coerce')
    print(f"  Sr/Ca ustuni: '{srca_col[0]}'")
else:
    print("  ❌ Sr/Ca ustuni topilmadi!")

if sca_col:
    df['SCa'] = pd.to_numeric(df_raw[sca_col[0]], errors='coerce')
    print(f"  S/Ca ustuni: '{sca_col[0]}'")
else:
    print("  ❌ S/Ca ustuni topilmadi!")

print(f"  Davr: {df['age_ka'].min():.1f} - {df['age_ka'].max():.1f} ka BP")
print(f"  Sr/Ca: {df['SrCa'].min():.6f} dan {df['SrCa'].max():.6f}")
print(f"  S/Ca:  {df['SCa'].min():.6f} dan {df['SCa'].max():.6f}")
print()

# ──────────────────────────────────────────
# 2. NaN VA OUTLIER TOZALASH (IQR usuli)
# ──────────────────────────────────────────
print("━" * 70)
print("  2. TOZALASH (IQR usuli)")
print("━" * 70)
print()

n_start = len(df)

# NaN olib tashlash
df = df.dropna(subset=['age_ka', 'SrCa', 'SCa'])
n_after_nan = len(df)
print(f"  NaN olib tashlandi: {n_start - n_after_nan} qator")

# Outlier (IQR × 1.5) — Sr/Ca
Q1_sr = df['SrCa'].quantile(0.25)
Q3_sr = df['SrCa'].quantile(0.75)
IQR_sr = Q3_sr - Q1_sr
lower_sr = Q1_sr - 1.5 * IQR_sr
upper_sr = Q3_sr + 1.5 * IQR_sr
mask_sr = (df['SrCa'] >= lower_sr) & (df['SrCa'] <= upper_sr)
n_outlier_sr = (~mask_sr).sum()

# Outlier (IQR × 1.5) — S/Ca
Q1_s = df['SCa'].quantile(0.25)
Q3_s = df['SCa'].quantile(0.75)
IQR_s = Q3_s - Q1_s
lower_s = Q1_s - 1.5 * IQR_s
upper_s = Q3_s + 1.5 * IQR_s
mask_s = (df['SCa'] >= lower_s) & (df['SCa'] <= upper_s)
n_outlier_s = (~mask_s).sum()

# Salbiy qiymatlarni ham olib tashlash
mask_positive = (df['SrCa'] > 0) & (df['SCa'] >= 0)

# Barcha filtrlarni birlashtirish
df = df[mask_sr & mask_s & mask_positive].copy()
n_after_outlier = len(df)

print(f"  Sr/Ca outlier (IQR×1.5): {n_outlier_sr} ta")
print(f"     Chegara: [{lower_sr:.6f}, {upper_sr:.6f}]")
print(f"  S/Ca outlier (IQR×1.5):  {n_outlier_s} ta")
print(f"     Chegara: [{lower_s:.6f}, {upper_s:.6f}]")
print(f"  Jami olib tashlandi: {n_after_nan - n_after_outlier} qator")
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

# Katta bo'shliqlarni aniqlash
age_diff = np.diff(df['age_ka'].values)
big_gaps = np.where(age_diff > 1.0)[0]  # > 1000 yil bo'shliq
if len(big_gaps) > 0:
    print(f"  ⚠️ Katta bo'shliqlar (>1000 yil):")
    for idx in big_gaps:
        gap_size = age_diff[idx]
        age_at = df['age_ka'].iloc[idx]
        print(f"     {age_at:.1f} ka BP da → {gap_size*1000:.0f} yillik uzilish")
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
SrCa_binned = []
SrCa_std_binned = []
SCa_binned = []
SCa_std_binned = []
n_points_binned = []

for i in range(len(bin_edges) - 1):
    mask = (df['age_ka'] >= bin_edges[i]) & (df['age_ka'] < bin_edges[i + 1])
    n = mask.sum()
    
    if n > 0:
        age_binned.append(df.loc[mask, 'age_ka'].mean())
        SrCa_binned.append(df.loc[mask, 'SrCa'].mean())
        SrCa_std_binned.append(df.loc[mask, 'SrCa'].std() if n > 1 else np.nan)
        SCa_binned.append(df.loc[mask, 'SCa'].mean())
        SCa_std_binned.append(df.loc[mask, 'SCa'].std() if n > 1 else np.nan)
        n_points_binned.append(n)

# DataFrame yaratish
df_binned = pd.DataFrame({
    'age_ka': age_binned,
    'age_BP': [a * 1000 for a in age_binned],
    'SrCa': SrCa_binned,
    'SrCa_std': SrCa_std_binned,
    'SCa': SCa_binned,
    'SCa_std': SCa_std_binned,
    'n_points': n_points_binned,
})

print(f"  Bin o'lchami: {BIN_SIZE} yil")
print(f"  Natija: {len(df_binned)} bin")
print(f"  Davr: {df_binned['age_ka'].min():.2f} - {df_binned['age_ka'].max():.2f} ka BP")
print(f"  O'rtacha nuqtalar/bin: {np.mean(n_points_binned):.1f}")
print(f"  Min/Max nuqtalar: {min(n_points_binned)} / {max(n_points_binned)}")
print()

# Statistika
print(f"  {'Parametr':<12} {'Min':<12} {'Max':<12} {'O\\'rtacha':<12} {'Std':<12}")
print(f"  {'─'*12} {'─'*12} {'─'*12} {'─'*12} {'─'*12}")
print(f"  {'Sr/Ca':<12} {df_binned['SrCa'].min():<12.6f} {df_binned['SrCa'].max():<12.6f} "
      f"{df_binned['SrCa'].mean():<12.6f} {df_binned['SrCa'].std():<12.6f}")
print(f"  {'S/Ca':<12} {df_binned['SCa'].min():<12.6f} {df_binned['SCa'].max():<12.6f} "
      f"{df_binned['SCa'].mean():<12.6f} {df_binned['SCa'].std():<12.6f}")
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

# Sr/Ca
ax1.plot(df['age_ka'], df['SrCa'], '.', color='#CCCCCC', ms=2, alpha=0.5, label='Xom nuqtalar')
ax1.plot(df_binned['age_ka'], df_binned['SrCa'], '-', color='#D73027', lw=1.5, label=f'{BIN_SIZE} yillik bin')
ax1.fill_between(df_binned['age_ka'],
                 df_binned['SrCa'] - df_binned['SrCa_std'].fillna(0),
                 df_binned['SrCa'] + df_binned['SrCa_std'].fillna(0),
                 alpha=0.2, color='#D73027')
ax1.set_ylabel('Sr/Ca', fontsize=11, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.text(0.02, 0.95, 'a) Sr/Ca', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
ax1.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

# S/Ca
ax2.plot(df['age_ka'], df['SCa'], '.', color='#D4E6F1', ms=2, alpha=0.5, label='Xom nuqtalar')
ax2.plot(df_binned['age_ka'], df_binned['SCa'], '-', color='#4393C3', lw=1.5, label=f'{BIN_SIZE} yillik bin')
ax2.fill_between(df_binned['age_ka'],
                 df_binned['SCa'] - df_binned['SCa_std'].fillna(0),
                 df_binned['SCa'] + df_binned['SCa_std'].fillna(0),
                 alpha=0.2, color='#4393C3')
ax2.set_xlabel('Yosh (ka BP)', fontsize=11, fontweight='bold')
ax2.set_ylabel('S/Ca', fontsize=11, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.text(0.02, 0.95, 'b) S/Ca', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')
ax2.ticklabel_format(axis='y', style='scientific', scilimits=(-4, -4))

ax2.set_xlim(0, df_binned['age_ka'].max() + 2)

plt.tight_layout()
plt.savefig('ton1_ready_check.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  💾 ton1_ready_check.png")
print()
print("=" * 70)
print("  ✨ 1.2-QADAM TAYYOR!")
print("  Keyingi qadam: Transfer funksiya (Sr/Ca → yog'ingarchilik)")
print("=" * 70)

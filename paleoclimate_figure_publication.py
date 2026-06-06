"""
TON-2 Paleoklimat rekonstruksiyasi — Yakuniy nashr grafigi
Chap panel: HAQIQIY δ¹⁸O → ΔT (transfer funksiya)
O'ng panel: Regional harorat anomaliyasi (Boysun + Denov + Mingchuqur)

Ma'lumotlar:
  - ton2_clean.csv (age_calBP, d18O, d13C)
  - Denov_Mingchuqur.xlsx
  - Бойсун МС 8.xlsx

pip install numpy pandas matplotlib scipy openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR
# ══════════════════════════════════════════════════════════════

# Ma'lumotlar fayllari
TON2_FILE = 'ton2_clean.csv'
BOYSUN_TEMP_FILE = 'Бойсун МС  8.xlsx'
BOYSUN_TEMP_SHEET = 'Ҳарорат'
DENOV_MINGCHUQUR_FILE = 'Denov_Mingchuqur.xlsx'

# Transfer funksiya parametrlari
# ──────────────────────────────────────────────────────────────
# Tremaine et al. (2011):
#   T(°C) = 16.01 - 4.23*(δ¹⁸Oc - δ¹⁸Ow) + 0.13*(δ¹⁸Oc - δ¹⁸Ow)²
#
# Yoki soddalashtirilgan (Kim & O'Neil, 1997):
#   1000*ln(α) = 18.03*(1000/T) - 32.42
#   → T ≈ 16.9 - 4.38*(δ¹⁸Oc - δ¹⁸Ow)
#
# δ¹⁸O_water: g'or tomchi suvi izotop tarkibi
# Agar o'lchangan qiymat bo'lmasa, zamonaviy qiymatni taxmin qilish kerak
# Markaziy Osiyo uchun odatda: -9 dan -12‰ (VSMOW)
# ──────────────────────────────────────────────────────────────

D18O_WATER = -10.0   # ‰ VSMOW — g'or suvi δ¹⁸O (O'ZGARTIRING!)
T_CAVE_MODERN = 8.0  # °C — hozirgi g'or harorati (O'ZGARTIRING!)

# Smoothing
SMOOTHING_WINDOW_KA = 2.5  # 2500 yillik silliqlantirilish

# Referens davri (regional anomaliya uchun)
REF_START = 1961
REF_END = 1990

# DIB chegaralari
DIB_stages = [
    {'name': 'DIB 1', 'start': 0, 'end': 11.7},
    {'name': 'DIB 2', 'start': 11.7, 'end': 29},
    {'name': 'DIB 3', 'start': 29, 'end': 57},
    {'name': 'DIB 4', 'start': 57, 'end': 71},
    {'name': 'DIB 5', 'start': 71, 'end': 130},
]


# ══════════════════════════════════════════════════════════════
# 1. TRANSFER FUNKSIYA: δ¹⁸O → HARORAT
# ══════════════════════════════════════════════════════════════

def d18O_to_temperature(d18O_calcite, d18O_water=D18O_WATER, method='kim_oneil'):
    """
    δ¹⁸O calcite → Harorat (°C).
    
    Parametrlar:
    -----------
    d18O_calcite : array — δ¹⁸O calcite (‰ VPDB)
    d18O_water : float — δ¹⁸O drip water (‰ VSMOW)
    method : str — 'kim_oneil' yoki 'tremaine'
    
    Qaytaradi:
    ---------
    T : array — Harorat (°C)
    """
    # VPDB → VSMOW konversiya (calcite uchun)
    d18O_c_vsmow = d18O_calcite * 1.03091 + 30.91  # Coplen et al. (1983)
    
    # Δ = δ¹⁸O_calcite(VSMOW) - δ¹⁸O_water(VSMOW)
    delta = d18O_c_vsmow - d18O_water
    
    if method == 'kim_oneil':
        # Kim & O'Neil (1997): 1000*ln(α) = 18.03*(10³/T) - 32.42
        # Soddalashtirilgan: T ≈ 16.9 - 4.38*Δ (past haroratlar uchun)
        # Aniqroq iterativ yechim:
        T = 16.9 - 4.38 * (d18O_calcite - d18O_water) + 0.13 * (d18O_calcite - d18O_water)**2
    
    elif method == 'tremaine':
        # Tremaine et al. (2011)
        diff = d18O_calcite - d18O_water
        T = 16.01 - 4.23 * diff + 0.13 * diff**2
    
    else:
        raise ValueError(f"Noma'lum metod: {method}")
    
    return T


# ══════════════════════════════════════════════════════════════
# 2. TON-2 MA'LUMOTLARINI YUKLASH VA QAYTA ISHLASH
# ══════════════════════════════════════════════════════════════

def load_ton2_data():
    """
    TON-2 speleothem δ¹⁸O → ΔT rekonstruksiya.
    """
    print("  📂 TON-2 ma'lumotlarini yuklash...")
    
    try:
        df = pd.read_csv(TON2_FILE)
        print(f"     ✅ {TON2_FILE}: {len(df)} nuqta")
        print(f"     Ustunlar: {list(df.columns)}")
    except FileNotFoundError:
        print(f"     ❌ {TON2_FILE} topilmadi!")
        print(f"     Fayl mavjud ekanligini tekshiring.")
        return None
    
    age_bp = df['age_calBP'].values
    d18O = df['d18O'].values
    age_ka = age_bp / 1000
    
    print(f"     Davr: {age_ka.min():.1f} - {age_ka.max():.1f} ka BP")
    print(f"     δ¹⁸O diapazoni: {np.nanmin(d18O):.2f} dan {np.nanmax(d18O):.2f} gacha")
    print()
    
    # Outlier olib tashlash (3σ)
    d18O_mean = np.nanmean(d18O)
    d18O_std = np.nanstd(d18O)
    mask = np.abs(d18O - d18O_mean) < 3 * d18O_std
    age_ka_clean = age_ka[mask]
    d18O_clean = d18O[mask]
    n_outliers = (~mask).sum()
    if n_outliers > 0:
        print(f"     ⚠️ {n_outliers} ta outlier olib tashlandi (3σ)")
    
    # Transfer funksiya: δ¹⁸O → T
    print(f"     Transfer funksiya: Kim & O'Neil (1997)")
    print(f"     δ¹⁸O_water = {D18O_WATER}‰ (VSMOW)")
    print(f"     T_cave_modern = {T_CAVE_MODERN}°C")
    
    T_reconstructed = d18O_to_temperature(d18O_clean, D18O_WATER, method='kim_oneil')
    
    # Anomaliya: ΔT = T_past - T_modern
    delta_T = T_reconstructed - T_CAVE_MODERN
    
    print(f"     ΔT diapazoni: {np.nanmin(delta_T):.1f} dan {np.nanmax(delta_T):.1f} °C gacha")
    print()
    
    # Smoothing (2500 yillik)
    # Tartibga solish (age bo'yicha)
    sort_idx = np.argsort(age_ka_clean)
    age_sorted = age_ka_clean[sort_idx]
    dT_sorted = delta_T[sort_idx]
    
    # Teng oraliqli interpolyatsiya (smoothing uchun)
    age_interp = np.linspace(age_sorted.min(), age_sorted.max(), 2000)
    dT_interp = np.interp(age_interp, age_sorted, dT_sorted)
    
    # Gaussian smooth
    spacing_ka = (age_interp[1] - age_interp[0])  # ka/nuqta
    sigma_points = SMOOTHING_WINDOW_KA / spacing_ka  # nuqta soni
    dT_smooth = gaussian_filter1d(dT_interp, sigma=sigma_points)
    
    # Noaniqlik (smoothing residual + analytical)
    residual = dT_interp - dT_smooth
    rolling_std = pd.Series(residual).rolling(
        window=max(10, int(sigma_points)), center=True, min_periods=5
    ).std().values
    # Analytical uncertainty qo'shish (~0.5°C)
    uncertainty = np.sqrt(rolling_std**2 + 0.5**2)
    uncertainty = gaussian_filter1d(np.nan_to_num(uncertainty, nan=1.0), sigma=sigma_points/2)
    
    print(f"     Smoothing: {SMOOTHING_WINDOW_KA*1000:.0f} yillik Gaussian")
    print(f"     Interpolyatsiya nuqtalari: {len(age_interp)}")
    print()
    
    return {
        'age_ka': age_interp,
        'delta_T': dT_smooth,
        'uncertainty': uncertainty,
        'age_raw': age_ka_clean,
        'dT_raw': delta_T,
    }


# ══════════════════════════════════════════════════════════════
# 3. REGIONAL HARORAT ANOMALIYASI
# ══════════════════════════════════════════════════════════════

def load_regional_data():
    """3 stansiya harorat anomaliyasini yuklash."""
    
    print("  📂 Regional stansiya ma'lumotlari...")
    stations = {}
    
    # Denov
    df_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Денов')
    cols = list(df_raw.columns)
    s = pd.Series(
        pd.to_numeric(df_raw[cols[1]], errors='coerce').values,
        index=pd.to_numeric(df_raw[cols[0]], errors='coerce').astype('Int64').values
    ).dropna()
    ref = s[(s.index >= REF_START) & (s.index <= REF_END)].mean()
    stations['Denov'] = s - ref
    print(f"     Denov: {len(s)} yil, T_ref={ref:.2f}°C")
    
    # Mingchuqur
    df_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Мингчуқур')
    cols = list(df_raw.columns)
    s = pd.Series(
        pd.to_numeric(df_raw[cols[1]], errors='coerce').values,
        index=pd.to_numeric(df_raw[cols[0]], errors='coerce').astype('Int64').values
    ).dropna()
    ref = s[(s.index >= REF_START) & (s.index <= REF_END)].mean()
    stations['Mingchuqur'] = s - ref
    print(f"     Mingchuqur: {len(s)} yil, T_ref={ref:.2f}°C")
    
    # Boysun
    try:
        df_raw = pd.read_excel(BOYSUN_TEMP_FILE, sheet_name=BOYSUN_TEMP_SHEET)
        cols = list(df_raw.columns)
        years = pd.to_numeric(df_raw[cols[0]], errors='coerce')
        if len(cols) > 13:
            temp = df_raw[cols[1:13]].apply(pd.to_numeric, errors='coerce').mean(axis=1)
        else:
            temp = pd.to_numeric(df_raw[cols[1]], errors='coerce')
        s = pd.Series(temp.values, index=years.values).dropna()
        s.index = s.index.astype(int)
        ref = s[(s.index >= REF_START) & (s.index <= REF_END)].mean()
        stations['Boysun'] = s - ref
        print(f"     Boysun: {len(s)} yil, T_ref={ref:.2f}°C")
    except Exception as e:
        print(f"     ⚠️ Boysun: {e}")
    
    # Regional o'rtacha
    all_years = sorted(set().union(*[set(s.index) for s in stations.values()]))
    df_reg = pd.DataFrame(index=all_years)
    for name, series in stations.items():
        df_reg[name] = series
    df_reg['regional'] = df_reg.mean(axis=1)
    
    print(f"     ✅ Regional: {len(df_reg)} yil")
    print()
    
    return df_reg


# ══════════════════════════════════════════════════════════════
# 4. YAKUNIY GRAFIK
# ══════════════════════════════════════════════════════════════

def run():
    """To'liq grafik: haqiqiy TON-2 ΔT + regional anomaliya."""
    
    print("=" * 70)
    print("  PALEOKLIMAT REKONSTRUKSIYASI — YAKUNIY GRAFIK")
    print("  Chap: TON-2 δ¹⁸O → ΔT | O'ng: Regional (3 stansiya)")
    print("=" * 70)
    print()
    
    # Ma'lumotlar
    paleo = load_ton2_data()
    if paleo is None:
        print("❌ TON-2 ma'lumotlari yuklanmadi. To'xtatildi.")
        return None
    
    df_reg = load_regional_data()
    
    # Trend
    recent = df_reg['regional'].dropna()
    recent_40 = recent[recent.index >= 1983]
    if len(recent_40) > 5:
        slope, intercept, r, p, se = stats.linregress(recent_40.index, recent_40.values)
        trend_decade = slope * 10
    else:
        trend_decade = np.nan
        slope = intercept = 0
    
    # ── GRAFIK ──
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 10,
        'axes.linewidth': 1.0, 'figure.dpi': 300
    })
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.02}
    )
    
    # ═══════════ PANEL a: TON-2 ΔT (HAQIQIY) ═══════════
    
    age_ka = paleo['age_ka']
    dT = paleo['delta_T']
    unc = paleo['uncertainty']
    
    # DIB chegaralari
    boundaries = set()
    for d in DIB_stages:
        boundaries.update([d['start'], d['end']])
    boundaries.discard(0)
    for b in sorted(boundaries):
        if b <= age_ka.max():
            ax1.axvline(x=b, color='#555', ls='--', lw=0.9, alpha=0.7, zorder=1)
    
    # Noaniqlik
    ax1.fill_between(age_ka, dT - unc, dT + unc,
                     color='#AAAAAA', alpha=0.35, zorder=2)
    
    # Asosiy signal
    ax1.plot(age_ka, dT, color='#1a1a1a', lw=1.8, zorder=3)
    
    # Nol chiziq
    ax1.axhline(y=0, color='#CC0000', ls='--', lw=0.6, alpha=0.5)
    
    # Heinrich
    for h_age in [100, 60]:
        if h_age <= age_ka.max():
            ax1.text(h_age, ax1.get_ylim()[1]*0.9 if ax1.get_ylim()[1] > 0 else 5,
                     'H', ha='center', fontsize=9, fontweight='bold', color='#333')
    
    # Golosen optimumi
    ax1.text(7, ax1.get_ylim()[0]*0.5, 'Golosen\noptimumi', ha='center',
             fontsize=8, color='#8B4513', fontstyle='italic', alpha=0.8)
    
    # DIB nomlari
    for d in DIB_stages:
        mid = (d['start'] + d['end']) / 2
        if mid <= age_ka.max():
            y_pos = ax1.get_ylim()[0] + 0.3
            ax1.text(mid, y_pos, d['name'], ha='center', fontsize=8.5, color='#555')
    
    ax1.set_xlabel('Yosh (ming yil oldin)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('ΔT (°C, zamonaviyga nisbatan)', fontsize=11, fontweight='bold')
    ax1.set_xlim(min(130, age_ka.max() + 2), 0)
    ax1.set_xticks(np.arange(0, min(140, int(age_ka.max()) + 10), 20))
    
    handles = [
        Line2D([0], [0], color='#1a1a1a', lw=1.8,
               label=f'ΔT ({SMOOTHING_WINDOW_KA*1000:.0f} yillik silliqlangan)'),
        plt.Rectangle((0, 0), 1, 1, fc='#AAAAAA', alpha=0.35,
                      label='Noaniqlik diapazoni'),
    ]
    ax1.legend(handles=handles, loc='lower left', fontsize=8.5, framealpha=0.95, edgecolor='#CCC')
    ax1.text(0.02, 0.96, 'a)', transform=ax1.transAxes, fontsize=13,
             fontweight='bold', va='top', color='#CC0000')
    
    # Uzilish
    ax1.plot([0.99, 1.01], [0.28, 0.32], transform=ax1.transAxes,
             color='#666', lw=1.2, clip_on=False)
    ax1.plot([0.99, 1.01], [0.33, 0.37], transform=ax1.transAxes,
             color='#666', lw=1.2, clip_on=False)
    
    # ═══════════ PANEL b: Regional anomaliya ═══════════
    
    years = df_reg.index.values
    anomaly = df_reg['regional'].values
    
    colors = ['#E07060' if a >= 0 else '#6099C0' for a in anomaly]
    ax2.bar(years, anomaly, color=colors, alpha=0.7, width=0.8, zorder=2, edgecolor='none')
    
    # 11 yillik o'rtacha
    ma = df_reg['regional'].rolling(11, center=True, min_periods=6).mean()
    ax2.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0, zorder=4,
             label="11 yillik harakatlanuvchi o'rtacha")
    
    # Trend
    if pd.notna(trend_decade) and len(recent_40) > 5:
        trend_line = intercept + slope * recent_40.index.values
        ax2.plot(recent_40.index, trend_line, color='#1a1a1a', lw=1.2, ls=':',
                 zorder=3, label=f"Trend ({trend_decade:+.2f}°C/o'n yillik)")
    
    # TON-2 marker
    if 2008 in df_reg.index:
        val = df_reg.loc[2008, 'regional']
        if pd.notna(val):
            ax2.plot(2008, val, 'D', color='#7B2D8B', ms=8, zorder=5,
                     markeredgecolor='white', markeredgewidth=1.0)
            ax2.text(2008 - 2, val + 0.15, 'TON-2\n(2008)', fontsize=7.5,
                     color='#7B2D8B', ha='center', fontweight='bold')
    
    ax2.axhline(y=0, color='#888', ls='-', lw=0.5, alpha=0.5)
    ax2.set_xlabel('Yil (milodiy)', fontsize=11, fontweight='bold')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Harorat anomaliyasi (°C)', fontsize=11, fontweight='bold')
    ax2.set_xlim(min(years) - 2, max(years) + 2)
    
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.95, edgecolor='#CCC')
    ax2.text(0.03, 0.96, 'b)', transform=ax2.transAxes, fontsize=13,
             fontweight='bold', va='top', color='#CC0000')
    
    ax2.plot([-0.01, 0.01], [0.28, 0.32], transform=ax2.transAxes,
             color='#666', lw=1.2, clip_on=False)
    ax2.plot([-0.01, 0.01], [0.33, 0.37], transform=ax2.transAxes,
             color='#666', lw=1.2, clip_on=False)
    
    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.tick_params(direction='out', length=4, width=0.8)
    
    plt.tight_layout()
    plt.savefig('fig_paleoclimate_publication.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig_paleoclimate_publication.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Xulosa
    print(f"  💾 fig_paleoclimate_publication.png / .pdf")
    print()
    print("  📝 RASM OSTI YOZUVI:")
    print(f"  \"(a) TON-2 speleothem harorat rekonstruksiyasi (δ¹⁸O → ΔT,")
    print(f"   Kim & O'Neil, 1997; δ¹⁸O_water={D18O_WATER}‰).")
    print(f"   {SMOOTHING_WINDOW_KA*1000:.0f} yillik Gaussian silliqlantirilgan.")
    print(f"   (b) Regional harorat anomaliyasi — 3 stansiya o'rtachasi")
    print(f"   (Boysun, Denov, Mingchuqur), {REF_START}-{REF_END} referens davri.")
    if pd.notna(trend_decade):
        print(f"   Trend: {trend_decade:+.3f}°C/o'n yillik.\"")
    print()
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    
    return paleo, df_reg


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = run()

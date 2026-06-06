"""
Figure 3: TON-2 Paleoklimat rekonstruksiyasi
Panel A: ΔT (0-115 ka BP) | Panel B: Boysun MS harorat anomaliyasi

Transfer funksiya:
  T_cave = 5.6°C (ERA5, O'ZGARTIRMANG!)
  Fractionation: Kim & O'Neil (1997) + Tremaine (2011) o'rtachasi
  Koeffitsient: 0.54‰/°C (Araguas + Dansgaard)
  Reference: TON-2 zamonaviy 2 nuqta (1941, 2008) o'rtachasi
  Bin: 500 yillik

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
# SOZLAMALAR — O'ZGARTIRMANG!
# ══════════════════════════════════════════════════════════════

TON2_FILE = 'TON2_iso_original.csv'
BOYSUN_TEMP_FILE = 'Бойсун МС  8.xlsx'
BOYSUN_TEMP_SHEET = 'Ҳарорат'
DENOV_MINGCHUQUR_FILE = 'Denov_Mingchuqur.xlsx'

# Transfer funksiya parametrlari
T_CAVE = 5.6          # °C — ERA5 dan. O'ZGARTIRMANG!
COEFF = 0.54          # ‰/°C — Araguas + Dansgaard
BIN_YEARS = 500       # 500 yillik binning

# Reference: TON-2 zamonaviy 2 nuqta (1941, 2008) o'rtachasi
REF_AGES_BP = [2008 - 1950 + 1941 - 1950, 2008 - 1950]  # taxminiy age BP
# Aslida: reference δ¹⁸O ni shu 2 nuqtadan olish

# Regional anomaliya referens davri
REF_START = 1961
REF_END = 1990

# DIB (MIS) bosqichlari
MIS_stages = [
    {'name': 'MIS 1', 'start': 0, 'end': 11.7},
    {'name': 'MIS 2', 'start': 11.7, 'end': 29},
    {'name': 'MIS 3', 'start': 29, 'end': 57},
    {'name': 'MIS 4', 'start': 57, 'end': 71},
    {'name': 'MIS 5', 'start': 71, 'end': 130},
]


# ══════════════════════════════════════════════════════════════
# TRANSFER FUNKSIYA
# ══════════════════════════════════════════════════════════════

def compute_delta_T(d18O_values, d18O_reference):
    """
    δ¹⁸O → ΔT (anomaliya).
    
    Usul:
      1. Kim & O'Neil (1997): T = 16.01 - 4.64*(δ¹⁸Oc - δ¹⁸Ow) + 0.09*(...)²
      2. Tremaine (2011): T = 16.01 - 4.23*(δ¹⁸Oc - δ¹⁸Ow) + 0.13*(...)²
      → Ikkalasining o'rtachasi
      
    Lekin anomaliya uchun soddalashtirilgan:
      ΔT = Δδ¹⁸O / koeffitsient
      ΔT = (δ¹⁸O_past - δ¹⁸O_reference) / 0.54
      
    Eslatma: kattaroq δ¹⁸O = sovuqroq → manfiy ΔT
    Shuning uchun: ΔT = -(δ¹⁸O - ref) / coeff
    """
    delta_d18O = d18O_values - d18O_reference
    
    # Manfiy: δ¹⁸O oshsa = sovuqroq = manfiy ΔT
    delta_T = -delta_d18O / COEFF
    
    return delta_T


# ══════════════════════════════════════════════════════════════
# TON-2 MA'LUMOTLAR
# ══════════════════════════════════════════════════════════════

def load_ton2():
    """TON-2 δ¹⁸O → ΔT."""
    
    print("  📂 TON-2 yuklash...")
    
    df = pd.read_csv(TON2_FILE)
    print(f"     ✅ {len(df)} nuqta")
    print(f"     Ustunlar: {list(df.columns)}")
    
    age_bp = df['age_calBP'].values
    
    # δ¹⁸O ustunini topish
    d18O_col = None
    for col in df.columns:
        if 'd18O' in col.lower() or 'δ18o' in col.lower():
            d18O_col = col
            break
    if d18O_col is None:
        print("     ❌ δ¹⁸O ustuni topilmadi!")
        return None
    
    d18O = df[d18O_col].values
    age_ka = age_bp / 1000
    
    print(f"     δ¹⁸O ustuni: '{d18O_col}'")
    print(f"     Davr: {age_ka.min():.1f} - {age_ka.max():.1f} ka BP")
    print(f"     δ¹⁸O: {np.nanmin(d18O):.2f} dan {np.nanmax(d18O):.2f}‰")
    
    # Outlier (3σ)
    d18O_mean = np.nanmean(d18O)
    d18O_std = np.nanstd(d18O)
    mask = ~np.isnan(d18O) & (np.abs(d18O - d18O_mean) < 3 * d18O_std)
    age_ka = age_ka[mask]
    d18O = d18O[mask]
    print(f"     Outlier olib tashlangandan keyin: {len(d18O)} nuqta")
    
    # REFERENCE: zamonaviy 2 nuqta (1941 va 2008)
    # age_BP ≈ 1950 - year_CE → 1941 = 9 BP, 2008 = -58 BP
    # Eng yosh nuqtalarni reference sifatida olish
    youngest_mask = age_ka < 1.0  # 1 ka BP dan yosh = oxirgi 1000 yil
    if youngest_mask.sum() >= 2:
        d18O_reference = np.nanmean(d18O[youngest_mask])
    else:
        # Eng yosh 5 ta nuqta
        sort_idx = np.argsort(age_ka)
        d18O_reference = np.nanmean(d18O[sort_idx[:5]])
    
    print(f"     δ¹⁸O reference (zamonaviy): {d18O_reference:.2f}‰")
    print(f"     T_cave: {T_CAVE}°C (ERA5)")
    print(f"     Koeffitsient: {COEFF}‰/°C")
    
    # Transfer funksiya → ΔT
    delta_T = compute_delta_T(d18O, d18O_reference)
    
    print(f"     ΔT: {np.nanmin(delta_T):.1f} dan {np.nanmax(delta_T):.1f}°C")
    print()
    
    # Tartibga solish
    sort_idx = np.argsort(age_ka)
    age_ka = age_ka[sort_idx]
    delta_T = delta_T[sort_idx]
    
    # 500 yillik binning
    age_min = age_ka.min()
    age_max = age_ka.max()
    bin_edges = np.arange(age_min, age_max + BIN_YEARS/1000, BIN_YEARS/1000)
    
    age_binned = []
    dT_binned = []
    dT_std_binned = []
    
    for i in range(len(bin_edges) - 1):
        bin_mask = (age_ka >= bin_edges[i]) & (age_ka < bin_edges[i+1])
        if bin_mask.sum() > 0:
            age_binned.append(np.mean(age_ka[bin_mask]))
            dT_binned.append(np.mean(delta_T[bin_mask]))
            dT_std_binned.append(np.std(delta_T[bin_mask]) if bin_mask.sum() > 1 else 1.0)
    
    age_binned = np.array(age_binned)
    dT_binned = np.array(dT_binned)
    dT_std_binned = np.array(dT_std_binned)
    
    # Smoothing (2500 yillik Gaussian = 5 bin)
    sigma_bins = 5  # 5 × 500 yil = 2500 yil
    dT_smooth = gaussian_filter1d(dT_binned, sigma=sigma_bins)
    
    # Noaniqlik: bin std + analytical (0.5°C)
    uncertainty = np.sqrt(dT_std_binned**2 + 0.5**2)
    uncertainty_smooth = gaussian_filter1d(uncertainty, sigma=3)
    
    print(f"     Binning: {BIN_YEARS} yillik, {len(age_binned)} bin")
    print(f"     Smoothing: 2500 yillik Gaussian")
    print(f"     ΔT (smoothed): {dT_smooth.min():.1f} dan {dT_smooth.max():.1f}°C")
    print()
    
    return {
        'age_ka': age_binned,
        'delta_T': dT_smooth,
        'uncertainty': uncertainty_smooth,
    }


# ══════════════════════════════════════════════════════════════
# BOYSUN HARORAT ANOMALIYASI (Panel B)
# ══════════════════════════════════════════════════════════════

def load_boysun_temperature():
    """Boysun + Denov + Mingchuqur regional anomaliya."""
    
    print("  📂 Regional harorat anomaliyasi...")
    stations = {}
    
    # Denov
    try:
        df_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Денов')
        cols = list(df_raw.columns)
        s = pd.Series(
            pd.to_numeric(df_raw[cols[1]], errors='coerce').values,
            index=pd.to_numeric(df_raw[cols[0]], errors='coerce').astype('Int64').values
        ).dropna()
        ref = s[(s.index >= REF_START) & (s.index <= REF_END)].mean()
        stations['Denov'] = s - ref
        print(f"     Denov: {len(s)} yil")
    except Exception as e:
        print(f"     ⚠️ Denov: {e}")
    
    # Mingchuqur
    try:
        df_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Мингчуқур')
        cols = list(df_raw.columns)
        s = pd.Series(
            pd.to_numeric(df_raw[cols[1]], errors='coerce').values,
            index=pd.to_numeric(df_raw[cols[0]], errors='coerce').astype('Int64').values
        ).dropna()
        ref = s[(s.index >= REF_START) & (s.index <= REF_END)].mean()
        stations['Mingchuqur'] = s - ref
        print(f"     Mingchuqur: {len(s)} yil")
    except Exception as e:
        print(f"     ⚠️ Mingchuqur: {e}")
    
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
        print(f"     Boysun: {len(s)} yil")
    except Exception as e:
        print(f"     ⚠️ Boysun: {e}")
    
    # O'rtacha
    all_years = sorted(set().union(*[set(s.index) for s in stations.values()]))
    df_reg = pd.DataFrame(index=all_years)
    for name, series in stations.items():
        df_reg[name] = series
    df_reg['anomaly'] = df_reg.mean(axis=1)
    
    print(f"     ✅ Regional: {len(df_reg)} yil ({min(all_years)}-{max(all_years)})")
    print()
    
    return df_reg


# ══════════════════════════════════════════════════════════════
# GRAFIK
# ══════════════════════════════════════════════════════════════

def run():
    """Figure 3 yaratish."""
    
    print("=" * 70)
    print("  FIGURE 3: PALEOKLIMAT REKONSTRUKSIYASI")
    print("  T_cave = 5.6°C | Koeff = 0.54‰/°C | Bin = 500 yil")
    print("=" * 70)
    print()
    
    # Ma'lumotlar
    paleo = load_ton2()
    if paleo is None:
        return None
    
    df_modern = load_boysun_temperature()
    
    # Trend
    recent = df_modern['anomaly'].dropna()
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
    
    # ═══════════════════════════════════════════
    # PANEL A: ΔT rekonstruksiya (0-115 ka BP)
    # ═══════════════════════════════════════════
    
    age = paleo['age_ka']
    dT = paleo['delta_T']
    unc = paleo['uncertainty']
    
    # MIS chegaralari (vertikal punktir)
    boundaries = set()
    for m in MIS_stages:
        boundaries.update([m['start'], m['end']])
    boundaries.discard(0)
    for b in sorted(boundaries):
        if b <= 115:
            ax1.axvline(x=b, color='#555', ls='--', lw=0.9, alpha=0.7, zorder=1)
    
    # Noaniqlik (kulrang shading)
    ax1.fill_between(age, dT - unc, dT + unc,
                     color='#AAAAAA', alpha=0.35, zorder=2)
    
    # Asosiy signal (qora chiziq)
    ax1.plot(age, dT, color='#1a1a1a', lw=1.8, zorder=3)
    
    # Nol chiziq
    ax1.axhline(y=0, color='#CC0000', ls='--', lw=0.6, alpha=0.5)
    
    # MIS nomlari (pastda)
    for m in MIS_stages:
        mid = (m['start'] + m['end']) / 2
        if mid <= 115:
            ax1.text(mid, -5.7, m['name'], ha='center', fontsize=8.5, color='#555')
    
    # LGM annotatsiya
    lgm_mask = (age >= 19) & (age <= 23)
    if lgm_mask.any():
        lgm_dt = dT[lgm_mask].min()
        ax1.annotate('LGM', xy=(21, lgm_dt), xytext=(30, lgm_dt - 1.5),
                     fontsize=9, fontweight='bold', ha='center',
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='#333'),
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#333', lw=0.8))
    
    # MIS 5e annotatsiya
    mis5e_mask = (age >= 80) & (age <= 90)
    if mis5e_mask.any():
        mis5e_dt = dT[mis5e_mask].max()
        ax1.annotate('MIS 5e', xy=(85, mis5e_dt), xytext=(75, mis5e_dt + 1.5),
                     fontsize=9, fontweight='bold', ha='center',
                     arrowprops=dict(arrowstyle='->', lw=1.2, color='#333'),
                     bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#333', lw=0.8))
    
    # Golosen optimumi
    hol_mask = (age >= 6) & (age <= 9)
    if hol_mask.any():
        ax1.text(7.5, -3.0, 'Golosen\noptimumi', ha='center', fontsize=8,
                 color='#8B4513', fontstyle='italic', alpha=0.8)
    
    # O'qlar
    ax1.set_xlabel('Yosh (ming yil oldin)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('ΔT (°C, zamonaviyga nisbatan)', fontsize=11, fontweight='bold')
    ax1.set_xlim(115, 0)
    ax1.set_ylim(-6, 6)
    ax1.set_xticks(np.arange(0, 120, 20))
    
    # Legend
    handles = [
        Line2D([0], [0], color='#1a1a1a', lw=1.8, label='ΔT (2500 yillik silliqlangan)'),
        plt.Rectangle((0, 0), 1, 1, fc='#AAAAAA', alpha=0.35, label='Noaniqlik diapazoni'),
    ]
    ax1.legend(handles=handles, loc='lower left', fontsize=8.5, framealpha=0.95, edgecolor='#CCC')
    ax1.text(0.02, 0.96, 'a)', transform=ax1.transAxes, fontsize=13,
             fontweight='bold', va='top', color='#CC0000')
    
    # Uzilish belgilari
    ax1.plot([0.99, 1.01], [0.28, 0.32], transform=ax1.transAxes,
             color='#666', lw=1.2, clip_on=False)
    ax1.plot([0.99, 1.01], [0.33, 0.37], transform=ax1.transAxes,
             color='#666', lw=1.2, clip_on=False)
    
    # ═══════════════════════════════════════════
    # PANEL B: Boysun MS harorat anomaliyasi
    # ═══════════════════════════════════════════
    
    years = df_modern.index.values
    anomaly = df_modern['anomaly'].values
    
    # Ustunlar (qizil/ko'k)
    colors = ['#E07060' if a >= 0 else '#6099C0' for a in anomaly]
    ax2.bar(years, anomaly, color=colors, alpha=0.7, width=0.8, zorder=2, edgecolor='none')
    
    # 11 yillik o'rtacha
    ma = df_modern['anomaly'].rolling(11, center=True, min_periods=6).mean()
    ax2.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0, zorder=4,
             label="11 yillik harakatlanuvchi o'rtacha")
    
    # Isish trendi
    if pd.notna(trend_decade) and len(recent_40) > 5:
        trend_line = intercept + slope * recent_40.index.values
        ax2.plot(recent_40.index, trend_line, color='#1a1a1a', lw=1.2, ls=':',
                 zorder=3, label=f"Trend ({trend_decade:+.2f}°C/o'n yillik)")
    
    # TON-2 marker (2008)
    if 2008 in df_modern.index:
        val = df_modern.loc[2008, 'anomaly']
        if pd.notna(val):
            ax2.plot(2008, val, 'D', color='#7B2D8B', ms=8, zorder=5,
                     markeredgecolor='white', markeredgewidth=1.0)
            ax2.text(2008, val + 0.2, 'TON-2\n(2008)', fontsize=7.5,
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
    
    # Uzilish
    ax2.plot([-0.01, 0.01], [0.28, 0.32], transform=ax2.transAxes,
             color='#666', lw=1.2, clip_on=False)
    ax2.plot([-0.01, 0.01], [0.33, 0.37], transform=ax2.transAxes,
             color='#666', lw=1.2, clip_on=False)
    
    # Tozalash
    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.tick_params(direction='out', length=4, width=0.8)
    
    plt.tight_layout()
    plt.savefig('fig3_paleoclimate.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig3_paleoclimate.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"  💾 fig3_paleoclimate.png / .pdf")
    print()
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    
    return paleo, df_modern


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    result = run()

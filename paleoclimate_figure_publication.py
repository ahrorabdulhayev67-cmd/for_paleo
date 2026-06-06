"""
TON-2 Paleoklimat rekonstruksiyasi — Yakuniy nashr grafigi
O'ng panel: Regional harorat anomaliyasi (Boysun + Denov + Mingchuqur)

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

BOYSUN_TEMP_FILE = 'Бойсун МС  8.xlsx'
BOYSUN_TEMP_SHEET = 'Ҳарорат'
DENOV_MINGCHUQUR_FILE = 'Denov_Mingchuqur.xlsx'

REF_START = 1961
REF_END = 1990

DIB_stages = [
    {'name': 'DIB 1', 'start': 0, 'end': 11.7},
    {'name': 'DIB 2', 'start': 11.7, 'end': 29},
    {'name': 'DIB 3', 'start': 29, 'end': 57},
    {'name': 'DIB 4', 'start': 57, 'end': 71},
    {'name': 'DIB 5', 'start': 71, 'end': 130},
]

# ══════════════════════════════════════════════════════════════
# 1. REGIONAL ANOMALIYA HISOBLASH (haqiqiy ma'lumotlar)
# ══════════════════════════════════════════════════════════════

def load_regional_data():
    """3 stansiya harorat anomaliyasini yuklash va o'rtachalash."""
    
    print("  📂 Stansiya ma'lumotlarini yuklash...")
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
        print(f"     ⚠️ Boysun harorat yuklanmadi: {e}")
    
    # Regional o'rtacha
    all_years = sorted(set().union(*[set(s.index) for s in stations.values()]))
    df_reg = pd.DataFrame(index=all_years)
    for name, series in stations.items():
        df_reg[name] = series
    df_reg['regional'] = df_reg.mean(axis=1)
    df_reg['n_stations'] = df_reg[list(stations.keys())].notna().sum(axis=1)
    
    print(f"     ✅ Regional: {len(df_reg)} yil ({min(all_years)}-{max(all_years)})")
    print()
    
    return df_reg


# ══════════════════════════════════════════════════════════════
# 2. PALEOKLIMATIK MA'LUMOTLAR (chap panel)
# ══════════════════════════════════════════════════════════════

def create_paleoclimate_data():
    """
    Sintetik paleoklimatik ΔT.
    ⚠️ HAQIQIY MA'LUMOTLARINGIZNI SHU YERGA YUKLANG!
    """
    age_ka = np.linspace(0, 130, 2600)
    np.random.seed(42)
    
    temp_signal = np.zeros_like(age_ka)
    for i, age in enumerate(age_ka):
        if age < 11.7:
            temp_signal[i] = 0.5 * np.sin(age / 3 * np.pi) + 0.3
        elif age < 29:
            temp_signal[i] = -2.5 - 1.5 * np.sin((age - 11.7) / 17.3 * np.pi)
        elif age < 57:
            temp_signal[i] = -0.5 + 1.5 * np.sin((age - 29) / 28 * 2 * np.pi)
        elif age < 71:
            temp_signal[i] = -1.0 - 0.8 * np.sin((age - 57) / 14 * np.pi)
        else:
            if 80 < age < 95:
                temp_signal[i] = 2.0 + 1.5 * np.sin((age - 80) / 15 * np.pi)
            elif 95 < age < 110:
                temp_signal[i] = -3.0 - 1.5 * np.sin((age - 95) / 15 * np.pi)
            elif age > 110:
                temp_signal[i] = 1.5 + 1.0 * np.sin((age - 110) / 20 * np.pi)
            else:
                temp_signal[i] = 1.0
    
    noise = np.random.randn(len(age_ka)) * 0.4
    temp_smoothed = gaussian_filter1d(temp_signal + noise, sigma=50)
    uncertainty = gaussian_filter1d(np.abs(np.random.randn(len(age_ka)) * 0.5) + 0.8, sigma=30)
    
    return age_ka, temp_smoothed, uncertainty


# ══════════════════════════════════════════════════════════════
# 3. YAKUNIY GRAFIK
# ══════════════════════════════════════════════════════════════

def run():
    """To'liq grafik yaratish."""
    
    print("=" * 70)
    print("  PALEOKLIMAT REKONSTRUKSIYASI — YAKUNIY GRAFIK")
    print("  Chap: TON-2 (130 ka) | O'ng: Regional anomaliya (3 stansiya)")
    print("=" * 70)
    print()
    
    # Ma'lumotlar
    age_ka, temp_smoothed, uncertainty = create_paleoclimate_data()
    df_reg = load_regional_data()
    
    # Trend (so'nggi 40 yil)
    recent = df_reg['regional'].dropna()
    recent_40 = recent[recent.index >= 1983]
    if len(recent_40) > 5:
        slope, intercept, r, p, se = stats.linregress(recent_40.index, recent_40.values)
        trend_decade = slope * 10
    else:
        trend_decade = np.nan
    
    # ── GRAFIK ──
    plt.rcParams.update({
        'font.family': 'Arial', 'font.size': 10,
        'axes.linewidth': 1.0, 'figure.dpi': 300
    })
    
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 5.5),
        gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.02}
    )
    
    # ═══════════ PANEL a: Paleoklimatik davr ═══════════
    
    # DIB chegaralari
    boundaries = set()
    for d in DIB_stages:
        boundaries.add(d['start'])
        boundaries.add(d['end'])
    boundaries.discard(0)
    for b in sorted(boundaries):
        if b <= 130:
            ax1.axvline(x=b, color='#555555', ls='--', lw=0.9, alpha=0.7, zorder=1)
    
    # Noaniqlik
    ax1.fill_between(age_ka, temp_smoothed - uncertainty, temp_smoothed + uncertainty,
                     color='#AAAAAA', alpha=0.35, zorder=2)
    
    # Signal
    ax1.plot(age_ka, temp_smoothed, color='#1a1a1a', lw=1.8, zorder=3)
    
    # Nol chiziq
    ax1.axhline(y=0, color='#CC0000', ls='--', lw=0.6, alpha=0.5)
    
    # Heinrich
    ax1.text(100, 5.3, 'H', ha='center', fontsize=9, fontweight='bold', color='#333')
    ax1.text(60, 5.3, 'H', ha='center', fontsize=9, fontweight='bold', color='#333')
    
    # Golosen optimumi
    ax1.text(7, -2.5, 'Golosen\noptimumi', ha='center', fontsize=8,
             color='#8B4513', fontstyle='italic', alpha=0.8)
    
    # DIB nomlari
    for d in DIB_stages:
        ax1.text((d['start'] + d['end']) / 2, -5.7, d['name'],
                 ha='center', fontsize=8.5, color='#555555')
    
    ax1.set_xlabel('Yosh (ming yil oldin)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('ΔT (°C, zamonaviyga nisbatan)', fontsize=11, fontweight='bold')
    ax1.set_xlim(130, 0)
    ax1.set_ylim(-6, 6)
    ax1.set_xticks(np.arange(0, 140, 20))
    
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
    
    # ═══════════ PANEL b: Regional anomaliya (HAQIQIY) ═══════════
    
    years = df_reg.index.values
    anomaly = df_reg['regional'].values
    
    # Ustunli diagramma
    colors = ['#E07060' if a >= 0 else '#6099C0' for a in anomaly]
    ax2.bar(years, anomaly, color=colors, alpha=0.7, width=0.8, zorder=2, edgecolor='none')
    
    # 11 yillik o'rtacha
    ma = df_reg['regional'].rolling(11, center=True, min_periods=6).mean()
    ax2.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0, zorder=4,
             label="11 yillik harakatlanuvchi o'rtacha")
    
    # Trend chizig'i
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
    
    # Nol chiziq
    ax2.axhline(y=0, color='#888', ls='-', lw=0.5, alpha=0.5)
    
    ax2.set_xlabel('Yil (milodiy)', fontsize=11, fontweight='bold')
    ax2.yaxis.set_label_position('right')
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Harorat anomaliyasi (°C)', fontsize=11, fontweight='bold')
    ax2.set_xlim(min(years) - 2, max(years) + 2)
    
    ax2.legend(loc='upper left', fontsize=8, framealpha=0.95, edgecolor='#CCC')
    ax2.text(0.03, 0.96, 'b)', transform=ax2.transAxes, fontsize=13,
             fontweight='bold', va='top', color='#CC0000')
    
    # Uzilish belgilari
    ax2.plot([-0.01, 0.01], [0.28, 0.32], transform=ax2.transAxes,
             color='#666', lw=1.2, clip_on=False)
    ax2.plot([-0.01, 0.01], [0.33, 0.37], transform=ax2.transAxes,
             color='#666', lw=1.2, clip_on=False)
    
    # Grid off
    for ax in [ax1, ax2]:
        ax.grid(False)
        ax.tick_params(direction='out', length=4, width=0.8)
    
    plt.tight_layout()
    plt.savefig('fig_paleoclimate_publication.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig_paleoclimate_publication.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"  💾 fig_paleoclimate_publication.png")
    print(f"  💾 fig_paleoclimate_publication.pdf")
    print()
    print("  📝 RASM OSTI YOZUVI:")
    print(f"  \"(a) Oxirgi 130 ming yil davomida harorat anomaliyasi (TON-2 speleothem).")
    print(f"   (b) Regional harorat anomaliyasi — 3 stansiya o'rtachasi")
    print(f"   (Boysun, Denov, Mingchuqur), {REF_START}-{REF_END} referens davriga nisbatan.")
    if pd.notna(trend_decade):
        print(f"   So'nggi 40 yillik trend: {trend_decade:+.3f}°C/o'n yillik.\"")
    print()
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    
    return df_reg


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = run()

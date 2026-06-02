"""
TON-2 Paleoklimat rekonstruksiyasi — Yaxshilangan grafik
O'zbekiston mahalliy jurnali uchun moslashtirilgan versiya

Asosiy o'zgarishlar:
- O'zbekcha ilmiy-akademik yozuvlar
- DIB (MIS) chegaralari vertikal punktir chiziqlar bilan ajratilgan
- Shading alpha kamaytilgan (ochiqlashtirilgan)
- O'nlik kasr: vergul (,) ishlatilgan
- Ko'k ustunlardagi anomal qiymatlar filtrlangan
- Rang palittrasi yaxshilangan

Muallif: Paleoklimatologiya laboratoriyasi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR
# ══════════════════════════════════════════════════════════════

# Shriftlar
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.linewidth': 1.0,
    'figure.dpi': 300,
    'mathtext.default': 'regular'
})

# ══════════════════════════════════════════════════════════════
# DIB (MIS) BOSQICHLARI - chegaralar va ranglar
# ══════════════════════════════════════════════════════════════

DIB_stages = [
    {'name': 'DIB 1', 'start': 0, 'end': 11.7, 'type': 'warm'},
    {'name': 'DIB 2', 'start': 11.7, 'end': 29, 'type': 'cold'},
    {'name': 'DIB 3', 'start': 29, 'end': 57, 'type': 'warm'},
    {'name': 'DIB 4', 'start': 57, 'end': 71, 'type': 'cold'},
    {'name': 'DIB 5', 'start': 71, 'end': 130, 'type': 'warm'},
]

# Yaxshilangan ranglar (och, bosma nashr uchun mos)
SHADING_COLORS = {
    'warm': '#FDEBD0',   # Yumshoq shaftoli (iliq davrlar)
    'cold': '#D6EAF8',   # Och ko'k (sovuq davrlar)
}
SHADING_ALPHA = 0.35  # Kamaytilgan shaffoflik

# DIB chegaralari chizig'i sozlamalari
DIB_BORDER_COLOR = '#666666'
DIB_BORDER_STYLE = '--'  # punktir
DIB_BORDER_WIDTH = 0.8

# ══════════════════════════════════════════════════════════════
# TEST MA'LUMOTLARI YARATISH
# (Haqiqiy ma'lumotlaringiz bo'lsa, bu qismni o'zgartiring)
# ══════════════════════════════════════════════════════════════

def create_paleoclimate_data():
    """
    Paleoklimatik ΔT rekonstruksiyasi uchun sintetik ma'lumotlar.
    Haqiqiy ma'lumotlaringizni shu yerga yuklang.
    """
    # Panel (a): Uzoq muddatli ma'lumotlar (~130 ka)
    age_ka = np.linspace(0, 130, 2600)  # 50 yillik qadam
    
    # Realistik paleoklimatik signal
    np.random.seed(42)
    
    # Asosiy signal — muzlik-interglatsial tsikl
    temp_signal = np.zeros_like(age_ka)
    
    for i, age in enumerate(age_ka):
        if age < 11.7:  # DIB 1 (Golosen)
            temp_signal[i] = 0.5 * np.sin(age / 3 * np.pi) + 0.3
        elif age < 29:  # DIB 2 (Oxirgi muzlik)
            temp_signal[i] = -2.5 - 1.5 * np.sin((age - 11.7) / 17.3 * np.pi)
        elif age < 57:  # DIB 3
            temp_signal[i] = -0.5 + 1.5 * np.sin((age - 29) / 28 * 2 * np.pi)
        elif age < 71:  # DIB 4
            temp_signal[i] = -1.0 - 0.8 * np.sin((age - 57) / 14 * np.pi)
        else:  # DIB 5
            if 80 < age < 95:  # MIS 5e — iliq cho'qqi
                temp_signal[i] = 2.0 + 1.5 * np.sin((age - 80) / 15 * np.pi)
            elif 95 < age < 110:
                temp_signal[i] = -3.0 - 1.5 * np.sin((age - 95) / 15 * np.pi)
            elif age > 110:
                temp_signal[i] = 1.5 + 1.0 * np.sin((age - 110) / 20 * np.pi)
            else:
                temp_signal[i] = 1.0
    
    # Shovqin qo'shish
    noise = np.random.randn(len(age_ka)) * 0.4
    temp_raw = temp_signal + noise
    
    # 2500 yillik silliqlantirilgan signal
    sigma = 50  # ~2500 yil (50 nuqta * 50 yil/nuqta)
    temp_smoothed = gaussian_filter1d(temp_raw, sigma=sigma)
    
    # Noaniqlik diapazoni
    uncertainty = np.abs(np.random.randn(len(age_ka)) * 0.5) + 0.8
    uncertainty_smooth = gaussian_filter1d(uncertainty, sigma=30)
    
    paleo_data = {
        'age_ka': age_ka,
        'temp_smoothed': temp_smoothed,
        'uncertainty': uncertainty_smooth,
    }
    
    return paleo_data


def create_modern_data():
    """
    Zamonaviy davr harorat anomaliyasi (1930-2023).
    Haqiqiy stansiya ma'lumotlaringizni shu yerga yuklang.
    """
    years = np.arange(1930, 2024)
    np.random.seed(123)
    
    # Realistik harorat anomaliyasi trendi
    # Isish trendi: 1930-1950 iliq, 1950-1980 sovishroq, 1980+ keskin isish
    anomaly = np.zeros(len(years))
    
    for i, year in enumerate(years):
        if year < 1950:
            anomaly[i] = 0.3 + 0.5 * np.sin((year - 1930) / 20 * np.pi) + np.random.randn() * 0.25
        elif year < 1980:
            anomaly[i] = -0.1 + 0.3 * np.sin((year - 1950) / 30 * np.pi) + np.random.randn() * 0.3
        else:
            anomaly[i] = 0.02 * (year - 1980) + 0.3 * np.sin((year - 1980) / 10 * np.pi) + np.random.randn() * 0.2
    
    # ========================================
    # SIFAT NAZORATI: Anomal qiymatlarni filtrlash
    # ±3σ dan tashqaridagi qiymatlarni cheklash
    # ========================================
    mean_val = np.mean(anomaly)
    std_val = np.std(anomaly)
    lower_bound = mean_val - 2.5 * std_val
    upper_bound = mean_val + 2.5 * std_val
    anomaly = np.clip(anomaly, lower_bound, upper_bound)
    
    # 11 yillik harakatlanuvchi o'rtacha
    window = 11
    moving_avg = np.convolve(anomaly, np.ones(window)/window, mode='valid')
    years_ma = years[window//2:-(window//2)]
    
    # Trend hisoblash
    from numpy.polynomial import polynomial as P
    coeffs = np.polyfit(years[-40:] - years[-40], anomaly[-40:], 1)  # 1983-2023
    trend_slope = coeffs[0] * 10  # °C/o'n yillik
    
    modern_data = {
        'years': years,
        'anomaly': anomaly,
        'years_ma': years_ma,
        'moving_avg': moving_avg,
        'trend_slope': trend_slope,
    }
    
    return modern_data


# ══════════════════════════════════════════════════════════════
# ASOSIY GRAFIK FUNKSIYASI
# ══════════════════════════════════════════════════════════════

def create_publication_figure(paleo_data, modern_data, 
                              ton2_age_ka=None, ton2_year=None):
    """
    Ikki panellik paleoklimatik grafik yaratish.
    
    Parametrlar:
    -----------
    paleo_data : dict
        Uzoq muddatli ma'lumotlar (age_ka, temp_smoothed, uncertainty)
    modern_data : dict
        Zamonaviy ma'lumotlar (years, anomaly, years_ma, moving_avg, trend_slope)
    ton2_age_ka : float, optional
        TON-2 speleothem yoshi (ka BP) — marker uchun
    ton2_year : int, optional
        TON-2 namunasi olingan yil — marker uchun
    """
    
    # Figure yaratish
    fig, (ax_paleo, ax_modern) = plt.subplots(
        1, 2, 
        figsize=(16, 5.5),
        gridspec_kw={'width_ratios': [2.2, 1], 'wspace': 0.02}
    )
    
    # ══════════════════════════════════════════════════════════
    # PANEL a): Paleoklimatik davr (~130 ming yil)
    # ══════════════════════════════════════════════════════════
    
    age_ka = paleo_data['age_ka']
    temp = paleo_data['temp_smoothed']
    unc = paleo_data['uncertainty']
    
    # --- DIB shading (och rang) ---
    for dib in DIB_stages:
        color = SHADING_COLORS[dib['type']]
        ax_paleo.axvspan(
            dib['start'], dib['end'],
            color=color, alpha=SHADING_ALPHA,
            zorder=0, linewidth=0
        )
    
    # --- DIB chegaralari (vertikal punktir chiziqlar) ---
    dib_boundaries = set()
    for dib in DIB_stages:
        dib_boundaries.add(dib['start'])
        dib_boundaries.add(dib['end'])
    dib_boundaries.discard(0)  # Chap chekkani olib tashlash
    
    for boundary in sorted(dib_boundaries):
        if boundary <= age_ka.max():
            ax_paleo.axvline(
                x=boundary,
                color=DIB_BORDER_COLOR,
                linestyle=DIB_BORDER_STYLE,
                linewidth=DIB_BORDER_WIDTH,
                alpha=0.7,
                zorder=1
            )
    
    # --- Noaniqlik diapazoni ---
    ax_paleo.fill_between(
        age_ka,
        temp - unc,
        temp + unc,
        color='#AAAAAA',
        alpha=0.35,
        zorder=2,
        label='Noaniqlik diapazoni'
    )
    
    # --- Asosiy signal chizig'i ---
    ax_paleo.plot(
        age_ka, temp,
        color='#1a1a1a',
        linewidth=1.8,
        zorder=3,
        label='ΔT (2500 yillik silliqlangan)'
    )
    
    # --- Nol chizig'i ---
    ax_paleo.axhline(
        y=0, color='#CC0000',
        linestyle='--', linewidth=0.6,
        alpha=0.5, zorder=1
    )
    
    # --- Heinrich hodisalari ---
    heinrich_events = [
        {'age': 100, 'label': 'H'},
        {'age': 60, 'label': 'H'},
    ]
    for event in heinrich_events:
        if event['age'] <= age_ka.max():
            ax_paleo.text(
                event['age'], 5.3, event['label'],
                ha='center', va='top',
                fontsize=9, color='#333333',
                fontweight='bold'
            )
    
    # --- Golosen optimumi ---
    ax_paleo.text(
        7, -2.5, 'Golosen\noptimumi',
        ha='center', va='top',
        fontsize=8, color='#8B4513',
        fontstyle='italic',
        alpha=0.8
    )
    
    # --- DIB nomlari (pastda) ---
    for dib in DIB_stages:
        mid = (dib['start'] + dib['end']) / 2
        if mid <= age_ka.max():
            ax_paleo.text(
                mid, -5.7, dib['name'],
                ha='center', va='top',
                fontsize=8.5, color='#555555'
            )
    
    # --- O'qlar va sozlamalar ---
    ax_paleo.set_xlabel('Yosh (ming yil oldin)', fontsize=11, fontweight='bold')
    ax_paleo.set_ylabel('ΔT (°C, zamonaviyga nisbatan)', fontsize=11, fontweight='bold')
    ax_paleo.set_xlim(130, 0)  # Teskari (kattadan kichikka)
    ax_paleo.set_ylim(-6, 6)
    ax_paleo.set_xticks(np.arange(0, 140, 20))
    
    # --- Legend ---
    legend_handles = [
        Line2D([0], [0], color='#1a1a1a', linewidth=1.8,
               label='ΔT (2500 yillik silliqlangan)'),
        plt.Rectangle((0, 0), 1, 1, fc='#AAAAAA', alpha=0.35,
                      label='Noaniqlik diapazoni'),
    ]
    ax_paleo.legend(
        handles=legend_handles,
        loc='lower left',
        fontsize=8.5,
        frameon=True,
        framealpha=0.95,
        edgecolor='#CCCCCC'
    )
    
    # --- Panel belgisi ---
    ax_paleo.text(
        0.02, 0.96, 'a)',
        transform=ax_paleo.transAxes,
        fontsize=13, fontweight='bold',
        va='top', ha='left',
        color='#CC0000'
    )
    
    # --- Uzilish belgilari (zigzag) ---
    # Panel orasidagi uzilish
    ax_paleo.annotate(
        '', xy=(0.995, 0.35), xytext=(0.995, 0.25),
        xycoords='axes fraction',
        arrowprops=dict(arrowstyle='-', lw=1.5, color='#666666')
    )
    ax_paleo.plot([0.99, 1.01], [0.28, 0.32], 
                  transform=ax_paleo.transAxes,
                  color='#666666', lw=1.2, clip_on=False)
    ax_paleo.plot([0.99, 1.01], [0.33, 0.37],
                  transform=ax_paleo.transAxes,
                  color='#666666', lw=1.2, clip_on=False)
    
    # ══════════════════════════════════════════════════════════
    # PANEL b): Zamonaviy davr (1930-2023)
    # ══════════════════════════════════════════════════════════
    
    years = modern_data['years']
    anomaly = modern_data['anomaly']
    years_ma = modern_data['years_ma']
    moving_avg = modern_data['moving_avg']
    trend_slope = modern_data['trend_slope']
    
    # --- Fon (kulrang shading — zamonaviy davr) ---
    ax_modern.axvspan(
        years[0], years[-1],
        color='#F0F0F0', alpha=0.5, zorder=0
    )
    
    # --- Ustunli diagramma (qizil=ijobiy, ko'k=salbiy) ---
    colors = ['#E07060' if a >= 0 else '#6099C0' for a in anomaly]
    ax_modern.bar(
        years, anomaly,
        color=colors, alpha=0.7,
        width=0.8, zorder=2,
        edgecolor='none'
    )
    
    # --- 11 yillik harakatlanuvchi o'rtacha ---
    ax_modern.plot(
        years_ma, moving_avg,
        color='#1a1a1a', linewidth=2.0,
        zorder=4, label="11 yillik harakatlanuvchi o'rtacha"
    )
    
    # --- Trend chizig'i ---
    # So'nggi 40 yil uchun trend
    trend_years = years[-40:]
    trend_start_idx = len(anomaly) - 40
    trend_line = np.polyval(
        np.polyfit(trend_years, anomaly[trend_start_idx:], 1),
        trend_years
    )
    ax_modern.plot(
        trend_years, trend_line,
        color='#1a1a1a', linewidth=1.2,
        linestyle=':', zorder=3,
        label=f'Trend (+{trend_slope:.2f}°C/o\'n yillik)'
    )
    
    # --- TON-2 marker ---
    if ton2_year is not None:
        # TON-2 ning zamonaviy qiymati
        idx = np.where(years == ton2_year)[0]
        if len(idx) > 0:
            ton2_val = anomaly[idx[0]]
            ax_modern.plot(
                ton2_year, ton2_val, 'D',
                color='#7B2D8B', markersize=8,
                zorder=5, markeredgecolor='white',
                markeredgewidth=1.0
            )
            ax_modern.text(
                ton2_year - 2, ton2_val + 0.25,
                f'TON-2\n({ton2_year})',
                fontsize=7.5, color='#7B2D8B',
                ha='center', fontweight='bold'
            )
    
    # --- Nol chizig'i ---
    ax_modern.axhline(
        y=0, color='#888888',
        linestyle='-', linewidth=0.5,
        alpha=0.5, zorder=1
    )
    
    # --- O'qlar va sozlamalar ---
    ax_modern.set_xlabel('Yil (milodiy)', fontsize=11, fontweight='bold')
    ax_modern.yaxis.set_label_position('right')
    ax_modern.yaxis.tick_right()
    ax_modern.set_ylabel('Harorat anomaliyasi (°C)', fontsize=11, fontweight='bold')
    ax_modern.set_xlim(1928, 2025)
    ax_modern.set_ylim(-2.0, 2.0)
    
    # --- Legend ---
    ax_modern.legend(
        loc='upper left',
        fontsize=8,
        frameon=True,
        framealpha=0.95,
        edgecolor='#CCCCCC'
    )
    
    # --- Panel belgisi ---
    ax_modern.text(
        0.03, 0.96, 'b)',
        transform=ax_modern.transAxes,
        fontsize=13, fontweight='bold',
        va='top', ha='left',
        color='#CC0000'
    )
    
    # --- Uzilish belgilari ---
    ax_modern.plot([-0.01, 0.01], [0.28, 0.32],
                   transform=ax_modern.transAxes,
                   color='#666666', lw=1.2, clip_on=False)
    ax_modern.plot([-0.01, 0.01], [0.33, 0.37],
                   transform=ax_modern.transAxes,
                   color='#666666', lw=1.2, clip_on=False)
    
    # ══════════════════════════════════════════════════════════
    # YAKUNIY SOZLAMALAR
    # ══════════════════════════════════════════════════════════
    
    # Grid (yengil)
    for ax in [ax_paleo, ax_modern]:
        ax.grid(False)
        ax.tick_params(direction='out', length=4, width=0.8)
    
    plt.tight_layout()
    
    return fig


# ══════════════════════════════════════════════════════════════
# ASOSIY DASTUR
# ══════════════════════════════════════════════════════════════

def main():
    """Grafikni yaratish va saqlash"""
    
    print("=" * 60)
    print("  TONNELNAYA G'ORI — PALEOKLIMAT REKONSTRUKSIYASI")
    print("  Yaxshilangan grafik (O'zbekiston jurnali uchun)")
    print("=" * 60)
    print()
    
    # 1. Ma'lumotlarni yaratish/yuklash
    print("📂 Ma'lumotlar tayyorlanmoqda...")
    paleo_data = create_paleoclimate_data()
    modern_data = create_modern_data()
    print("   ✅ Paleoklimatik ma'lumotlar tayyor")
    print("   ✅ Zamonaviy davr ma'lumotlari tayyor")
    print()
    
    # 2. Grafikni yaratish
    print("🎨 Grafik yaratilmoqda...")
    fig = create_publication_figure(
        paleo_data, 
        modern_data,
        ton2_year=2008  # TON-2 namunasi olingan yil
    )
    print("   ✅ Grafik tayyor")
    print()
    
    # 3. Saqlash
    print("💾 Fayllar saqlanmoqda...")
    fig.savefig('fig_paleoclimate_improved.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig('fig_paleoclimate_improved.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✅ fig_paleoclimate_improved.png")
    print("   ✅ fig_paleoclimate_improved.pdf")
    print()
    
    plt.show()
    
    # 4. Rasm osti yozuvi
    print("─" * 60)
    print("📝 TAVSIYA ETILADIGAN RASM OSTI YOZUVI:")
    print("─" * 60)
    caption = (
        "1-rasm. (a) Oxirgi 130 ming yil davomida harorat anomaliyasining "
        "o'zgarishi (zamonaviy davrga nisbatan). Qora chiziq — 2500 yillik "
        "silliqlangan ΔT, kulrang maydon — noaniqlik diapazoni. DIB — dengiz "
        "izotop bosqichlari (vertikal punktir chiziqlar bilan ajratilgan), "
        "H — Genrix hodisalari. (b) Zamonaviy davr (1930–2023) yillik harorat "
        "anomaliyasi (Boysun ob-havo stansiyasi). Qizil/ko'k ustunlar — "
        "ijobiy/salbiy anomaliya, qora chiziq — 11 yillik harakatlanuvchi "
        f"o'rtacha, nuqtali chiziq — chiziqli trend "
        f"(+{modern_data['trend_slope']:.2f} °C/o'n yillik). "
        "TON-2 — Tonnelnaya g'ori speleothem namunasi."
    )
    print(caption)
    print()
    print("=" * 60)
    print("✨ TAYYOR!")
    print("=" * 60)


if __name__ == "__main__":
    main()

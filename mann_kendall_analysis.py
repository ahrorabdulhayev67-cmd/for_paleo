"""
Mann-Kendall Trend Testi — TON-2 Paleoklimat Ma'lumotlari
═══════════════════════════════════════════════════════════

Mann-Kendall testi — vaqt qatorlarida monoton trendning mavjudligini
tekshirish uchun noparametrik statistik test.

Afzalliklari:
- Ma'lumotlar normal taqsimotga ega bo'lishi shart emas
- Outlier'larga chidamli
- Iqlim tadqiqotlarida standart usul (IPCC tomonidan qo'llaniladi)

Natija:
- tau: Kendall tau koeffitsienti (-1 dan +1 gacha)
- p-value: Statistik ahamiyatlilik (p < 0.05 = trend ahamiyatli)
- Sen's slope: Trend tezligi (°C/yil yoki °C/o'n yillik)

Muallif: Paleoklimatologiya laboratoriyasi
Sana: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# MANN-KENDALL TEST FUNKSIYASI
# ══════════════════════════════════════════════════════════════

def mann_kendall_test(data, alpha=0.05):
    """
    Mann-Kendall monoton trend testi.
    
    Parametrlar:
    -----------
    data : array-like
        Vaqt qatori ma'lumotlari
    alpha : float
        Ahamiyatlilik darajasi (default: 0.05)
    
    Qaytaradi:
    ---------
    dict: {
        'tau': Kendall tau koeffitsienti,
        'p_value': p-qiymat,
        'trend': 'ortib boruvchi' / 'kamayib boruvchi' / 'trend yo\'q',
        'significant': True/False,
        'S': Mann-Kendall S statistikasi,
        'var_S': S ning dispersiyasi,
        'z': Z-statistika
    }
    """
    n = len(data)
    
    # S statistikasini hisoblash
    S = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0:
                S += 1
            elif diff < 0:
                S -= 1
    
    # Bog'langan qiymatlar (ties)
    unique, counts = np.unique(data, return_counts=True)
    ties = counts[counts > 1]
    
    # Dispersiyani hisoblash
    var_S = (n * (n - 1) * (2 * n + 5)) / 18
    
    if len(ties) > 0:
        for t in ties:
            var_S -= (t * (t - 1) * (2 * t + 5)) / 18
    
    # Z-statistika
    if S > 0:
        z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
    else:
        z = 0
    
    # p-qiymat (ikki tomonlama)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    # Kendall tau
    tau = S / (n * (n - 1) / 2)
    
    # Trend yo'nalishi
    if p_value <= alpha:
        if S > 0:
            trend = 'ortib boruvchi'
        else:
            trend = 'kamayib boruvchi'
        significant = True
    else:
        trend = "trend yo'q"
        significant = False
    
    return {
        'tau': tau,
        'p_value': p_value,
        'trend': trend,
        'significant': significant,
        'S': S,
        'var_S': var_S,
        'z': z
    }


def sens_slope(data, time=None):
    """
    Sen's slope estimator — trend tezligini hisoblash.
    
    Parametrlar:
    -----------
    data : array-like
        Vaqt qatori ma'lumotlari
    time : array-like, optional
        Vaqt o'qi (default: 0, 1, 2, ...)
    
    Qaytaradi:
    ---------
    dict: {
        'slope': Mediana trend tezligi,
        'intercept': Y-o'qi kesishish nuqtasi,
        'slope_per_decade': O'n yillik trend tezligi
    }
    """
    n = len(data)
    
    if time is None:
        time = np.arange(n)
    
    # Barcha juft nuqtalar orasidagi qiyalikni hisoblash
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            if time[j] != time[i]:
                slope = (data[j] - data[i]) / (time[j] - time[i])
                slopes.append(slope)
    
    # Mediana qiyalik
    median_slope = np.median(slopes)
    
    # Intercept
    intercept = np.median(data) - median_slope * np.median(time)
    
    return {
        'slope': median_slope,
        'intercept': intercept,
        'slope_per_decade': median_slope * 10
    }


# ══════════════════════════════════════════════════════════════
# MA'LUMOTLARNI TAYYORLASH
# ══════════════════════════════════════════════════════════════

def create_test_modern_data():
    """
    Zamonaviy davr harorat anomaliyasi (test ma'lumotlar).
    HAQIQIY MA'LUMOTLARINGIZNI SHU YERGA YUKLANG.
    """
    years = np.arange(1930, 2024)
    np.random.seed(123)
    
    anomaly = np.zeros(len(years))
    
    for i, year in enumerate(years):
        if year < 1950:
            anomaly[i] = 0.3 + 0.5 * np.sin((year - 1930) / 20 * np.pi) + np.random.randn() * 0.25
        elif year < 1980:
            anomaly[i] = -0.1 + 0.3 * np.sin((year - 1950) / 30 * np.pi) + np.random.randn() * 0.3
        else:
            anomaly[i] = 0.02 * (year - 1980) + 0.3 * np.sin((year - 1980) / 10 * np.pi) + np.random.randn() * 0.2
    
    # Sifat nazorati
    mean_val = np.mean(anomaly)
    std_val = np.std(anomaly)
    anomaly = np.clip(anomaly, mean_val - 2.5 * std_val, mean_val + 2.5 * std_val)
    
    return years, anomaly


# ══════════════════════════════════════════════════════════════
# ASOSIY TAHLIL VA GRAFIK
# ══════════════════════════════════════════════════════════════

def run_full_analysis():
    """To'liq Mann-Kendall tahlili va grafik."""
    
    print("=" * 70)
    print("  MANN-KENDALL TREND TESTI — TON-2 HARORAT ANOMALIYASI")
    print("=" * 70)
    print()
    
    # Ma'lumotlarni yuklash
    years, anomaly = create_test_modern_data()
    
    # ══════════════════════════════════════════════════════════
    # 1. TO'LIQ DAVR (1930-2023)
    # ══════════════════════════════════════════════════════════
    print("━" * 70)
    print("  1. TO'LIQ DAVR (1930–2023)")
    print("━" * 70)
    
    mk_full = mann_kendall_test(anomaly)
    sen_full = sens_slope(anomaly, years)
    
    print(f"  Kendall tau        : {mk_full['tau']:.4f}")
    print(f"  Z-statistika       : {mk_full['z']:.4f}")
    print(f"  p-qiymat           : {mk_full['p_value']:.6f}")
    print(f"  Trend              : {mk_full['trend']}")
    print(f"  Statistik ahamiyat : {'HA (p < 0,05)' if mk_full['significant'] else 'YO`Q'}")
    print(f"  Sen's slope        : {sen_full['slope']:.4f} °C/yil")
    print(f"  Sen's slope        : {sen_full['slope_per_decade']:.3f} °C/o'n yillik")
    print()
    
    # ══════════════════════════════════════════════════════════
    # 2. DAVRLAR BO'YICHA TAHLIL
    # ══════════════════════════════════════════════════════════
    print("━" * 70)
    print("  2. DAVRLAR BO'YICHA TAHLIL")
    print("━" * 70)
    
    periods = [
        ('1930–1960', 1930, 1961),
        ('1961–1990', 1961, 1991),
        ('1991–2023', 1991, 2024),
        ('1980–2023', 1980, 2024),
    ]
    
    results = []
    
    print(f"\n  {'Davr':<12} {'tau':<8} {'p-qiymat':<12} {'Trend':<20} {'Sen slope (°C/dec)':<20}")
    print(f"  {'─'*12} {'─'*8} {'─'*12} {'─'*20} {'─'*20}")
    
    for name, start, end in periods:
        mask = (years >= start) & (years < end)
        period_data = anomaly[mask]
        period_years = years[mask]
        
        mk = mann_kendall_test(period_data)
        sen = sens_slope(period_data, period_years)
        
        significance = '***' if mk['p_value'] < 0.001 else '**' if mk['p_value'] < 0.01 else '*' if mk['p_value'] < 0.05 else 'ns'
        
        print(f"  {name:<12} {mk['tau']:<8.4f} {mk['p_value']:<12.6f} {mk['trend']:<20} {sen['slope_per_decade']:<+20.4f} {significance}")
        
        results.append({
            'period': name,
            'mk': mk,
            'sen': sen,
            'years': period_years,
            'data': period_data
        })
    
    print()
    print("  Belgilar: *** p<0,001; ** p<0,01; * p<0,05; ns — ahamiyatsiz")
    print()
    
    # ══════════════════════════════════════════════════════════
    # 3. GRAFIK
    # ══════════════════════════════════════════════════════════
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 1.0,
        'figure.dpi': 300,
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), 
                             gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.25})
    
    # ── Panel (a): Harorat anomaliyasi + trend chiziqlari ──
    ax1 = axes[0]
    
    # Ustunli diagramma
    colors = ['#E07060' if a >= 0 else '#6099C0' for a in anomaly]
    ax1.bar(years, anomaly, color=colors, alpha=0.7, width=0.8, zorder=2)
    
    # Sen's slope chizig'i (to'liq davr)
    sen_line_full = sen_full['intercept'] + sen_full['slope'] * years
    ax1.plot(years, sen_line_full, 
             color='#1a1a1a', linewidth=1.5, linestyle='--', zorder=3,
             label=f"Sen's slope: {sen_full['slope_per_decade']:+.3f} °C/o'n yillik (1930–2023)")
    
    # So'nggi davr uchun Sen's slope
    mask_recent = (years >= 1980)
    mk_recent = mann_kendall_test(anomaly[mask_recent])
    sen_recent = sens_slope(anomaly[mask_recent], years[mask_recent])
    sen_line_recent = sen_recent['intercept'] + sen_recent['slope'] * years[mask_recent]
    ax1.plot(years[mask_recent], sen_line_recent,
             color='#CC0000', linewidth=2.0, linestyle='-', zorder=4,
             label=f"Sen's slope: {sen_recent['slope_per_decade']:+.3f} °C/o'n yillik (1980–2023)")
    
    # 11 yillik harakatlanuvchi o'rtacha
    window = 11
    moving_avg = np.convolve(anomaly, np.ones(window)/window, mode='valid')
    years_ma = years[window//2:-(window//2)]
    ax1.plot(years_ma, moving_avg, color='#333333', linewidth=2.0, zorder=5,
             label="11 yillik harakatlanuvchi o'rtacha")
    
    # Nol chizig'i
    ax1.axhline(y=0, color='#888888', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Annotatsiya — test natijalari
    text_box = (
        f"Mann-Kendall (1930–2023):\n"
        f"  τ = {mk_full['tau']:.3f}, p = {mk_full['p_value']:.4f}\n"
        f"  Trend: {mk_full['trend']}\n\n"
        f"Mann-Kendall (1980–2023):\n"
        f"  τ = {mk_recent['tau']:.3f}, p = {mk_recent['p_value']:.4f}\n"
        f"  Trend: {mk_recent['trend']}"
    )
    ax1.text(0.02, 0.97, text_box,
             transform=ax1.transAxes,
             fontsize=8.5, va='top', ha='left',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#CCCCCC', alpha=0.95))
    
    ax1.set_xlabel('Yil (milodiy)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Harorat anomaliyasi (°C)', fontsize=11, fontweight='bold')
    ax1.set_xlim(1928, 2025)
    ax1.legend(loc='upper right', fontsize=8.5, frameon=True,
               framealpha=0.95, edgecolor='#CCCCCC')
    ax1.set_title("Mann-Kendall trend tahlili — harorat anomaliyasi",
                  fontsize=12, fontweight='bold', pad=10)
    ax1.text(0.98, 0.97, 'a)', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right', color='#CC0000')
    
    # ── Panel (b): Davrlar bo'yicha Sen's slope ──
    ax2 = axes[1]
    
    period_names = [r['period'] for r in results]
    slopes = [r['sen']['slope_per_decade'] for r in results]
    p_values = [r['mk']['p_value'] for r in results]
    
    # Ranglar: ijobiy=qizil, salbiy=ko'k
    bar_colors = ['#E07060' if s > 0 else '#6099C0' for s in slopes]
    
    # Ahamiyatlilik bo'yicha alpha
    bar_alphas = [0.9 if p < 0.05 else 0.4 for p in p_values]
    
    bars = ax2.bar(period_names, slopes, color=bar_colors, 
                   edgecolor='#333333', linewidth=0.8, width=0.6, zorder=2)
    
    # Alpha qo'llash
    for bar, alpha_val in zip(bars, bar_alphas):
        bar.set_alpha(alpha_val)
    
    # Ahamiyatlilik belgilari
    for i, (name, slope, p) in enumerate(zip(period_names, slopes, p_values)):
        significance = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        offset = 0.01 if slope >= 0 else -0.01
        ax2.text(i, slope + offset, significance,
                 ha='center', va='bottom' if slope >= 0 else 'top',
                 fontsize=10, fontweight='bold',
                 color='#CC0000' if p < 0.05 else '#888888')
    
    ax2.axhline(y=0, color='#888888', linestyle='-', linewidth=0.8)
    ax2.set_ylabel("Sen's slope (°C/o'n yillik)", fontsize=11, fontweight='bold')
    ax2.set_xlabel('Davr', fontsize=11, fontweight='bold')
    ax2.set_title("Davrlar bo'yicha trend tezligi (Sen's slope)",
                  fontsize=12, fontweight='bold', pad=10)
    ax2.text(0.98, 0.97, 'b)', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', ha='right', color='#CC0000')
    
    # Grid
    for ax in axes:
        ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        ax.tick_params(direction='out', length=4, width=0.8)
    
    plt.tight_layout()
    
    # Saqlash
    plt.savefig('fig_mann_kendall_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('fig_mann_kendall_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print("━" * 70)
    print("  XULOSA")
    print("━" * 70)
    print()
    print(f"  • To'liq davr (1930–2023): isish trendi {sen_full['slope_per_decade']:+.3f} °C/o'n yillik")
    if mk_full['significant']:
        print(f"    → Statistik ahamiyatli (p = {mk_full['p_value']:.4f})")
    else:
        print(f"    → Statistik ahamiyatsiz (p = {mk_full['p_value']:.4f})")
    print()
    print(f"  • So'nggi davr (1980–2023): isish trendi {sen_recent['slope_per_decade']:+.3f} °C/o'n yillik")
    if mk_recent['significant']:
        print(f"    → Statistik ahamiyatli (p = {mk_recent['p_value']:.4f})")
    else:
        print(f"    → Statistik ahamiyatsiz (p = {mk_recent['p_value']:.4f})")
    print()
    print("  💾 Fayllar saqlandi:")
    print("     - fig_mann_kendall_analysis.png")
    print("     - fig_mann_kendall_analysis.pdf")
    print()
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)


# ══════════════════════════════════════════════════════════════
# ISHGA TUSHIRISH
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_full_analysis()

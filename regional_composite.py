"""
Regional harorat anomaliyasi — 3 stansiya o'rtachasi
═══════════════════════════════════════════════════════

Stansiyalar: Boysun + Denov + Mingchuqur
Usul: Anomaliyalarni o'rtachalash (har stansiya o'z klimatologiyasidan)

Natija: Paleoklimat grafikining o'ng paneli uchun regional T anomaliyasi

pip install numpy pandas matplotlib scipy openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR
# ══════════════════════════════════════════════════════════════

BOYSUN_PRECIP_FILE = 'boysun_precipitation_clean.csv'       # yoki filled versiya
BOYSUN_TEMP_FILE = 'Бойсун МС  8.xlsx'
BOYSUN_TEMP_SHEET = 'Ҳарорат'   # o'zgartiring agar boshqacha bo'lsa
DENOV_MINGCHUQUR_FILE = 'Denov_Mingchuqur.xlsx'

# Klimatologik referens davri (anomaliya hisoblash uchun)
REF_START = 1961
REF_END = 1990

# ══════════════════════════════════════════════════════════════
# ASOSIY KOD
# ══════════════════════════════════════════════════════════════

def run():
    """Regional harorat anomaliyasi hisoblash va grafik."""
    
    print("=" * 70)
    print("  REGIONAL HARORAT ANOMALIYASI")
    print("  Boysun + Denov + Mingchuqur")
    print("=" * 70)
    print()
    
    # ──────────────────────────────────────────
    # 1. MA'LUMOTLARNI YUKLASH
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  1. MA'LUMOTLARNI YUKLASH")
    print("━" * 70)
    print()
    
    # Denov harorat
    df_denov_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Денов')
    cols = list(df_denov_raw.columns)
    df_denov = pd.DataFrame()
    df_denov['year'] = pd.to_numeric(df_denov_raw[cols[0]], errors='coerce').astype('Int64')
    df_denov['temp'] = pd.to_numeric(df_denov_raw[cols[1]], errors='coerce')
    df_denov = df_denov.dropna(subset=['year']).set_index('year')
    print(f"  ✅ Denov harorat: {len(df_denov)} yil ({df_denov.index.min()}-{df_denov.index.max()})")
    
    # Mingchuqur harorat
    df_ming_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Мингчуқур')
    cols = list(df_ming_raw.columns)
    df_ming = pd.DataFrame()
    df_ming['year'] = pd.to_numeric(df_ming_raw[cols[0]], errors='coerce').astype('Int64')
    df_ming['temp'] = pd.to_numeric(df_ming_raw[cols[1]], errors='coerce')
    df_ming = df_ming.dropna(subset=['year']).set_index('year')
    print(f"  ✅ Mingchuqur harorat: {len(df_ming)} yil ({df_ming.index.min()}-{df_ming.index.max()})")
    
    # Boysun harorat
    # Avval Excel dan yillik haroratni olishga harakat qilish
    try:
        df_boysun_raw = pd.read_excel(BOYSUN_TEMP_FILE, sheet_name=BOYSUN_TEMP_SHEET)
        cols = list(df_boysun_raw.columns)
        df_boysun = pd.DataFrame()
        df_boysun['year'] = pd.to_numeric(df_boysun_raw[cols[0]], errors='coerce').astype('Int64')
        # Agar oylik bo'lsa — o'rtacha olish, agar yillik bo'lsa — to'g'ridan
        if len(cols) > 13:
            # Oylik: o'rtacha hisoblash
            month_data = df_boysun_raw[cols[1:13]].apply(pd.to_numeric, errors='coerce')
            df_boysun['temp'] = month_data.mean(axis=1)
        elif len(cols) == 3:
            # yil | harorat | yogin formatida
            df_boysun['temp'] = pd.to_numeric(df_boysun_raw[cols[1]], errors='coerce')
        else:
            # Birinchi raqamli ustunni olish
            df_boysun['temp'] = pd.to_numeric(df_boysun_raw[cols[1]], errors='coerce')
        
        df_boysun = df_boysun.dropna(subset=['year']).set_index('year')
        df_boysun['year_int'] = df_boysun.index.astype(int)
        df_boysun.index = df_boysun['year_int']
        df_boysun = df_boysun.drop(columns=['year_int'])
        print(f"  ✅ Boysun harorat: {len(df_boysun)} yil ({df_boysun.index.min()}-{df_boysun.index.max()})")
    except Exception as e:
        print(f"  ⚠️  Boysun harorat yuklanmadi: {e}")
        print("     Faqat Denov + Mingchuqur ishlatiladi")
        df_boysun = pd.DataFrame(columns=['temp'])
    
    print()
    
    # ──────────────────────────────────────────
    # 2. ANOMALIYA HISOBLASH
    # ──────────────────────────────────────────
    print("━" * 70)
    print(f"  2. ANOMALIYA HISOBLASH (referens: {REF_START}-{REF_END})")
    print("━" * 70)
    print()
    
    stations = {}
    
    # Denov anomaliya
    ref_mask_d = (df_denov.index >= REF_START) & (df_denov.index <= REF_END)
    denov_ref_mean = df_denov.loc[ref_mask_d, 'temp'].mean()
    df_denov['anomaly'] = df_denov['temp'] - denov_ref_mean
    stations['Denov'] = df_denov['anomaly']
    print(f"  Denov:      T_ref = {denov_ref_mean:.2f}°C (n={ref_mask_d.sum()} yil)")
    
    # Mingchuqur anomaliya
    ref_mask_m = (df_ming.index >= REF_START) & (df_ming.index <= REF_END)
    ming_ref_mean = df_ming.loc[ref_mask_m, 'temp'].mean()
    df_ming['anomaly'] = df_ming['temp'] - ming_ref_mean
    stations['Mingchuqur'] = df_ming['anomaly']
    print(f"  Mingchuqur: T_ref = {ming_ref_mean:.2f}°C (n={ref_mask_m.sum()} yil)")
    
    # Boysun anomaliya
    if len(df_boysun) > 0 and 'temp' in df_boysun.columns:
        ref_mask_b = (df_boysun.index >= REF_START) & (df_boysun.index <= REF_END)
        boysun_ref_mean = df_boysun.loc[ref_mask_b, 'temp'].mean()
        if pd.notna(boysun_ref_mean):
            df_boysun['anomaly'] = df_boysun['temp'] - boysun_ref_mean
            stations['Boysun'] = df_boysun['anomaly']
            print(f"  Boysun:     T_ref = {boysun_ref_mean:.2f}°C (n={ref_mask_b.sum()} yil)")
    
    n_stations = len(stations)
    print(f"\n  Jami stansiyalar: {n_stations} ({', '.join(stations.keys())})")
    print()
    
    # ──────────────────────────────────────────
    # 3. REGIONAL O'RTACHA HISOBLASH
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  3. REGIONAL O'RTACHA")
    print("━" * 70)
    print()
    
    # Barcha yillarni birlashtirish
    all_years = set()
    for name, series in stations.items():
        all_years.update(series.dropna().index.tolist())
    all_years = sorted(all_years)
    
    # Regional composite DataFrame
    df_regional = pd.DataFrame(index=all_years)
    df_regional.index.name = 'year'
    
    for name, series in stations.items():
        df_regional[name] = series
    
    # O'rtacha anomaliya (mavjud stansiyalar bo'yicha)
    df_regional['regional_anomaly'] = df_regional[list(stations.keys())].mean(axis=1)
    
    # N stansiya (har yil uchun)
    df_regional['n_stations'] = df_regional[list(stations.keys())].notna().sum(axis=1)
    
    print(f"  Davr: {min(all_years)}-{max(all_years)}")
    print(f"  Jami yillar: {len(all_years)}")
    print(f"  Yillar (3 stansiya): {(df_regional['n_stations'] == 3).sum()}")
    print(f"  Yillar (2 stansiya): {(df_regional['n_stations'] == 2).sum()}")
    print(f"  Yillar (1 stansiya): {(df_regional['n_stations'] == 1).sum()}")
    print()
    
    # Trend hisoblash (so'nggi 40 yil)
    recent = df_regional['regional_anomaly'].dropna()
    recent_40 = recent[recent.index >= 1983]
    if len(recent_40) > 5:
        from scipy import stats
        slope, intercept, r, p, se = stats.linregress(recent_40.index, recent_40.values)
        trend_decade = slope * 10
        print(f"  Trend (1983-hozir): {trend_decade:+.3f}°C/o'n yillik (p={p:.4f})")
    else:
        trend_decade = np.nan
    print()
    
    # ──────────────────────────────────────────
    # 4. GRAFIK
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  4. GRAFIK")
    print("━" * 70)
    print()
    
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'figure.dpi': 150})
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={'hspace': 0.3})
    
    # ── Panel (a): Regional harorat anomaliyasi ──
    ax1 = axes[0]
    
    years = df_regional.index.values
    anomaly = df_regional['regional_anomaly'].values
    
    # Ustunli diagramma (qizil/ko'k)
    colors = ['#E07060' if a >= 0 else '#6099C0' for a in anomaly]
    ax1.bar(years, anomaly, color=colors, alpha=0.7, width=0.8, zorder=2)
    
    # 11 yillik harakatlanuvchi o'rtacha
    ma = df_regional['regional_anomaly'].rolling(11, center=True, min_periods=6).mean()
    ax1.plot(ma.index, ma.values, color='#1a1a1a', lw=2.2, zorder=4,
             label="11 yillik harakatlanuvchi o'rtacha")
    
    # Trend chizig'i
    if pd.notna(trend_decade) and len(recent_40) > 5:
        trend_line = intercept + slope * recent_40.index.values
        ax1.plot(recent_40.index, trend_line, color='#1a1a1a', lw=1.3, ls=':',
                 zorder=3, label=f"Trend ({trend_decade:+.2f}°C/o'n yillik)")
    
    # Nol chizig'i
    ax1.axhline(y=0, color='#888888', ls='-', lw=0.5, alpha=0.5)
    
    ax1.set_xlabel('Yil (milodiy)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Harorat anomaliyasi (°C)', fontsize=11, fontweight='bold')
    ax1.set_title(f"Regional harorat anomaliyasi ({n_stations} stansiya o'rtachasi, ref: {REF_START}-{REF_END})",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.text(0.02, 0.95, 'a)', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', color='#CC0000')
    
    # ── Panel (b): Har bir stansiya alohida ──
    ax2 = axes[1]
    
    colors_st = {'Boysun': '#E63946', 'Denov': '#457B9D', 'Mingchuqur': '#2A9D8F'}
    
    for name, series in stations.items():
        s = series.dropna()
        # 11 yillik smoothed
        s_ma = s.rolling(11, center=True, min_periods=6).mean()
        ax2.plot(s_ma.index, s_ma.values, lw=1.8, 
                 color=colors_st.get(name, '#333'),
                 label=f"{name} (11 yillik)")
    
    ax2.axhline(y=0, color='#888888', ls='-', lw=0.5, alpha=0.5)
    ax2.set_xlabel('Yil (milodiy)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Harorat anomaliyasi (°C)', fontsize=11, fontweight='bold')
    ax2.set_title("Stansiyalar bo'yicha harorat anomaliyasi (11 yillik silliqlangan)",
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.text(0.02, 0.95, 'b)', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', color='#CC0000')
    
    plt.tight_layout()
    plt.savefig('regional_temperature_anomaly.png', dpi=300, bbox_inches='tight')
    plt.savefig('regional_temperature_anomaly.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ──────────────────────────────────────────
    # 5. SAQLASH
    # ──────────────────────────────────────────
    output = 'regional_temperature_anomaly.csv'
    df_regional.to_csv(output)
    
    print(f"  💾 {output}")
    print(f"  💾 regional_temperature_anomaly.png")
    print()
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    print()
    print("  📋 Maqolada yozish uchun:")
    print("  ─────────────────────────")
    print(f"  \"Regional harorat anomaliyasi {n_stations} ta stansiya")
    print(f"   (Boysun, Denov, Mingchuqur) yillik o'rtacha qiymatlarini")
    print(f"   {REF_START}-{REF_END} referens davriga nisbatan hisoblash")
    print(f"   orqali yaratildi. So'nggi 40 yillik trend")
    print(f"   {trend_decade:+.3f}°C/o'n yillik ni tashkil etadi.\"")
    print()
    
    return df_regional


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = run()

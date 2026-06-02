"""
Fig. 3: Broken Axis ΔT Figure
Left panel  → Paleodavr (ΔT speleothem reconstruction, TON-2)
Right panel → Instrumental davr (Boysun MS temperature anomaly)

HAQIQIY MA'LUMOTLARGA ASOSLANGAN:
- TON2_iso_original.csv → δ¹⁸O calcite → fractionation → ΔT
- Бойсун МС.xlsx → Instrumental annual temperature

Fractionation equations:
- Kim & O'Neil (1997): 1000 ln(α) = 18.03*(1000/T) - 32.42
- Tremaine et al. (2011): 1000 ln(α) = 16.1*(1000/T) - 24.6

Author: TON-2 paleoclimate study
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# RANGLAR (COLOR SCHEME)
# ══════════════════════════════════════════════════════════════

COLORS = {
    'deltaT_line': '#D73027',       # ΔT asosiy chiziq (qizil)
    'uncertainty': '#D73027',       # Noaniqlik shading
    'warm_bar': '#D73027',          # Issiq yillar (Boysun)
    'cold_bar': '#4393C3',          # Sovuq yillar (Boysun)
    'smooth_line': '#1A1A2E',       # Moving average (qora)
    'ton2_marker': '#7B2D8B',       # TON-2 nuqtalar (binafsha)
    'zero_line': '#666666',         # Nol chizig'i
    'hiatus': '#999999',            # Hiatus zonalari
    'trend_line': '#333333',        # Isish trendi
}

# ══════════════════════════════════════════════════════════════
# FIZIK KONSTANTALAR VA PARAMETRLAR
# ══════════════════════════════════════════════════════════════

# Cave temperature (Boysun + empirik lapse rate bilan aniqlangan)
T_CAVE_BEST = 5.675   # °C (notebook Cell 62 dan)

# Rekonstruksiya koeffitsiyenti (notebook Cell 63 dan)
COEFF_T_BEST = 0.54   # ‰/°C (drip water d18O vs T)
COEFF_T_LOW = 0.50    # ‰/°C (pastki chegara)
COEFF_T_HIGH = 0.58   # ‰/°C (yuqori chegara)

# Binning
BIN_W = 500  # 500 yillik bin

# MIS bosqichlari
MIS_STAGES = [
    {'name': 'MIS 1', 'start': 0, 'end': 11.7, 'color': '#FFF4E6'},
    {'name': 'MIS 2', 'start': 11.7, 'end': 29, 'color': '#E3F2FD'},
    {'name': 'MIS 3', 'start': 29, 'end': 57, 'color': '#FFF8E1'},
    {'name': 'MIS 4', 'start': 57, 'end': 71, 'color': '#E1F5FE'},
    {'name': 'MIS 5', 'start': 71, 'end': 130, 'color': '#FFFDE7'},
]

# Hiatus davrlari (notebook Cell 0 dagi katta bo'shliqlar asosida)
HIATUS_PERIODS_KA = [
    (36.9, 59.5),   # ~37 ka dan ~59.5 ka gacha
    (101.7, 113.2), # ~101.7 ka dan ~113.2 ka gacha
]

# ══════════════════════════════════════════════════════════════
# FRACTIONATION FUNKSIYALARI
# ══════════════════════════════════════════════════════════════

def alpha_kim_oneil(TK):
    """Kim & O'Neil (1997) calcite-water fractionation"""
    return np.exp((18.03 * (1000.0 / TK) - 32.42) / 1000.0)


def alpha_tremaine(TK):
    """Tremaine et al. (2011) fractionation"""
    return np.exp((16.1 * (1000.0 / TK) - 24.6) / 1000.0)


def vpdb_to_vsmow(d18O_vpdb):
    """VPDB → VSMOW konversiyasi"""
    return 1.03091 * d18O_vpdb + 30.91


def water_from_calcite(d18O_calcite_smow, alpha):
    """Calcite d18O → Water d18O"""
    return ((1000.0 + d18O_calcite_smow) / alpha) - 1000.0



# ══════════════════════════════════════════════════════════════
# MA'LUMOTLARNI YUKLASH VA QAYTA ISHLASH
# ══════════════════════════════════════════════════════════════

def load_ton2_and_compute_deltaT():
    """
    TON2_iso_original.csv dan δ¹⁸O ni o'qib, ΔT rekonstruksiya qilish.
    
    Jarayon:
    1. TON2 d18O calcite (VPDB) → VSMOW konversiya
    2. Fractionation equation (Kim&O'Neil va Tremaine) → d18O water
    3. Modern reference (age ≤ 10 BP) dan anomaliya hisoblash
    4. d18O water anomaliya / koeffitsient → ΔT (°C)
    """
    print("📂 TON2_iso_original.csv yuklanmoqda...")
    ton2 = pd.read_csv("TON2_iso_original.csv")
    ton2 = ton2.dropna(subset=["age_calBP", "d18OcarbVPDB"]).copy()
    ton2 = ton2.sort_values("age_calBP").reset_index(drop=True)
    print(f"   ✅ {len(ton2)} rows | {ton2['age_calBP'].min():.0f} → {ton2['age_calBP'].max():.0f} cal yr BP")

    # ── Zamonaviy reference ──────────────────────────────────
    modern = ton2[ton2["age_calBP"] <= 10].copy()
    modern_d18O_vpdb = modern["d18OcarbVPDB"].mean()
    modern_d18O_smow = vpdb_to_vsmow(modern_d18O_vpdb)

    T_K_best = T_CAVE_BEST + 273.15
    a_ko_best = alpha_kim_oneil(T_K_best)
    a_tr_best = alpha_tremaine(T_K_best)
    w_ko_modern = water_from_calcite(modern_d18O_smow, a_ko_best)
    w_tr_modern = water_from_calcite(modern_d18O_smow, a_tr_best)
    w_modern = (w_ko_modern + w_tr_modern) / 2

    print(f"   Modern reference (n={len(modern)}): d18O_c = {modern_d18O_vpdb:.3f}‰ VPDB")
    print(f"   d18O_w modern (mean KO+TR): {w_modern:.3f}‰ VSMOW")

    # ── 500-yillik binning ───────────────────────────────────
    # Faqat paleo qism (age >= 0)
    ton2_paleo = ton2[ton2["age_calBP"] >= 0].copy()
    ton2_paleo["age_bin"] = (ton2_paleo["age_calBP"] // BIN_W) * BIN_W

    b = ton2_paleo.groupby("age_bin", as_index=False).agg(
        d18O_vpdb=('d18OcarbVPDB', 'mean'),
        d18O_std=('d18OcarbVPDB', 'std'),
        n_samples=('d18OcarbVPDB', 'count')
    )

    # ── VPDB → VSMOW ────────────────────────────────────────
    b["d18O_smow"] = vpdb_to_vsmow(b["d18O_vpdb"])

    # ── Fractionation → water d18O ──────────────────────────
    T_K = T_CAVE_BEST + 273.15
    a_ko = alpha_kim_oneil(T_K)
    a_tr = alpha_tremaine(T_K)

    b["d18O_w_ko"] = water_from_calcite(b["d18O_smow"], a_ko)
    b["d18O_w_tr"] = water_from_calcite(b["d18O_smow"], a_tr)
    b["d18O_w"] = (b["d18O_w_ko"] + b["d18O_w_tr"]) / 2

    # ── Noaniqlik ────────────────────────────────────────────
    # Fractionation equation farqi (Kim&O'Neil vs Tremaine)
    b["unc_frac"] = (b["d18O_w_ko"] - b["d18O_w_tr"]).abs() / 2
    # Analitik noaniqlik
    b["unc_anal"] = b["d18O_std"] / np.sqrt(b["n_samples"].clip(1))
    # Umumiy noaniqlik (RSS)
    b["unc_total"] = np.sqrt(b["unc_frac"]**2 + b["unc_anal"]**2)

    # ── Water d18O anomaliya (modern ga nisbatan) ────────────
    b["dW"] = b["d18O_w"] - w_modern

    # ── ΔT hisoblash ─────────────────────────────────────────
    # Best estimate: COEFF_T_BEST = 0.54 ‰/°C
    b["dT_best"] = b["dW"] / COEFF_T_BEST
    b["dT_unc"] = b["unc_total"] / COEFF_T_BEST

    # Koeffitsient noaniqligidan keladigan qo'shimcha unc
    b["dT_low"] = b["dW"] / COEFF_T_HIGH   # 0.58 → kichikroq ΔT
    b["dT_high"] = b["dW"] / COEFF_T_LOW   # 0.50 → kattaroq ΔT

    # Umumiy noaniqlik (fractionation + koeffitsient)
    b["dT_unc_combined"] = np.sqrt(
        b["dT_unc"]**2 +
        ((b["dT_high"] - b["dT_low"]) / 2)**2
    )

    # ── Smoothing (5 bin = 2500 yil moving average) ──────────
    b["dT_smooth"] = uniform_filter1d(b["dT_best"].values, size=5)

    # ── Age ni ka ga o'tkazish ───────────────────────────────
    b["age_ka"] = b["age_bin"] / 1000.0

    print(f"   ✅ ΔT rekonstruksiya: {len(b)} bins")
    print(f"   ΔT range: {b['dT_best'].min():.1f} to {b['dT_best'].max():.1f} °C")

    return b



def load_boysun_instrumental():
    """
    Бойсун МС.xlsx dan yillik harorat anomaliyasi hisoblash.
    Baseline: 1961-1990
    
    Agar Excel fayl topilmasa, boysun_harorat_tozalangan_va_indekslar.csv ni sinaydi.
    """
    print("\n📂 Boysun MS ma'lumotlarini yuklanmoqda...")

    boysun_df = None

    # Variant 1: Excel fayl
    try:
        raw = pd.read_excel("Бойсун МС.xlsx", sheet_name='Ҳарорат')
        # Ustunlarni rename
        df = raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
        df = df.rename(columns={
            "янв": "jan", "фев": "feb", "мар": "mar", "апр": "apr",
            "май": "may", "июн": "jun", "июл": "jul", "авг": "aug",
            "сен": "sep", "окт": "oct", "ноя": "nov", "дек": "dec",
            "йил": "annual"
        })
        # Year ustunini topish
        first_col = df.columns[0]
        if first_col.startswith("Unnamed") or first_col not in [
            "year", "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec", "annual"
        ]:
            df = df.rename(columns={first_col: "year"})

        if "year" not in df.columns:
            df.insert(0, "year", raw.iloc[:, 0])

        # Son tipiga o'tkazish
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        for c in [x for x in df.columns if x != "year"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
        boysun_df = df
        print(f"   ✅ Excel dan yuklandi: {len(boysun_df)} yil")

    except Exception as e:
        print(f"   ⚠️  Excel yuklanmadi: {e}")

    # Variant 2: CSV fallback
    if boysun_df is None:
        try:
            boysun_df = pd.read_csv("boysun_harorat_tozalangan_va_indekslar.csv")
            boysun_df = boysun_df.rename(columns={"йил": "annual", "yil": "annual"})
            # Year ustunini topish
            year_col = None
            for c in boysun_df.columns:
                if any(x in c.lower() for x in ['год', 'year', 'yil', 'йил']):
                    year_col = c
                    break
            if year_col and year_col != "year":
                boysun_df = boysun_df.rename(columns={year_col: "year"})

            boysun_df["year"] = pd.to_numeric(boysun_df["year"], errors="coerce")
            boysun_df["annual"] = pd.to_numeric(
                boysun_df["annual"].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
            print(f"   ✅ CSV dan yuklandi: {len(boysun_df)} yil")

        except Exception as e2:
            print(f"   ❌ CSV ham yuklanmadi: {e2}")
            raise FileNotFoundError(
                "Boysun MS ma'lumotlari topilmadi! "
                "Iltimos 'Бойсун МС.xlsx' yoki "
                "'boysun_harorat_tozalangan_va_indekslar.csv' faylini qo'ying."
            )

    # ── Annual haroratni olish ───────────────────────────────
    if "annual" not in boysun_df.columns:
        # Oylik o'rtachadan annual hisoblash
        month_cols = ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug", "sep", "oct", "nov", "dec"]
        available = [c for c in month_cols if c in boysun_df.columns]
        if available:
            boysun_df["annual"] = boysun_df[available].mean(axis=1)
        else:
            raise ValueError("Boysun jadvalida harorat ustunlari topilmadi!")

    d = boysun_df[["year", "annual"]].dropna().copy()
    d["year"] = d["year"].astype(int)
    d = d.sort_values("year").reset_index(drop=True)

    # ── 3σ artifact olib tashlash ────────────────────────────
    mean_t = d["annual"].mean()
    std_t = d["annual"].std()
    d = d[(d["annual"] > mean_t - 3 * std_t) &
          (d["annual"] < mean_t + 3 * std_t)].copy()

    # ── Anomaliya (1961-1990 baseline) ───────────────────────
    baseline = d[(d["year"] >= 1961) & (d["year"] <= 1990)]
    if len(baseline) < 10:
        # Fallback: butun davr
        baseline_mean = d["annual"].mean()
    else:
        baseline_mean = baseline["annual"].mean()

    d["anomaly"] = d["annual"] - baseline_mean

    # ── 11-yillik moving average ─────────────────────────────
    d["ma11"] = d["anomaly"].rolling(11, center=True, min_periods=6).mean()

    # ── Trend (1990 dan keyin) ───────────────────────────────
    recent = d[d["year"] >= 1990].dropna(subset=["anomaly"])
    if len(recent) >= 5:
        trend_coeff = np.polyfit(recent["year"], recent["anomaly"], 1)
        trend_per_decade = trend_coeff[0] * 10
    else:
        trend_coeff = [0, 0]
        trend_per_decade = 0

    print(f"   Davr: {d['year'].min()} – {d['year'].max()}")
    print(f"   Baseline (1961-1990): {baseline_mean:.2f}°C")
    print(f"   Warming trend (1990+): +{trend_per_decade:.3f}°C/decade")

    return d, trend_coeff, trend_per_decade



# ══════════════════════════════════════════════════════════════
# BROKEN AXIS FIGURE YARATISH
# ══════════════════════════════════════════════════════════════

def create_broken_axis_figure(paleo_data, boysun_data, trend_coeff, trend_per_decade):
    """
    Broken axis figure:
    Left panel (70%) → Paleodavr ΔT (speleothem)
    Right panel (30%) → Instrumental davr (Boysun MS)
    """

    print("\n🎨 Figure yaratilmoqda...")

    # ── Figure setup ─────────────────────────────────────────
    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.linewidth': 1.2,
        'figure.dpi': 300,
    })

    fig = plt.figure(figsize=(16, 7), facecolor='white')

    # GridSpec: 70% left, 30% right with small gap
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[0.70, 0.30],
        wspace=0.08,
        left=0.06, right=0.96,
        top=0.88, bottom=0.14
    )

    ax_paleo = fig.add_subplot(gs[0, 0])
    ax_instr = fig.add_subplot(gs[0, 1])

    # ══════════════════════════════════════════════════════════
    # CHAP PANEL: PALEODAVR ΔT
    # ══════════════════════════════════════════════════════════

    age_ka = paleo_data["age_ka"].values
    dT_best = paleo_data["dT_best"].values
    dT_smooth = paleo_data["dT_smooth"].values
    dT_unc = paleo_data["dT_unc_combined"].values

    # MIS bosqichlari (yuqorida, yengil)
    for mis in MIS_STAGES:
        if mis['start'] / 1000 <= age_ka.max():
            ax_paleo.axvspan(
                mis['start'], mis['end'],
                ymin=0.92, ymax=1.0,
                color=mis['color'], alpha=0.7,
                zorder=0, linewidth=0
            )
            mid = (mis['start'] + mis['end']) / 2
            if mid <= age_ka.max():
                ax_paleo.text(
                    mid, ax_paleo.get_ylim()[1] if ax_paleo.get_ylim()[1] != 1 else 8,
                    mis['name'],
                    ha='center', va='bottom', fontsize=7.5,
                    color='#555555', fontweight='600',
                    transform=ax_paleo.get_xaxis_transform(),
                    y=0.94
                )

    # Hiatus zonalari (grey shading)
    for h_start, h_end in HIATUS_PERIODS_KA:
        ax_paleo.axvspan(
            h_start, h_end,
            color=COLORS['hiatus'], alpha=0.15,
            zorder=0, edgecolor='#888888',
            linewidth=0.5, linestyle='--'
        )
        mid = (h_start + h_end) / 2
        ax_paleo.text(
            mid, -6.5, 'HIATUS',
            ha='center', va='center', fontsize=7,
            color='#888888', style='italic',
            rotation=90, alpha=0.7
        )

    # Nol chizig'i (modern reference)
    ax_paleo.axhline(
        y=0, color=COLORS['zero_line'],
        linewidth=1.0, linestyle='--', alpha=0.6, zorder=1
    )

    # Noaniqlik oralig'i (shading)
    # Ikki qatlam: 1) fractionation + analytical, 2) + coefficient
    ax_paleo.fill_between(
        age_ka,
        dT_smooth - dT_unc,
        dT_smooth + dT_unc,
        color=COLORS['uncertainty'], alpha=0.15,
        zorder=2, label='Uncertainty (frac. + coeff.)'
    )

    # ΔT smoothed chiziq (asosiy)
    ax_paleo.plot(
        age_ka, dT_smooth,
        color=COLORS['deltaT_line'], linewidth=2.0,
        zorder=4, label='ΔT (smoothed)'
    )

    # ── Muhim annotatsiyalar ─────────────────────────────────
    # LGM (~21 ka)
    lgm_mask = (age_ka >= 19) & (age_ka <= 23)
    if lgm_mask.any():
        lgm_idx = np.argmin(dT_smooth[lgm_mask])
        lgm_age = age_ka[lgm_mask][lgm_idx]
        lgm_val = dT_smooth[lgm_mask][lgm_idx]
        ax_paleo.annotate(
            f'LGM\n~{lgm_val:.1f}°C',
            xy=(lgm_age, lgm_val),
            xytext=(lgm_age + 8, lgm_val - 1.5),
            arrowprops=dict(
                arrowstyle='->', lw=1.5,
                color='#333333', connectionstyle='arc3,rad=0.2'
            ),
            fontsize=9, fontweight='bold', ha='center',
            bbox=dict(
                boxstyle='round,pad=0.4', facecolor='white',
                edgecolor='#333333', linewidth=1, alpha=0.95
            )
        )

    # MIS 5e (~125 ka)
    mis5e_mask = (age_ka >= 120) & (age_ka <= 130)
    if mis5e_mask.any():
        mis5e_idx = np.argmax(dT_smooth[mis5e_mask])
        mis5e_age = age_ka[mis5e_mask][mis5e_idx]
        mis5e_val = dT_smooth[mis5e_mask][mis5e_idx]
        ax_paleo.annotate(
            f'MIS 5e\n~+{mis5e_val:.1f}°C',
            xy=(mis5e_age, mis5e_val),
            xytext=(mis5e_age - 10, mis5e_val + 2),
            arrowprops=dict(
                arrowstyle='->', lw=1.5,
                color='#333333', connectionstyle='arc3,rad=-0.2'
            ),
            fontsize=9, fontweight='bold', ha='center',
            bbox=dict(
                boxstyle='round,pad=0.4', facecolor='white',
                edgecolor='#333333', linewidth=1, alpha=0.95
            )
        )

    # Holosen optimum (~8-10 ka)
    hol_mask = (age_ka >= 8) & (age_ka <= 12)
    if hol_mask.any():
        hol_idx = np.argmax(dT_smooth[hol_mask])
        hol_age = age_ka[hol_mask][hol_idx]
        hol_val = dT_smooth[hol_mask][hol_idx]
        ax_paleo.annotate(
            'Holocene\nOptimum',
            xy=(hol_age, hol_val),
            xytext=(hol_age + 8, hol_val + 1.5),
            arrowprops=dict(
                arrowstyle='->', lw=1.5,
                color='#333333', connectionstyle='arc3,rad=0.3'
            ),
            fontsize=9, fontweight='bold', ha='center',
            bbox=dict(
                boxstyle='round,pad=0.4', facecolor='white',
                edgecolor='#333333', linewidth=1, alpha=0.95
            )
        )

    # ── Chap panel formatting ────────────────────────────────
    ax_paleo.set_xlim(age_ka.max() + 2, age_ka.min() - 1)  # Inverted
    y_max_abs = max(abs(dT_smooth.min()), abs(dT_smooth.max())) + 3
    ax_paleo.set_ylim(-y_max_abs, y_max_abs)
    ax_paleo.set_xlabel('Age (ka BP) → older', fontsize=11, fontweight='bold')
    ax_paleo.set_ylabel('ΔT (°C, vs modern)', fontsize=11, fontweight='bold')
    ax_paleo.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax_paleo.legend(
        loc='lower left', fontsize=8.5, frameon=True,
        framealpha=0.95, edgecolor='#CCCCCC', fancybox=False
    )
    ax_paleo.text(
        0.02, 0.97, 'a  Paleodavr (TON-2 speleothem)',
        transform=ax_paleo.transAxes, fontsize=11,
        fontweight='bold', va='top', ha='left',
        color='#333333'
    )

    # MIS labels yuqorida
    for mis in MIS_STAGES:
        mid = (mis['start'] + mis['end']) / 2
        if mid <= age_ka.max():
            ax_paleo.text(
                mid, 0.96, mis['name'],
                ha='center', va='top', fontsize=7.5,
                color='#555555', fontweight='600',
                transform=ax_paleo.get_xaxis_transform()
            )

    # Spines - o'ng tomonni olib tashlash
    ax_paleo.spines['right'].set_visible(False)
    ax_paleo.tick_params(axis='y', right=False)



    # ══════════════════════════════════════════════════════════
    # O'NG PANEL: INSTRUMENTAL DAVR (BOYSUN MS)
    # ══════════════════════════════════════════════════════════

    years = boysun_data["year"].values
    anomaly = boysun_data["anomaly"].values
    moving_avg = boysun_data["ma11"].values

    # Bar chart: annual anomaly
    bar_colors = [COLORS['warm_bar'] if a > 0 else COLORS['cold_bar']
                  for a in anomaly]

    ax_instr.bar(
        years, anomaly,
        color=bar_colors, width=0.8,
        alpha=0.7, zorder=2,
        edgecolor='none'
    )

    # Nol chizig'i
    ax_instr.axhline(
        y=0, color=COLORS['zero_line'],
        linewidth=1.0, linestyle='--', alpha=0.6, zorder=1
    )

    # 11-yillik moving average (qora chiziq)
    ax_instr.plot(
        years, moving_avg,
        color=COLORS['smooth_line'], linewidth=2.2,
        zorder=5, label='11-yr moving average'
    )

    # Isish trendi (dotted line, 1990+)
    trend_years = np.arange(1990, years.max() + 1)
    trend_line = np.polyval(trend_coeff, trend_years)
    ax_instr.plot(
        trend_years, trend_line,
        color=COLORS['trend_line'], linewidth=1.5,
        linestyle=':', zorder=4,
        label=f'Warming trend (+{trend_per_decade:.2f}°C/decade)'
    )

    # ── TON-2 zamonaviy nuqtalar (1941, 2008) ───────────────
    # 1941 nuqtasi
    mask_1941 = boysun_data["year"] == 1941
    if mask_1941.any():
        val_1941 = boysun_data.loc[mask_1941, "anomaly"].values[0]
        ax_instr.plot(
            1941, val_1941,
            marker='D', color=COLORS['ton2_marker'],
            ms=10, zorder=6,
            markeredgecolor='white', markeredgewidth=1.5
        )
        ax_instr.annotate(
            'TON-2\n(1941)',
            xy=(1941, val_1941),
            xytext=(1948, val_1941 - 0.8),
            arrowprops=dict(
                arrowstyle='->', lw=1.2,
                color=COLORS['ton2_marker']
            ),
            fontsize=8, fontweight='bold',
            color=COLORS['ton2_marker'], ha='center',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white',
                edgecolor=COLORS['ton2_marker'], linewidth=0.8, alpha=0.95
            )
        )

    # 2008 nuqtasi
    mask_2008 = boysun_data["year"] == 2008
    if mask_2008.any():
        val_2008 = boysun_data.loc[mask_2008, "anomaly"].values[0]
        ax_instr.plot(
            2008, val_2008,
            marker='D', color=COLORS['ton2_marker'],
            ms=10, zorder=6,
            markeredgecolor='white', markeredgewidth=1.5
        )
        ax_instr.annotate(
            'TON-2\n(2008)',
            xy=(2008, val_2008),
            xytext=(2000, val_2008 + 0.7),
            arrowprops=dict(
                arrowstyle='->', lw=1.2,
                color=COLORS['ton2_marker']
            ),
            fontsize=8, fontweight='bold',
            color=COLORS['ton2_marker'], ha='center',
            bbox=dict(
                boxstyle='round,pad=0.3', facecolor='white',
                edgecolor=COLORS['ton2_marker'], linewidth=0.8, alpha=0.95
            )
        )

    # ── Baza davri annotation ────────────────────────────────
    ax_instr.axvspan(
        1961, 1990, color='#E8E8E8', alpha=0.3, zorder=0
    )
    y_lim = ax_instr.get_ylim()
    ax_instr.text(
        1975.5, -2.0, '1961–1990\nbaseline',
        ha='center', va='bottom', fontsize=7,
        color='#888888', style='italic'
    )

    # ── O'ng panel formatting ────────────────────────────────
    ax_instr.set_xlim(years.min() - 2, years.max() + 2)
    ax_instr.set_ylim(-2.5, 2.5)
    ax_instr.set_xlabel('Year (CE)', fontsize=11, fontweight='bold')
    ax_instr.set_ylabel('T anomaly (°C, 1961–1990)', fontsize=11, fontweight='bold')
    ax_instr.yaxis.set_label_position('right')
    ax_instr.yaxis.tick_right()
    ax_instr.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax_instr.legend(
        loc='upper left', fontsize=8, frameon=True,
        framealpha=0.95, edgecolor='#CCCCCC', fancybox=False
    )
    ax_instr.text(
        0.03, 0.97, 'b  Instrumental (Boysun MS)',
        transform=ax_instr.transAxes, fontsize=11,
        fontweight='bold', va='top', ha='left',
        color='#333333'
    )

    # Spines - chap tomonni olib tashlash
    ax_instr.spines['left'].set_visible(False)
    ax_instr.tick_params(axis='y', left=False)



    # ══════════════════════════════════════════════════════════
    # BROKEN AXIS MARKER (//)
    # ══════════════════════════════════════════════════════════

    d = 0.015  # size of diagonal lines
    kwargs = dict(
        transform=fig.transFigure, color='#333333',
        linewidth=1.5, clip_on=False, zorder=10
    )

    # Panel chegaralarini olish
    fig.canvas.draw()
    paleo_bbox = ax_paleo.get_position()
    instr_bbox = ax_instr.get_position()

    # Ikki panel orasidagi x pozitsiya
    x_break = (paleo_bbox.x1 + instr_bbox.x0) / 2

    # "//" belgisini bir necha balandlikda chizish
    for y_frac in [0.3, 0.5, 0.7]:
        y_pos = paleo_bbox.y0 + (paleo_bbox.y1 - paleo_bbox.y0) * y_frac
        fig.lines.append(plt.Line2D(
            [x_break - d, x_break + d],
            [y_pos - d, y_pos + d],
            **kwargs
        ))
        fig.lines.append(plt.Line2D(
            [x_break - d - 0.005, x_break + d - 0.005],
            [y_pos - d, y_pos + d],
            **kwargs
        ))

    # ══════════════════════════════════════════════════════════
    # FIGURE TITLE & CAPTION
    # ══════════════════════════════════════════════════════════

    fig.suptitle(
        'Temperature Reconstruction: TON-2 Speleothem vs Instrumental Record',
        fontsize=13, fontweight='bold', y=0.95,
        color='#1A1A2E'
    )

    caption = (
        "Fig. 3. Broken-axis comparison of paleotemperature and instrumental records. "
        "(a) ΔT reconstruction from TON-2 speleothem δ¹⁸O using Kim & O'Neil (1997) "
        "and Tremaine et al. (2011) fractionation equations (red line = 2500-yr smoothed). "
        "Shaded envelope represents combined uncertainty from fractionation equation choice "
        "and drip-water coefficient range (0.50–0.58 ‰/°C). "
        "Gray zones indicate growth hiatuses. MIS stages shown at top. "
        "(b) Boysun meteorological station (1000 m a.s.l.) annual temperature anomaly "
        "relative to 1961–1990 baseline. Red/blue bars = warm/cold years; black line = "
        "11-yr moving average. Purple diamonds mark TON-2 modern calibration points "
        "(1941, 2008). Dotted line shows recent warming trend."
    )

    fig.text(
        0.06, 0.01, caption,
        fontsize=7.5, color='#444444', style='italic',
        ha='left', va='bottom', wrap=True,
        multialignment='left'
    )

    return fig



# ══════════════════════════════════════════════════════════════
# ASOSIY DASTUR
# ══════════════════════════════════════════════════════════════

def main():
    """Asosiy funksiya"""

    print("=" * 60)
    print("FIG. 3: BROKEN AXIS ΔT FIGURE")
    print("  Haqiqiy ma'lumotlarga asoslangan")
    print("  TON2_iso_original.csv + Boysun MS")
    print("=" * 60)
    print()

    # ── 1. Paleo data: TON-2 ΔT rekonstruksiya ──────────────
    paleo_data = load_ton2_and_compute_deltaT()

    # ── 2. Instrumental data: Boysun MS ─────────────────────
    boysun_data, trend_coeff, trend_per_decade = load_boysun_instrumental()

    # ── 3. Figure yaratish ───────────────────────────────────
    fig = create_broken_axis_figure(
        paleo_data, boysun_data, trend_coeff, trend_per_decade
    )

    # ── 4. Saqlash ───────────────────────────────────────────
    print("\n💾 Saqlash...")
    fig.savefig(
        'fig3_broken_axis_deltaT.png',
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    fig.savefig(
        'fig3_broken_axis_deltaT.pdf',
        dpi=300, bbox_inches='tight', facecolor='white'
    )

    print("   ✅ fig3_broken_axis_deltaT.png")
    print("   ✅ fig3_broken_axis_deltaT.pdf")

    plt.close(fig)

    print()
    print("=" * 60)
    print("✨ TAYYOR! Broken axis figure yaratildi.")
    print("=" * 60)
    print()
    print("📋 Xususiyatlar:")
    print("   • Chap panel (70%): ΔT rekonstruksiya (haqiqiy TON-2 data)")
    print("     - Kim & O'Neil (1997) + Tremaine (2011) fractionation")
    print("     - T_cave = 5.675°C (empirik lapse rate)")
    print("     - Koeff: 0.54 ‰/°C (noaniqlik: 0.50–0.58)")
    print("     - 500-yr bins, 2500-yr smoothing")
    print("   • O'ng panel (30%): Boysun instrumental")
    print("     - Anomaliya: 1961-1990 baseline")
    print("     - 11-yr moving average")
    print("     - Warming trend (1990+)")
    print("     - TON-2 kalibratsiya nuqtalar (1941, 2008)")
    print("   • Broken axis '//' belgisi")
    print()


if __name__ == "__main__":
    main()

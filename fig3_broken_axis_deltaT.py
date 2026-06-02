"""
1-rasm: Broken Axis ΔT Figure
Chap panel  → Paleodavr (ΔT stalagmit rekonstruksiyasi, TON-2)
O'ng panel  → Instrumental davr (Boysun MS harorat anomaliyasi)

HAQIQIY MA'LUMOTLARGA ASOSLANGAN:
- TON2_iso_original.csv → δ¹⁸O kalsit → fraktsionatsiya → ΔT
- Бойсун МС.xlsx → Instrumental yillik harorat

O'zbekiston ilmiy jurnallari uslubida.
Quaternary Science Reviews ranglar palitrasida.

Author: TON-2 paleoklimat tadqiqoti
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# RANGLAR — O'zbekiston ilmiy jurnallari / QSR uslubi
# ══════════════════════════════════════════════════════════════

COLORS = {
    'deltaT_line': '#333333',       # ΔT asosiy chiziq (qo'ng'ir-qora)
    'uncertainty': '#CCCCCC',       # Noaniqlik shading (och kulrang)
    'warm_bar': '#C0392B',          # Ijobiy anomaliya (Boysun)
    'cold_bar': '#2980B9',          # Salbiy anomaliya (Boysun)
    'smooth_line': '#1A1A2E',       # Harakatlanuvchi o'rtacha (qora)
    'ton2_marker': '#6C3483',       # TON-2 nuqtalar (to'q binafsha)
    'zero_line': '#666666',         # Nol chizig'i
    'hiatus': '#E0E0E0',           # Hiatus zonalari (och kulrang)
    'trend_line': '#666666',        # Isish trendi
    'mis_warm': '#FDEBD0',          # DIB iliq davrlar (yumshoq shaftoli)
    'mis_cold': '#D6EAF8',          # DIB sovuq davrlar (och ko'k)
}

# ══════════════════════════════════════════════════════════════
# FIZIK KONSTANTALAR VA PARAMETRLAR
# ══════════════════════════════════════════════════════════════

T_CAVE_BEST = 5.675   # °C
COEFF_T_BEST = 0.54   # ‰/°C
COEFF_T_LOW = 0.50    # ‰/°C
COEFF_T_HIGH = 0.58   # ‰/°C
BIN_W = 500            # 500 yillik bin

# MIS bosqichlari (iliq/sovuq ranglar) — DIB (Dengiz izotop bosqichi)
MIS_STAGES = [
    {'name': 'DIB 1', 'start': 0, 'end': 11.7, 'color': COLORS['mis_warm']},
    {'name': 'DIB 2', 'start': 11.7, 'end': 29, 'color': COLORS['mis_cold']},
    {'name': 'DIB 3', 'start': 29, 'end': 57, 'color': COLORS['mis_warm']},
    {'name': 'DIB 4', 'start': 57, 'end': 71, 'color': COLORS['mis_cold']},
    {'name': 'DIB 5', 'start': 71, 'end': 130, 'color': COLORS['mis_warm']},
]

# Hiatus davrlari (ka)
HIATUS_PERIODS_KA = [
    (36.9, 59.5),
    (101.7, 113.2),
]

# ══════════════════════════════════════════════════════════════
# FRACTIONATION FUNKSIYALARI
# ══════════════════════════════════════════════════════════════

def alpha_kim_oneil(TK):
    """Kim & O'Neil (1997)"""
    return np.exp((18.03 * (1000.0 / TK) - 32.42) / 1000.0)

def alpha_tremaine(TK):
    """Tremaine et al. (2011)"""
    return np.exp((16.1 * (1000.0 / TK) - 24.6) / 1000.0)

def vpdb_to_vsmow(d18O_vpdb):
    return 1.03091 * d18O_vpdb + 30.91

def water_from_calcite(d18O_calcite_smow, alpha):
    return ((1000.0 + d18O_calcite_smow) / alpha) - 1000.0


# ══════════════════════════════════════════════════════════════
# MA'LUMOTLARNI YUKLASH
# ══════════════════════════════════════════════════════════════

def load_ton2_and_compute_deltaT():
    """TON2_iso_original.csv dan ΔT rekonstruksiya."""
    print("📂 TON2_iso_original.csv yuklanmoqda...")
    ton2 = pd.read_csv("TON2_iso_original.csv")
    ton2 = ton2.dropna(subset=["age_calBP", "d18OcarbVPDB"]).copy()
    ton2 = ton2.sort_values("age_calBP").reset_index(drop=True)
    print(f"   ✅ {len(ton2)} rows | "
          f"{ton2['age_calBP'].min():.0f} → {ton2['age_calBP'].max():.0f} cal yr BP")

    # Zamonaviy reference
    modern = ton2[ton2["age_calBP"] <= 10].copy()
    modern_d18O_vpdb = modern["d18OcarbVPDB"].mean()
    modern_d18O_smow = vpdb_to_vsmow(modern_d18O_vpdb)

    T_K_best = T_CAVE_BEST + 273.15
    w_ko_modern = water_from_calcite(modern_d18O_smow, alpha_kim_oneil(T_K_best))
    w_tr_modern = water_from_calcite(modern_d18O_smow, alpha_tremaine(T_K_best))
    w_modern = (w_ko_modern + w_tr_modern) / 2

    print(f"   Modern ref (n={len(modern)}): d18O_c={modern_d18O_vpdb:.3f}‰, "
          f"d18O_w={w_modern:.3f}‰")

    # 500-yr binning
    ton2_paleo = ton2[ton2["age_calBP"] >= 0].copy()
    ton2_paleo["age_bin"] = (ton2_paleo["age_calBP"] // BIN_W) * BIN_W

    b = ton2_paleo.groupby("age_bin", as_index=False).agg(
        d18O_vpdb=('d18OcarbVPDB', 'mean'),
        d18O_std=('d18OcarbVPDB', 'std'),
        n_samples=('d18OcarbVPDB', 'count')
    )

    b["d18O_smow"] = vpdb_to_vsmow(b["d18O_vpdb"])
    T_K = T_CAVE_BEST + 273.15
    a_ko = alpha_kim_oneil(T_K)
    a_tr = alpha_tremaine(T_K)

    b["d18O_w_ko"] = water_from_calcite(b["d18O_smow"], a_ko)
    b["d18O_w_tr"] = water_from_calcite(b["d18O_smow"], a_tr)
    b["d18O_w"] = (b["d18O_w_ko"] + b["d18O_w_tr"]) / 2

    b["unc_frac"] = (b["d18O_w_ko"] - b["d18O_w_tr"]).abs() / 2
    b["unc_anal"] = b["d18O_std"] / np.sqrt(b["n_samples"].clip(1))
    b["unc_total"] = np.sqrt(b["unc_frac"]**2 + b["unc_anal"]**2)

    b["dW"] = b["d18O_w"] - w_modern
    b["dT_best"] = b["dW"] / COEFF_T_BEST
    b["dT_unc"] = b["unc_total"] / COEFF_T_BEST
    b["dT_low"] = b["dW"] / COEFF_T_HIGH
    b["dT_high"] = b["dW"] / COEFF_T_LOW
    b["dT_unc_combined"] = np.sqrt(
        b["dT_unc"]**2 + ((b["dT_high"] - b["dT_low"]) / 2)**2
    )

    b["dT_smooth"] = uniform_filter1d(b["dT_best"].values, size=5)
    b["age_ka"] = b["age_bin"] / 1000.0

    print(f"   ✅ ΔT: {len(b)} bins | range: "
          f"{b['dT_best'].min():.1f} to {b['dT_best'].max():.1f} °C")
    return b


def load_boysun_instrumental():
    """Boysun MS dan anomaliya hisoblash."""
    print("\n📂 Boysun MS yuklanmoqda...")
    boysun_df = None

    # Excel
    try:
        raw = pd.read_excel("Бойсун МС.xlsx", sheet_name='Ҳарорат')
        df = raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
        df = df.rename(columns={
            "янв": "jan", "фев": "feb", "мар": "mar", "апр": "apr",
            "май": "may", "июн": "jun", "июл": "jul", "авг": "aug",
            "сен": "sep", "окт": "oct", "ноя": "nov", "дек": "dec",
            "йил": "annual"
        })
        first_col = df.columns[0]
        if first_col.startswith("Unnamed") or first_col not in [
            "year", "jan", "feb", "mar", "apr", "may", "jun",
            "jul", "aug", "sep", "oct", "nov", "dec", "annual"
        ]:
            df = df.rename(columns={first_col: "year"})
        if "year" not in df.columns:
            df.insert(0, "year", raw.iloc[:, 0])
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.replace(",", ".", regex=False)
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
        for c in [x for x in df.columns if x != "year"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["year"]).sort_values("year").reset_index(drop=True)
        boysun_df = df
        print(f"   ✅ Excel: {len(boysun_df)} yil")
    except Exception as e:
        print(f"   ⚠️  Excel: {e}")

    # CSV fallback
    if boysun_df is None:
        try:
            boysun_df = pd.read_csv("boysun_harorat_tozalangan_va_indekslar.csv")
            boysun_df = boysun_df.rename(columns={"йил": "annual", "yil": "annual"})
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
                errors="coerce")
            print(f"   ✅ CSV: {len(boysun_df)} yil")
        except Exception as e2:
            raise FileNotFoundError(
                "Boysun MS topilmadi! 'Бойсун МС.xlsx' yoki "
                "'boysun_harorat_tozalangan_va_indekslar.csv' kerak.")

    if "annual" not in boysun_df.columns:
        month_cols = ["jan", "feb", "mar", "apr", "may", "jun",
                      "jul", "aug", "sep", "oct", "nov", "dec"]
        available = [c for c in month_cols if c in boysun_df.columns]
        if available:
            boysun_df["annual"] = boysun_df[available].mean(axis=1)
        else:
            raise ValueError("Annual harorat ustuni topilmadi!")

    d = boysun_df[["year", "annual"]].dropna().copy()
    d["year"] = d["year"].astype(int)
    d = d.sort_values("year").reset_index(drop=True)

    # 3σ tozalash
    mean_t = d["annual"].mean()
    std_t = d["annual"].std()
    d = d[(d["annual"] > mean_t - 3 * std_t) &
          (d["annual"] < mean_t + 3 * std_t)].copy()

    # Anomaliya (1961-1990)
    baseline = d[(d["year"] >= 1961) & (d["year"] <= 1990)]
    baseline_mean = baseline["annual"].mean() if len(baseline) >= 10 else d["annual"].mean()
    d["anomaly"] = d["annual"] - baseline_mean

    # 11-yr MA
    d["ma11"] = d["anomaly"].rolling(11, center=True, min_periods=6).mean()

    # Trend (1990+)
    recent = d[d["year"] >= 1990].dropna(subset=["anomaly"])
    if len(recent) >= 5:
        trend_coeff = np.polyfit(recent["year"], recent["anomaly"], 1)
        trend_per_decade = trend_coeff[0] * 10
    else:
        trend_coeff = [0, 0]
        trend_per_decade = 0

    print(f"   Davr: {d['year'].min()}–{d['year'].max()} | "
          f"Trend: +{trend_per_decade:.3f}°C/decade")
    return d, trend_coeff, trend_per_decade


# ══════════════════════════════════════════════════════════════
# FIGURE YARATISH
# ══════════════════════════════════════════════════════════════

def create_broken_axis_figure(paleo_data, boysun_data, trend_coeff, trend_per_decade):
    """Broken axis figure: 65% paleo + 35% instrumental."""

    print("\n🎨 Figure yaratilmoqda...")

    plt.style.use('default')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 1.0,
        'figure.dpi': 300,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig = plt.figure(figsize=(16, 6.5), facecolor='white')

    # 65% chap, 35% o'ng
    gs = gridspec.GridSpec(
        1, 2,
        width_ratios=[0.65, 0.35],
        wspace=0.06,
        left=0.06, right=0.96,
        top=0.90, bottom=0.12
    )

    ax_paleo = fig.add_subplot(gs[0, 0])
    ax_instr = fig.add_subplot(gs[0, 1])

    # ══════════════════════════════════════════════════════════
    # CHAP PANEL: PALEODAVR ΔT
    # ══════════════════════════════════════════════════════════

    age_ka = paleo_data["age_ka"].values
    dT_smooth = paleo_data["dT_smooth"].values
    dT_unc = paleo_data["dT_unc_combined"].values

    # Y limitsni oldin o'rnatamiz (±6°C)
    ax_paleo.set_ylim(-6, 6)
    # X limitsni o'rnatamiz (MIS 5 ko'rinishi uchun 130 ka gacha)
    ax_paleo.set_xlim(132, -1)  # Inverted: eski chap, yangi o'ng

    # MIS bosqichlari (fon shading, to'liq panel balandlikda)
    for mis in MIS_STAGES:
        ax_paleo.axvspan(
            mis['start'], mis['end'],
            color=mis['color'], alpha=0.5,
            zorder=0, linewidth=0
        )

    # DIB yorliqlari — pastroqda, kichik font
    for mis in MIS_STAGES:
        mid = (mis['start'] + mis['end']) / 2
        ax_paleo.text(
            mid, 0.03, mis['name'],
            ha='center', va='bottom', fontsize=7,
            color='#777777', fontweight='500',
            transform=ax_paleo.get_xaxis_transform()
        )

    # Hiatus zonalari — "H" belgisi yuqorida
    for h_start, h_end in HIATUS_PERIODS_KA:
        ax_paleo.axvspan(
            h_start, h_end,
            color=COLORS['hiatus'], alpha=0.6,
            zorder=1, linewidth=0
        )
        mid = (h_start + h_end) / 2
        ax_paleo.text(
            mid, 0.95, 'H',
            ha='center', va='top', fontsize=8,
            color='#999999', fontweight='bold',
            transform=ax_paleo.get_xaxis_transform()
        )

    # Nol chizig'i
    ax_paleo.axhline(
        y=0, color=COLORS['zero_line'],
        linewidth=0.8, linestyle='--', alpha=0.5, zorder=2
    )

    # Noaniqlik oralig'i (shading)
    ax_paleo.fill_between(
        age_ka,
        dT_smooth - dT_unc,
        dT_smooth + dT_unc,
        color=COLORS['uncertainty'], alpha=0.4,
        zorder=3, linewidth=0,
        label='Noaniqlik diapazoni'
    )

    # ΔT smoothed chiziq
    ax_paleo.plot(
        age_ka, dT_smooth,
        color=COLORS['deltaT_line'], linewidth=1.8,
        zorder=5, label='ΔT (2500 yillik silliqlangan)'
    )

    # ── Annotatsiyalar ───────────────────────────────────────
    # LGM (~21 ka)
    lgm_mask = (age_ka >= 19) & (age_ka <= 23)
    if lgm_mask.any():
        lgm_idx = np.argmin(dT_smooth[lgm_mask])
        lgm_age = age_ka[lgm_mask][lgm_idx]
        lgm_val = dT_smooth[lgm_mask][lgm_idx]
        ax_paleo.annotate(
            f'OGM\n{lgm_val:.1f}°C',
            xy=(lgm_age, lgm_val),
            xytext=(lgm_age + 10, lgm_val - 1.2),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#555555'),
            fontsize=8, ha='center', color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#AAAAAA', linewidth=0.6, alpha=0.9)
        )

    # MIS 5e (~125 ka)
    mis5e_mask = (age_ka >= 120) & (age_ka <= 130)
    if mis5e_mask.any():
        mis5e_idx = np.argmax(dT_smooth[mis5e_mask])
        mis5e_age = age_ka[mis5e_mask][mis5e_idx]
        mis5e_val = dT_smooth[mis5e_mask][mis5e_idx]
        ax_paleo.annotate(
            f'DIB 5e\n+{mis5e_val:.1f}°C',
            xy=(mis5e_age, mis5e_val),
            xytext=(mis5e_age - 12, mis5e_val + 1.2),
            arrowprops=dict(arrowstyle='->', lw=1.0, color='#555555'),
            fontsize=8, ha='center', color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#AAAAAA', linewidth=0.6, alpha=0.9)
        )

    # Holocene Optimum — strelkasiz, Y o'qi yonida
    hol_mask = (age_ka >= 8) & (age_ka <= 12)
    if hol_mask.any():
        hol_idx = np.argmax(dT_smooth[hol_mask])
        hol_val = dT_smooth[hol_mask][hol_idx]
        ax_paleo.text(
            0.98, (hol_val - (-6)) / 12,
            'Golosen\noptimumi',
            ha='right', va='center', fontsize=7.5,
            color='#555555', style='italic',
            transform=ax_paleo.transAxes
        )

    # ── Chap panel formatting ────────────────────────────────
    ax_paleo.set_xlabel('Yosh (ming yil oldin)', fontsize=10, fontweight='bold')
    ax_paleo.set_ylabel('ΔT (°C, zamonaviyga nisbatan)', fontsize=10, fontweight='bold')
    ax_paleo.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
    ax_paleo.legend(
        loc='lower left', fontsize=8, frameon=True,
        framealpha=0.9, edgecolor='#CCCCCC', fancybox=False
    )

    # Panel label
    ax_paleo.text(
        0.02, 0.92, 'a)',
        transform=ax_paleo.transAxes, fontsize=12,
        fontweight='bold', va='top', ha='left',
        color='#222222'
    )

    ax_paleo.spines['right'].set_visible(False)
    ax_paleo.tick_params(axis='y', right=False)

    # ══════════════════════════════════════════════════════════
    # O'NG PANEL: INSTRUMENTAL (BOYSUN MS)
    # ══════════════════════════════════════════════════════════

    years = boysun_data["year"].values
    anomaly = boysun_data["anomaly"].values
    moving_avg = boysun_data["ma11"].values

    # Bar chart
    bar_colors = [COLORS['warm_bar'] if a > 0 else COLORS['cold_bar']
                  for a in anomaly]
    ax_instr.bar(
        years, anomaly,
        color=bar_colors, width=0.8,
        alpha=0.6, zorder=2, edgecolor='none'
    )

    # Nol chizig'i
    ax_instr.axhline(y=0, color=COLORS['zero_line'],
                     linewidth=0.8, linestyle='--', alpha=0.5, zorder=1)

    # 11 yillik harakatlanuvchi o'rtacha
    ax_instr.plot(
        years, moving_avg,
        color=COLORS['smooth_line'], linewidth=2.0,
        zorder=5, label='11 yillik harakatlanuvchi o\'rtacha'
    )

    # Trend (1990+)
    trend_years = np.arange(1990, years.max() + 1)
    trend_line_vals = np.polyval(trend_coeff, trend_years)
    ax_instr.plot(
        trend_years, trend_line_vals,
        color=COLORS['trend_line'], linewidth=1.3,
        linestyle=':', zorder=4,
        label=f'Trend (+{trend_per_decade:.2f}°C/o\'n yillik)'
    )

    # TON-2 nuqtalar: 1941 va 2008
    for target_year in [1941, 2008]:
        mask = boysun_data["year"] == target_year
        if mask.any():
            val = boysun_data.loc[mask, "anomaly"].values[0]
            ax_instr.plot(
                target_year, val,
                marker='D', color=COLORS['ton2_marker'],
                ms=9, zorder=6,
                markeredgecolor='white', markeredgewidth=1.2
            )
            # Annotatsiya
            y_off = -0.6 if target_year == 1941 else 0.6
            x_off = 6 if target_year == 1941 else -6
            ax_instr.annotate(
                f'TON-2\n({target_year})',
                xy=(target_year, val),
                xytext=(target_year + x_off, val + y_off),
                arrowprops=dict(arrowstyle='->', lw=0.8,
                                color=COLORS['ton2_marker']),
                fontsize=7, fontweight='bold',
                color=COLORS['ton2_marker'], ha='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=COLORS['ton2_marker'],
                          linewidth=0.6, alpha=0.9)
            )

    # Baseline shading
    ax_instr.axvspan(1961, 1990, color='#F0F0F0', alpha=0.4, zorder=0)

    # O'ng panel formatting
    ax_instr.set_xlim(years.min() - 2, years.max() + 2)
    ax_instr.set_ylim(-2, 2)
    ax_instr.set_xlabel('Yil (milodiy)', fontsize=10, fontweight='bold')
    ax_instr.set_ylabel('Harorat anomaliyasi (°C)', fontsize=10, fontweight='bold')
    ax_instr.yaxis.set_label_position('right')
    ax_instr.yaxis.tick_right()
    ax_instr.grid(True, alpha=0.15, linestyle=':', linewidth=0.5)
    ax_instr.legend(
        loc='upper left', fontsize=7.5, frameon=True,
        framealpha=0.9, edgecolor='#CCCCCC', fancybox=False
    )

    ax_instr.text(
        0.03, 0.92, 'b)',
        transform=ax_instr.transAxes, fontsize=12,
        fontweight='bold', va='top', ha='left',
        color='#222222'
    )

    ax_instr.spines['left'].set_visible(False)
    ax_instr.tick_params(axis='y', left=False)

    # ══════════════════════════════════════════════════════════
    # BROKEN AXIS — ingichka diagonal chiziqlar
    # ══════════════════════════════════════════════════════════

    fig.canvas.draw()
    paleo_bbox = ax_paleo.get_position()
    instr_bbox = ax_instr.get_position()
    x_break = (paleo_bbox.x1 + instr_bbox.x0) / 2

    d_size = 0.012
    kwargs = dict(
        transform=fig.transFigure, color='#555555',
        linewidth=1.2, clip_on=False, zorder=10
    )

    for y_frac in [0.25, 0.50, 0.75]:
        y_pos = paleo_bbox.y0 + (paleo_bbox.y1 - paleo_bbox.y0) * y_frac
        # Birinchi chiziq
        fig.lines.append(plt.Line2D(
            [x_break - d_size, x_break + d_size],
            [y_pos - d_size, y_pos + d_size], **kwargs
        ))
        # Ikkinchi chiziq (parallel)
        fig.lines.append(plt.Line2D(
            [x_break - d_size + 0.004, x_break + d_size + 0.004],
            [y_pos - d_size, y_pos + d_size], **kwargs
        ))

    return fig


# ══════════════════════════════════════════════════════════════
# ASOSIY DASTUR
# ══════════════════════════════════════════════════════════════

def main():
    """Asosiy funksiya"""
    print("=" * 60)
    print("1-RASM: BROKEN AXIS ΔT")
    print("  O'zbekiston ilmiy jurnallari uslubida")
    print("=" * 60)

    paleo_data = load_ton2_and_compute_deltaT()
    boysun_data, trend_coeff, trend_per_decade = load_boysun_instrumental()

    fig = create_broken_axis_figure(
        paleo_data, boysun_data, trend_coeff, trend_per_decade
    )

    print("\n💾 Saqlash...")
    fig.savefig('fig3_broken_axis_deltaT.png',
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig('fig3_broken_axis_deltaT.pdf',
                dpi=300, bbox_inches='tight', facecolor='white')
    print("   ✅ fig3_broken_axis_deltaT.png")
    print("   ✅ fig3_broken_axis_deltaT.pdf")
    plt.close(fig)

    print("\n✨ TAYYOR!")


if __name__ == "__main__":
    main()

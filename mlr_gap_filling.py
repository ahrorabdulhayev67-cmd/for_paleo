"""
Boysun MS yog'ingarchilik bo'shliqlarini to'ldirish
Denov + Mingchuqur stansiyalari yordamida (MLR)
═══════════════════════════════════════════════════

Usul: P_Boysun = β₀ + β₁×P_Denov + β₂×P_Mingchuqur

pip install numpy pandas matplotlib scipy openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR — SHU YERDA O'ZGARTIRING
# ══════════════════════════════════════════════════════════════

BOYSUN_FILE = 'boysun_precipitation_clean.csv'
DENOV_MINGCHUQUR_FILE = 'Denov_Mingchuqur.xlsx'
DENOV_SHEET = None          # None = avtomatik topadi
MINGCHUQUR_SHEET = None     # None = avtomatik topadi
GAP_START = 1977
GAP_END = 1984

# Oy ustunlari — Boysun CSV dagi haqiqiy nomlar asosida aniqlanadi
# Agar CSV da I, II, III... bo'lsa yoki 1, 2, 3... bo'lsa — avtomatik topadi
MONTH_COLS = None  # run() ichida avtomatik aniqlanadi

# ══════════════════════════════════════════════════════════════
# FUNKSIYALAR
# ══════════════════════════════════════════════════════════════

def load_station(filepath, sheet_name, month_cols, name=''):
    """Excel dan stansiya ma'lumotlarini yuklash."""
    df_raw = pd.read_excel(filepath, sheet_name=sheet_name)
    cols = list(df_raw.columns)
    
    df = pd.DataFrame()
    df['year'] = pd.to_numeric(df_raw[cols[0]], errors='coerce').astype('Int64')
    
    # Birinchi ustun = yil, keyingi 12 ta = oylar
    for i in range(min(12, len(cols) - 1)):
        col_name = month_cols[i] if i < len(month_cols) else f"m{i+1}"
        df[col_name] = pd.to_numeric(df_raw[cols[i + 1]], errors='coerce')
        df.loc[df[col_name] < 0, col_name] = 0
    
    available_months = [c for c in month_cols if c in df.columns]
    df['annual'] = df[available_months].sum(axis=1, min_count=10)
    df = df.dropna(subset=['year']).sort_values('year').reset_index(drop=True)
    df['year'] = df['year'].astype(int)
    
    print(f"  ✅ {name}: {len(df)} yil ({df['year'].min()}-{df['year'].max()})")
    return df


def find_sheets(filepath):
    """Sheet nomlarini avtomatik topish."""
    xl = pd.ExcelFile(filepath)
    sheets = xl.sheet_names
    xl.close()
    
    denov_sheet = None
    mingchuqur_sheet = None
    
    for s in sheets:
        sl = s.lower()
        if 'денов' in sl or 'denov' in sl:
            denov_sheet = s
        elif 'минг' in sl or 'ming' in sl or 'мингчуқур' in sl:
            mingchuqur_sheet = s
    
    if denov_sheet is None and len(sheets) >= 1:
        denov_sheet = sheets[0]
    if mingchuqur_sheet is None and len(sheets) >= 2:
        mingchuqur_sheet = sheets[1]
    
    return denov_sheet, mingchuqur_sheet, sheets


def run():
    """To'liq jarayon: yuklash → MLR → to'ldirish → validatsiya → grafik."""
    
    global MONTH_COLS
    
    print("=" * 70)
    print("  MLR BO'SHLIQ TO'LDIRISH")
    print("  Boysun MS ← Denov + Mingchuqur")
    print("=" * 70)
    print()
    
    # ──────────────────────────────────────────
    # 1. MA'LUMOTLARNI YUKLASH
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  1. MA'LUMOTLARNI YUKLASH")
    print("━" * 70)
    print()
    
    # Sheet nomlarini topish
    global DENOV_SHEET, MINGCHUQUR_SHEET
    denov_sh, ming_sh, all_sheets = find_sheets(DENOV_MINGCHUQUR_FILE)
    
    if DENOV_SHEET is None:
        DENOV_SHEET = denov_sh
    if MINGCHUQUR_SHEET is None:
        MINGCHUQUR_SHEET = ming_sh
    
    print(f"  Fayllar: {DENOV_MINGCHUQUR_FILE}")
    print(f"  Mavjud sheetlar: {all_sheets}")
    print(f"  Denov sheet: '{DENOV_SHEET}'")
    print(f"  Mingchuqur sheet: '{MINGCHUQUR_SHEET}'")
    print()
    
    # Boysun yuklash va oy ustunlarini aniqlash
    df_boysun = pd.read_csv(BOYSUN_FILE)
    print(f"  ✅ Boysun: {len(df_boysun)} yil ({df_boysun['year'].min()}-{df_boysun['year'].max()})")
    print(f"  Ustunlar: {list(df_boysun.columns)}")
    
    # Oy ustunlarini avtomatik aniqlash
    skip_cols = ['year', 'annual', 'data_quality', 'annual_calc']
    MONTH_COLS = [c for c in df_boysun.columns if c not in skip_cols]
    
    # Faqat birinchi 12 tasini olish (oylar)
    if len(MONTH_COLS) > 12:
        MONTH_COLS = MONTH_COLS[:12]
    
    print(f"  Oy ustunlari ({len(MONTH_COLS)}): {MONTH_COLS}")
    print()
    
    df_denov = load_station(DENOV_MINGCHUQUR_FILE, DENOV_SHEET, MONTH_COLS, 'Denov')
    df_mingchuqur = load_station(DENOV_MINGCHUQUR_FILE, MINGCHUQUR_SHEET, MONTH_COLS, 'Mingchuqur')
    print()
    
    # ──────────────────────────────────────────
    # 2. MLR KOEFFITSIENTLARNI HISOBLASH
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  2. MLR KOEFFITSIENTLAR (har bir oy uchun)")
    print("━" * 70)
    print()
    
    gap_years = set(range(GAP_START, GAP_END + 1))
    boysun_yrs = set(df_boysun.dropna(subset=['annual'])['year'])
    denov_yrs = set(df_denov.dropna(subset=['annual'])['year'])
    ming_yrs = set(df_mingchuqur.dropna(subset=['annual'])['year'])
    common_years = sorted((boysun_yrs & denov_yrs & ming_yrs) - gap_years)
    
    print(f"  Umumiy kuzatilgan yillar: {len(common_years)} ({min(common_years)}-{max(common_years)})")
    print()
    
    print(f"  {'Oy':<5} {'R²':<7} {'R²adj':<7} {'RMSE':<7} {'β₀':<8} {'β₁(Den)':<9} {'β₂(Ming)':<9} {'n':<4}")
    print(f"  {'─'*5} {'─'*7} {'─'*7} {'─'*7} {'─'*8} {'─'*9} {'─'*9} {'─'*4}")
    
    mlr_params = {}
    
    for col in MONTH_COLS:
        y_list, x1_list, x2_list = [], [], []
        
        for yr in common_years:
            b = df_boysun[df_boysun['year'] == yr][col].values
            d = df_denov[df_denov['year'] == yr][col].values
            m = df_mingchuqur[df_mingchuqur['year'] == yr][col].values
            
            if len(b) and len(d) and len(m):
                bv, dv, mv = b[0], d[0], m[0]
                if pd.notna(bv) and pd.notna(dv) and pd.notna(mv):
                    y_list.append(bv)
                    x1_list.append(dv)
                    x2_list.append(mv)
        
        y = np.array(y_list)
        n = len(y)
        
        if n < 10:
            mlr_params[col] = None
            print(f"  {col:<5} ⚠️  kam ma'lumot ({n})")
            continue
        
        X = np.column_stack([np.ones(n), x1_list, x2_list])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        r_sq_adj = 1 - (1 - r_sq) * (n - 1) / (n - 3)
        rmse = np.sqrt(ss_res / n)
        
        mlr_params[col] = {'b0': beta[0], 'b1': beta[1], 'b2': beta[2],
                           'r_sq': r_sq, 'rmse': rmse}
        
        print(f"  {col:<5} {r_sq:<7.3f} {r_sq_adj:<7.3f} {rmse:<7.1f} "
              f"{beta[0]:<8.2f} {beta[1]:<9.3f} {beta[2]:<9.3f} {n:<4}")
    
    print()
    
    # ──────────────────────────────────────────
    # 3. BO'SHLIQNI TO'LDIRISH
    # ──────────────────────────────────────────
    print("━" * 70)
    print(f"  3. BO'SHLIQ TO'LDIRISH ({GAP_START}-{GAP_END})")
    print("━" * 70)
    print()
    
    filled_rows = []
    
    print(f"  {'Yil':<5}", end='')
    for col in MONTH_COLS:
        print(f" {col:<5}", end='')
    print(f"  {'Yillik'}")
    print(f"  {'─'*5}", end='')
    for _ in MONTH_COLS:
        print(f" {'─'*5}", end='')
    print(f"  {'─'*7}")
    
    for year in range(GAP_START, GAP_END + 1):
        row = {'year': year}
        d_row = df_denov[df_denov['year'] == year]
        m_row = df_mingchuqur[df_mingchuqur['year'] == year]
        
        for col in MONTH_COLS:
            p = mlr_params.get(col)
            if p is None:
                row[col] = np.nan
                continue
            
            dv = d_row[col].values[0] if len(d_row) > 0 else np.nan
            mv = m_row[col].values[0] if len(m_row) > 0 else np.nan
            
            if pd.notna(dv) and pd.notna(mv):
                pred = p['b0'] + p['b1'] * dv + p['b2'] * mv
            elif pd.notna(dv):
                pred = p['b0'] + p['b1'] * dv
            elif pd.notna(mv):
                pred = p['b0'] + p['b2'] * mv
            else:
                row[col] = np.nan
                continue
            
            row[col] = max(0, round(pred, 1))
        
        vals = [row[c] for c in MONTH_COLS if pd.notna(row.get(c))]
        row['annual'] = round(sum(vals), 1) if vals else np.nan
        row['data_quality'] = 'infilled_mlr'
        filled_rows.append(row)
        
        print(f"  {year:<5}", end='')
        for col in MONTH_COLS:
            v = row[col]
            print(f" {v:<5.1f}" if pd.notna(v) else f" {'--':<5}", end='')
        print(f"  {row['annual']:.1f}" if pd.notna(row['annual']) else "  --")
    
    print()
    
    # Birlashtirish
    df_filled = df_boysun.copy()
    if 'data_quality' not in df_filled.columns:
        df_filled['data_quality'] = 'observed'
    df_filled = df_filled[~df_filled['year'].isin(range(GAP_START, GAP_END + 1))]
    df_filled = pd.concat([df_filled, pd.DataFrame(filled_rows)], ignore_index=True)
    df_filled = df_filled.sort_values('year').reset_index(drop=True)
    
    print(f"  ✅ {len(filled_rows)} yil to'ldirildi")
    print()
    
    # ──────────────────────────────────────────
    # 4. CROSS-VALIDATION
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  4. CROSS-VALIDATION")
    print("━" * 70)
    print()
    
    np.random.seed(42)
    n_test = min(8, len(common_years) // 3)
    test_years = sorted(np.random.choice(common_years, size=n_test, replace=False))
    print(f"  Test yillari: {list(test_years)}")
    
    errors = []
    for yr in test_years:
        obs = df_boysun[df_boysun['year'] == yr]['annual'].values[0]
        d_row = df_denov[df_denov['year'] == yr]
        m_row = df_mingchuqur[df_mingchuqur['year'] == yr]
        
        pred = 0
        for col in MONTH_COLS:
            p = mlr_params.get(col)
            if p is None:
                continue
            dv = d_row[col].values[0] if len(d_row) > 0 else 0
            mv = m_row[col].values[0] if len(m_row) > 0 else 0
            pred += max(0, p['b0'] + p['b1'] * dv + p['b2'] * mv)
        
        if pd.notna(obs):
            errors.append(pred - obs)
    
    errors = np.array(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    obs_mean = df_boysun['annual'].dropna().mean()
    
    print(f"\n  MAE:  {mae:.1f} mm")
    print(f"  RMSE: {rmse:.1f} mm")
    print(f"  Bias: {bias:+.1f} mm")
    print(f"  Nisbiy xatolik: {mae/obs_mean*100:.1f}%")
    print()
    
    # ──────────────────────────────────────────
    # 5. GRAFIK
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  5. GRAFIK")
    print("━" * 70)
    print()
    
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(14, 5))
    
    obs = df_filled[df_filled['data_quality'] == 'observed']
    inf = df_filled[df_filled['data_quality'] == 'infilled_mlr']
    mean_ann = obs['annual'].mean()
    
    ax.bar(obs['year'], obs['annual'], color='#457B9D', alpha=0.7, width=0.8, label='Kuzatilgan')
    ax.bar(inf['year'], inf['annual'], color='#E07060', alpha=0.8, width=0.8,
           label=f"MLR ({GAP_START}-{GAP_END})", edgecolor='#CC0000', linewidth=0.5)
    ax.axhline(y=mean_ann, color='#1a1a1a', ls='--', lw=1.2,
               label=f"O'rtacha: {mean_ann:.0f} mm")
    ax.axvspan(GAP_START-0.5, GAP_END+0.5, color='#FFEEEE', alpha=0.3, zorder=0)
    
    all_ann = df_filled.set_index('year')['annual'].dropna()
    ma = all_ann.rolling(11, center=True, min_periods=6).mean()
    ax.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0, label="11 yillik o'rtacha")
    
    ax.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax.set_ylabel("Yillik yog'ingarchilik (mm)", fontsize=11, fontweight='bold')
    ax.set_title("Boysun MS — MLR (Denov + Mingchuqur) bilan to'ldirilgan",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(1934, 2026)
    ax.text(0.02, 0.95, f"MAE={mae:.1f} mm\nRMSE={rmse:.1f} mm",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CCC', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('mlr_gap_filling_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlr_gap_filling_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ──────────────────────────────────────────
    # 6. SAQLASH
    # ──────────────────────────────────────────
    output = 'boysun_precipitation_filled_mlr.csv'
    df_filled.to_csv(output, index=False)
    
    print(f"  💾 {output}")
    print(f"  💾 mlr_gap_filling_results.png")
    print()
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    
    return df_filled


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    df = run()

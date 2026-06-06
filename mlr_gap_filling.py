"""
Boysun MS yog'ingarchilik bo'shliqlarini to'ldirish
Denov + Mingchuqur YILLIK ma'lumotlari yordamida
═══════════════════════════════════════════════════

Ma'lumotlar tuzilishi:
  Denov/Mingchuqur: yil | harorat (°C) | yog'in (mm/yil)
  Boysun: yil | jan | feb | ... | dec | annual

Usul: Chiziqli regressiya (yillik darajada)
  P_Boysun_annual = β₀ + β₁×P_Denov + β₂×P_Mingchuqur

Keyin: oylik taqsimlash (klimatologik nisbatlar asosida)

pip install numpy pandas matplotlib scipy openpyxl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# SOZLAMALAR
# ══════════════════════════════════════════════════════════════

BOYSUN_FILE = 'boysun_precipitation_clean.csv'
DENOV_MINGCHUQUR_FILE = 'Denov_Mingchuqur.xlsx'
GAP_START = 1977
GAP_END = 1984

# ══════════════════════════════════════════════════════════════
# ASOSIY KOD
# ══════════════════════════════════════════════════════════════

def run():
    """To'liq jarayon."""
    
    print("=" * 70)
    print("  YILLIK REGRESSIYA BILAN BO'SHLIQ TO'LDIRISH")
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
    
    # Boysun
    df_boysun = pd.read_csv(BOYSUN_FILE)
    month_cols = [c for c in df_boysun.columns 
                  if c not in ['year', 'annual', 'data_quality', 'annual_calc']]
    print(f"  ✅ Boysun: {len(df_boysun)} yil ({df_boysun['year'].min()}-{df_boysun['year'].max()})")
    print(f"     Oy ustunlari: {month_cols}")
    
    # Denov
    df_denov_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Денов')
    cols_d = list(df_denov_raw.columns)
    df_denov = pd.DataFrame()
    df_denov['year'] = pd.to_numeric(df_denov_raw[cols_d[0]], errors='coerce').astype('Int64')
    df_denov['temp'] = pd.to_numeric(df_denov_raw[cols_d[1]], errors='coerce')
    df_denov['precip'] = pd.to_numeric(df_denov_raw[cols_d[2]], errors='coerce')
    df_denov = df_denov.dropna(subset=['year']).reset_index(drop=True)
    df_denov['year'] = df_denov['year'].astype(int)
    print(f"  ✅ Denov: {len(df_denov)} yil ({df_denov['year'].min()}-{df_denov['year'].max()})")
    print(f"     Harorat: {df_denov['temp'].notna().sum()} yil mavjud")
    print(f"     Yog'in:  {df_denov['precip'].notna().sum()} yil mavjud")
    
    # Mingchuqur
    df_ming_raw = pd.read_excel(DENOV_MINGCHUQUR_FILE, sheet_name='Мингчуқур')
    cols_m = list(df_ming_raw.columns)
    df_ming = pd.DataFrame()
    df_ming['year'] = pd.to_numeric(df_ming_raw[cols_m[0]], errors='coerce').astype('Int64')
    df_ming['temp'] = pd.to_numeric(df_ming_raw[cols_m[1]], errors='coerce')
    df_ming['precip'] = pd.to_numeric(df_ming_raw[cols_m[2]], errors='coerce')
    df_ming = df_ming.dropna(subset=['year']).reset_index(drop=True)
    df_ming['year'] = df_ming['year'].astype(int)
    print(f"  ✅ Mingchuqur: {len(df_ming)} yil ({df_ming['year'].min()}-{df_ming['year'].max()})")
    print(f"     Harorat: {df_ming['temp'].notna().sum()} yil mavjud")
    print(f"     Yog'in:  {df_ming['precip'].notna().sum()} yil mavjud")
    print()
    
    # ──────────────────────────────────────────
    # 2. REGRESSIYA MODELI QURISH
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  2. REGRESSIYA MODELI")
    print("━" * 70)
    print()
    
    gap_years = set(range(GAP_START, GAP_END + 1))
    
    # Umumiy yillar topish — bo'shliqsiz, barcha stansiyada yillik yog'in bor
    boysun_annual = df_boysun.dropna(subset=['annual']).set_index('year')['annual']
    denov_precip = df_denov.dropna(subset=['precip']).set_index('year')['precip']
    ming_precip = df_ming.dropna(subset=['precip']).set_index('year')['precip']
    
    # Qaysi kombinatsiyalar mavjud?
    common_all = sorted(
        (set(boysun_annual.index) & set(denov_precip.index) & set(ming_precip.index)) - gap_years
    )
    common_denov = sorted(
        (set(boysun_annual.index) & set(denov_precip.index)) - gap_years
    )
    common_ming = sorted(
        (set(boysun_annual.index) & set(ming_precip.index)) - gap_years
    )
    
    print(f"  Boysun + Denov + Mingchuqur: {len(common_all)} umumiy yil")
    print(f"  Boysun + Denov:              {len(common_denov)} umumiy yil")
    print(f"  Boysun + Mingchuqur:         {len(common_ming)} umumiy yil")
    print()
    
    # Eng yaxshi modelni tanlash
    best_model = None
    best_r2 = -1
    
    # Model A: Ikki stansiya (agar yetarli ma'lumot bo'lsa)
    if len(common_all) >= 10:
        y = np.array([boysun_annual[yr] for yr in common_all])
        x1 = np.array([denov_precip[yr] for yr in common_all])
        x2 = np.array([ming_precip[yr] for yr in common_all])
        X = np.column_stack([np.ones(len(y)), x1, x2])
        
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ beta
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - y.mean())**2)
        r2 = 1 - ss_res / ss_tot
        rmse = np.sqrt(ss_res / len(y))
        
        print(f"  Model A (Denov + Mingchuqur): R²={r2:.3f}, RMSE={rmse:.1f} mm")
        print(f"    P_Boysun = {beta[0]:.1f} + {beta[1]:.3f}×P_Denov + {beta[2]:.3f}×P_Mingchuqur")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = {'type': 'both', 'beta': beta, 'r2': r2, 'rmse': rmse,
                          'years': common_all}
    
    # Model B: Faqat Denov
    if len(common_denov) >= 10:
        y = np.array([boysun_annual[yr] for yr in common_denov])
        x1 = np.array([denov_precip[yr] for yr in common_denov])
        slope, intercept, r, p, se = stats.linregress(x1, y)
        r2 = r**2
        rmse = np.sqrt(np.mean((y - (slope*x1 + intercept))**2))
        
        print(f"  Model B (faqat Denov):        R²={r2:.3f}, RMSE={rmse:.1f} mm")
        print(f"    P_Boysun = {intercept:.1f} + {slope:.3f}×P_Denov")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = {'type': 'denov', 'slope': slope, 'intercept': intercept,
                          'r2': r2, 'rmse': rmse, 'years': common_denov}
    
    # Model C: Faqat Mingchuqur
    if len(common_ming) >= 10:
        y = np.array([boysun_annual[yr] for yr in common_ming])
        x2 = np.array([ming_precip[yr] for yr in common_ming])
        slope, intercept, r, p, se = stats.linregress(x2, y)
        r2 = r**2
        rmse = np.sqrt(np.mean((y - (slope*x2 + intercept))**2))
        
        print(f"  Model C (faqat Mingchuqur):   R²={r2:.3f}, RMSE={rmse:.1f} mm")
        print(f"    P_Boysun = {intercept:.1f} + {slope:.3f}×P_Mingchuqur")
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = {'type': 'ming', 'slope': slope, 'intercept': intercept,
                          'r2': r2, 'rmse': rmse, 'years': common_ming}
    
    if best_model is None:
        print("\n  ❌ Yetarli ma'lumot yo'q! Regressiya qurib bo'lmadi.")
        return None
    
    print(f"\n  🏆 ENG YAXSHI MODEL: {best_model['type'].upper()} (R²={best_model['r2']:.3f})")
    print()
    
    # ──────────────────────────────────────────
    # 3. BO'SHLIQ TO'LDIRISH (YILLIK)
    # ──────────────────────────────────────────
    print("━" * 70)
    print(f"  3. BO'SHLIQ TO'LDIRISH ({GAP_START}-{GAP_END})")
    print("━" * 70)
    print()
    
    # Oylik klimatologik nisbatlar (yillik jamidan oylik ulush)
    obs_only = df_boysun[df_boysun['annual'].notna() & (df_boysun['annual'] > 0)]
    obs_only = obs_only[~obs_only['year'].isin(gap_years)]
    
    monthly_fractions = {}
    for col in month_cols:
        frac = (obs_only[col] / obs_only['annual']).mean()
        monthly_fractions[col] = frac
    
    # Normalizatsiya (jami = 1 bo'lishi uchun)
    total_frac = sum(monthly_fractions.values())
    for col in month_cols:
        monthly_fractions[col] /= total_frac
    
    print("  Oylik klimatologik ulushlar:")
    for col in month_cols:
        print(f"    {col}: {monthly_fractions[col]*100:.1f}%", end='')
    print("\n")
    
    # Yillik bashorat
    filled_rows = []
    print(f"  {'Yil':<6} {'Denov':<8} {'Ming':<8} {'Bashorat':<10} {'Manba'}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*10} {'─'*15}")
    
    for year in range(GAP_START, GAP_END + 1):
        d_val = denov_precip.get(year, np.nan)
        m_val = ming_precip.get(year, np.nan)
        
        # Bashorat qilish
        predicted = np.nan
        source = ''
        
        if best_model['type'] == 'both' and pd.notna(d_val) and pd.notna(m_val):
            b = best_model['beta']
            predicted = b[0] + b[1] * d_val + b[2] * m_val
            source = 'Denov+Ming'
        elif best_model['type'] == 'denov' and pd.notna(d_val):
            predicted = best_model['intercept'] + best_model['slope'] * d_val
            source = 'Denov'
        elif best_model['type'] == 'ming' and pd.notna(m_val):
            predicted = best_model['intercept'] + best_model['slope'] * m_val
            source = 'Mingchuqur'
        elif pd.notna(d_val):
            # Fallback: faqat Denov bilan oddiy regressiya
            if len(common_denov) >= 5:
                y_t = np.array([boysun_annual[yr] for yr in common_denov])
                x_t = np.array([denov_precip[yr] for yr in common_denov])
                sl, ic, _, _, _ = stats.linregress(x_t, y_t)
                predicted = ic + sl * d_val
                source = 'Denov(fallback)'
        elif pd.notna(m_val):
            if len(common_ming) >= 5:
                y_t = np.array([boysun_annual[yr] for yr in common_ming])
                x_t = np.array([ming_precip[yr] for yr in common_ming])
                sl, ic, _, _, _ = stats.linregress(x_t, y_t)
                predicted = ic + sl * m_val
                source = 'Ming(fallback)'
        
        predicted = max(0, predicted) if pd.notna(predicted) else np.nan
        
        # Oylik taqsimlash
        row = {'year': year, 'data_quality': 'infilled_mlr'}
        if pd.notna(predicted):
            for col in month_cols:
                row[col] = round(predicted * monthly_fractions[col], 1)
            row['annual'] = round(predicted, 1)
        else:
            for col in month_cols:
                row[col] = np.nan
            row['annual'] = np.nan
        
        filled_rows.append(row)
        
        d_str = f"{d_val:.0f}" if pd.notna(d_val) else "--"
        m_str = f"{m_val:.0f}" if pd.notna(m_val) else "--"
        p_str = f"{predicted:.1f}" if pd.notna(predicted) else "--"
        print(f"  {year:<6} {d_str:<8} {m_str:<8} {p_str:<10} {source}")
    
    print()
    
    # Birlashtirish
    df_filled = df_boysun.copy()
    if 'data_quality' not in df_filled.columns:
        df_filled['data_quality'] = 'observed'
    df_filled = df_filled[~df_filled['year'].isin(range(GAP_START, GAP_END + 1))]
    df_filled = pd.concat([df_filled, pd.DataFrame(filled_rows)], ignore_index=True)
    df_filled = df_filled.sort_values('year').reset_index(drop=True)
    
    print(f"  ✅ {sum(1 for r in filled_rows if pd.notna(r['annual']))} yil to'ldirildi")
    print()
    
    # ──────────────────────────────────────────
    # 4. CROSS-VALIDATION
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  4. CROSS-VALIDATION")
    print("━" * 70)
    print()
    
    np.random.seed(42)
    cv_years = best_model['years']
    n_test = min(8, len(cv_years) // 3)
    test_years = sorted(np.random.choice(cv_years, size=n_test, replace=False))
    
    print(f"  Test yillari: {list(test_years)}")
    
    errors = []
    for yr in test_years:
        obs = boysun_annual[yr]
        d_val = denov_precip.get(yr, np.nan)
        m_val = ming_precip.get(yr, np.nan)
        
        if best_model['type'] == 'both' and pd.notna(d_val) and pd.notna(m_val):
            b = best_model['beta']
            pred = b[0] + b[1] * d_val + b[2] * m_val
        elif best_model['type'] == 'denov' and pd.notna(d_val):
            pred = best_model['intercept'] + best_model['slope'] * d_val
        elif best_model['type'] == 'ming' and pd.notna(m_val):
            pred = best_model['intercept'] + best_model['slope'] * m_val
        else:
            continue
        
        errors.append(pred - obs)
    
    errors = np.array(errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    obs_mean = boysun_annual.mean()
    
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
    
    obs_df = df_filled[df_filled['data_quality'] == 'observed']
    inf_df = df_filled[df_filled['data_quality'] == 'infilled_mlr']
    mean_ann = obs_df['annual'].mean()
    
    ax.bar(obs_df['year'], obs_df['annual'], color='#457B9D', alpha=0.7, 
           width=0.8, label='Kuzatilgan')
    ax.bar(inf_df['year'], inf_df['annual'], color='#E07060', alpha=0.8, 
           width=0.8, label=f"MLR to'ldirilgan ({GAP_START}-{GAP_END})",
           edgecolor='#CC0000', linewidth=0.5)
    ax.axhline(y=mean_ann, color='#1a1a1a', ls='--', lw=1.2,
               label=f"O'rtacha: {mean_ann:.0f} mm")
    ax.axvspan(GAP_START-0.5, GAP_END+0.5, color='#FFEEEE', alpha=0.3, zorder=0)
    
    # 11 yillik o'rtacha
    all_ann = df_filled.set_index('year')['annual'].dropna()
    ma = all_ann.rolling(11, center=True, min_periods=6).mean()
    ax.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0, label="11 yillik o'rtacha")
    
    ax.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax.set_ylabel("Yillik yog'ingarchilik (mm)", fontsize=11, fontweight='bold')
    ax.set_title(f"Boysun MS — {best_model['type'].upper()} regressiya bilan to'ldirilgan",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(1934, 2026)
    ax.text(0.02, 0.95,
            f"R² = {best_model['r2']:.3f}\nMAE = {mae:.1f} mm\nRMSE = {rmse:.1f} mm",
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

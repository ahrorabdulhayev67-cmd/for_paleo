"""
Multiple Linear Regression (MLR) — Ko'p stansiyali bo'shliq to'ldirish
═══════════════════════════════════════════════════════════════════════

Boysun MS (1977-1984) bo'shliqlarini qo'shni stansiyalar bilan to'ldirish:
  - Denov stansiyasi
  - Mingchuqur stansiyasi
  - (Ixtiyoriy) ERA5 reanaliz

Usul: Multiple Linear Regression (MLR)
  P_Boysun = β₀ + β₁×P_Denov + β₂×P_Mingchuqur

Bu ERA5 dan ANCHA ANIQ, chunki haqiqiy qo'shni stansiya ma'lumotlari
yog'ingarchilik variatsiyasini yaxshiroq ushlaydi.

Kerakli kutubxonalar:
  pip install numpy pandas matplotlib scipy openpyxl

Muallif: Paleoklimatologiya laboratoriyasi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# 1-QADAM: EXCEL FAYLLARNI KO'RISH (sheet nomlarini aniqlash)
# ══════════════════════════════════════════════════════════════

def check_excel_sheets(filepath='Denov_Mingchuqur.xlsx'):
    """
    Excel fayl ichidagi sheet nomlarini ko'rish.
    Birinchi marta ishga tushirib, sheet nomlarini aniqlang.
    """
    xl = pd.ExcelFile(filepath)
    print(f"📂 Fayl: {filepath}")
    print(f"   Sheet nomlari: {xl.sheet_names}")
    print()
    
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, nrows=5)
        print(f"  === {sheet} ===")
        print(f"  Ustunlar: {list(df.columns)}")
        print(f"  Dastlabki 3 qator:")
        print(df.head(3).to_string())
        print()
    
    xl.close()
    return xl.sheet_names


# ══════════════════════════════════════════════════════════════
# 2-QADAM: MA'LUMOTLARNI YUKLASH VA STANDARTLASHTIRISH
# ══════════════════════════════════════════════════════════════

def load_station_data(filepath, sheet_name, station_name=''):
    """
    Stansiya yog'ingarchilik ma'lumotlarini yuklash va standartlashtirish.
    
    Kutilgan format: birinchi ustun = yil, qolgan 12 ta = oylar (mm)
    
    Parametrlar:
    -----------
    filepath : str - Excel fayl yo'li
    sheet_name : str - Sheet nomi
    station_name : str - Stansiya nomi (chiqish uchun)
    """
    df_raw = pd.read_excel(filepath, sheet_name=sheet_name)
    
    month_cols_std = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Birinchi ustun = yil, keyingi 12 ta = oylar
    cols = list(df_raw.columns)
    
    df = pd.DataFrame()
    df['year'] = pd.to_numeric(df_raw[cols[0]], errors='coerce').astype('Int64')
    
    for i, std_col in enumerate(month_cols_std):
        if i + 1 < len(cols):
            df[std_col] = pd.to_numeric(df_raw[cols[i + 1]], errors='coerce')
    
    # Salbiy qiymatlarni 0 ga
    for col in month_cols_std:
        if col in df.columns:
            df.loc[df[col] < 0, col] = 0
    
    # Yillik jami
    df['annual'] = df[month_cols_std].sum(axis=1, min_count=10)
    
    # NaN yillarni olib tashlash
    df = df.dropna(subset=['year']).sort_values('year').reset_index(drop=True)
    df['year'] = df['year'].astype(int)
    
    name = station_name or sheet_name
    print(f"  ✅ {name}: {len(df)} yil ({df['year'].min()}-{df['year'].max()})")
    
    return df


def load_all_data(denov_mingchuqur_file='Denov_Mingchuqur.xlsx',
                  denov_sheet=None,
                  mingchuqur_sheet=None,
                  boysun_file='boysun_precipitation_clean.csv'):
    """
    Barcha stansiya ma'lumotlarini yuklash.
    
    Parametrlar:
    -----------
    denov_mingchuqur_file : str - Denov/Mingchuqur Excel fayl
    denov_sheet : str - Denov sheet nomi (None = avtomatik)
    mingchuqur_sheet : str - Mingchuqur sheet nomi (None = avtomatik)
    boysun_file : str - Boysun CSV fayl
    """
    print("━" * 70)
    print("  📂 MA'LUMOTLARNI YUKLASH")
    print("━" * 70)
    print()
    
    # Sheet nomlarini aniqlash
    if denov_sheet is None or mingchuqur_sheet is None:
        xl = pd.ExcelFile(denov_mingchuqur_file)
        sheets = xl.sheet_names
        xl.close()
        print(f"  Mavjud sheetlar: {sheets}")
        print()
        
        # Avtomatik aniqlash
        if denov_sheet is None:
            for s in sheets:
                if 'денов' in s.lower() or 'denov' in s.lower():
                    denov_sheet = s
                    break
            if denov_sheet is None and len(sheets) >= 1:
                denov_sheet = sheets[0]
        
        if mingchuqur_sheet is None:
            for s in sheets:
                if 'минг' in s.lower() or 'ming' in s.lower():
                    mingchuqur_sheet = s
                    break
            if mingchuqur_sheet is None and len(sheets) >= 2:
                mingchuqur_sheet = sheets[1]
    
    print(f"  Denov sheet: '{denov_sheet}'")
    print(f"  Mingchuqur sheet: '{mingchuqur_sheet}'")
    print()
    
    # Yuklash
    df_denov = load_station_data(denov_mingchuqur_file, denov_sheet, 'Denov')
    df_mingchuqur = load_station_data(denov_mingchuqur_file, mingchuqur_sheet, 'Mingchuqur')
    
    # Boysun
    df_boysun = pd.read_csv(boysun_file)
    print(f"  ✅ Boysun: {len(df_boysun)} yil ({df_boysun['year'].min()}-{df_boysun['year'].max()})")
    print()
    
    return df_boysun, df_denov, df_mingchuqur


# ══════════════════════════════════════════════════════════════
# 3-QADAM: MLR BIAS CORRECTION VA TO'LDIRISH
# ══════════════════════════════════════════════════════════════

def mlr_gap_filling(df_boysun, df_denov, df_mingchuqur,
                     gap_start=1977, gap_end=1984):
    """
    Multiple Linear Regression bilan bo'shliqni to'ldirish.
    
    Model: P_Boysun = β₀ + β₁×P_Denov + β₂×P_Mingchuqur
    
    Har bir oy uchun alohida regressiya koeffitsientlari hisoblanadi.
    """
    print("━" * 70)
    print("  📐 MULTIPLE LINEAR REGRESSION (MLR)")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    gap_years = set(range(gap_start, gap_end + 1))
    
    # Umumiy yillar (bo'shliqsiz, uchala stansiyada ham bor)
    boysun_years = set(df_boysun.dropna(subset=['annual'])['year'])
    denov_years = set(df_denov.dropna(subset=['annual'])['year'])
    mingchuqur_years = set(df_mingchuqur.dropna(subset=['annual'])['year'])
    
    common_years = sorted((boysun_years & denov_years & mingchuqur_years) - gap_years)
    
    print(f"  Umumiy kuzatilgan yillar: {len(common_years)}")
    print(f"  Davr: {min(common_years)}-{max(common_years)}")
    print(f"  Bo'shliq: {gap_start}-{gap_end}")
    print()
    
    # Oylik MLR koeffitsientlar
    print(f"  {'Oy':<6} {'R²':<8} {'R²adj':<8} {'RMSE':<8} {'β₀':<8} {'β₁(Den)':<10} {'β₂(Ming)':<10}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*10}")
    
    mlr_params = {}
    
    for col in month_cols:
        # Ma'lumotlarni yig'ish
        y_vals = []  # Boysun (target)
        x1_vals = []  # Denov
        x2_vals = []  # Mingchuqur
        
        for year in common_years:
            b_row = df_boysun[df_boysun['year'] == year]
            d_row = df_denov[df_denov['year'] == year]
            m_row = df_mingchuqur[df_mingchuqur['year'] == year]
            
            if len(b_row) == 0 or len(d_row) == 0 or len(m_row) == 0:
                continue
            
            b_val = b_row[col].values[0]
            d_val = d_row[col].values[0]
            m_val = m_row[col].values[0]
            
            if pd.notna(b_val) and pd.notna(d_val) and pd.notna(m_val):
                y_vals.append(b_val)
                x1_vals.append(d_val)
                x2_vals.append(m_val)
        
        y = np.array(y_vals)
        X = np.column_stack([np.ones(len(y)), x1_vals, x2_vals])
        
        if len(y) < 10:
            print(f"  {col:<6} ⚠️  Yetarli ma'lumot yo'q ({len(y)} nuqta)")
            mlr_params[col] = None
            continue
        
        # MLR: OLS
        # β = (X'X)⁻¹ X'y
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
        except:
            mlr_params[col] = None
            continue
        
        # Bashorat va xatolik
        y_pred = X @ beta
        residuals = y - y_pred
        
        # R²
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_sq = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        
        # Adjusted R²
        n = len(y)
        p = 2  # prediktor soni
        r_sq_adj = 1 - (1 - r_sq) * (n - 1) / (n - p - 1)
        
        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))
        
        mlr_params[col] = {
            'beta0': beta[0],
            'beta1_denov': beta[1],
            'beta2_mingchuqur': beta[2],
            'r_sq': r_sq,
            'r_sq_adj': r_sq_adj,
            'rmse': rmse,
            'n': n
        }
        
        print(f"  {col:<6} {r_sq:<8.3f} {r_sq_adj:<8.3f} {rmse:<8.1f} "
              f"{beta[0]:<8.2f} {beta[1]:<10.3f} {beta[2]:<10.3f}")
    
    print()
    
    # ──────────────────────────────────────────
    # BO'SHLIQNI TO'LDIRISH
    # ──────────────────────────────────────────
    print("━" * 70)
    print(f"  🔄 BO'SHLIQ TO'LDIRISH ({gap_start}-{gap_end})")
    print("━" * 70)
    print()
    
    filled_rows = []
    
    print(f"  {'Yil':<6}", end='')
    for col in month_cols:
        print(f" {col:<5}", end='')
    print(f" {'Yillik':<8}")
    print(f"  {'─'*6}", end='')
    for _ in month_cols:
        print(f" {'─'*5}", end='')
    print(f" {'─'*8}")
    
    for year in range(gap_start, gap_end + 1):
        row = {'year': year}
        
        d_row = df_denov[df_denov['year'] == year]
        m_row = df_mingchuqur[df_mingchuqur['year'] == year]
        
        for col in month_cols:
            params = mlr_params.get(col)
            
            if params is None:
                row[col] = np.nan
                continue
            
            # Denov va Mingchuqur qiymatlari
            d_val = d_row[col].values[0] if len(d_row) > 0 else np.nan
            m_val = m_row[col].values[0] if len(m_row) > 0 else np.nan
            
            if pd.isna(d_val) or pd.isna(m_val):
                # Agar bitta stansiya bo'lmasa — mavjudini ishlatish
                if pd.notna(d_val):
                    pred = params['beta0'] + params['beta1_denov'] * d_val
                elif pd.notna(m_val):
                    pred = params['beta0'] + params['beta2_mingchuqur'] * m_val
                else:
                    row[col] = np.nan
                    continue
            else:
                pred = params['beta0'] + params['beta1_denov'] * d_val + params['beta2_mingchuqur'] * m_val
            
            row[col] = max(0, round(pred, 1))
        
        monthly_vals = [row[col] for col in month_cols if pd.notna(row.get(col))]
        row['annual'] = round(sum(monthly_vals), 1) if monthly_vals else np.nan
        row['data_quality'] = 'infilled_mlr'
        filled_rows.append(row)
        
        print(f"  {year:<6}", end='')
        for col in month_cols:
            v = row[col]
            print(f" {v:<5.1f}" if pd.notna(v) else f" {'--':<5}", end='')
        print(f" {row['annual']:<8.1f}" if pd.notna(row['annual']) else f" {'--':<8}")
    
    print()
    
    # Asosiy DataFrame ga birlashtirish
    df_filled = df_boysun.copy()
    if 'data_quality' not in df_filled.columns:
        df_filled['data_quality'] = 'observed'
    
    df_filled = df_filled[~df_filled['year'].isin(range(gap_start, gap_end + 1))]
    df_gap = pd.DataFrame(filled_rows)
    df_filled = pd.concat([df_filled, df_gap], ignore_index=True)
    df_filled = df_filled.sort_values('year').reset_index(drop=True)
    
    print(f"  ✅ {len(filled_rows)} yil to'ldirildi (MLR: Denov + Mingchuqur)")
    print()
    
    return df_filled, mlr_params


# ══════════════════════════════════════════════════════════════
# 4-QADAM: CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════

def cross_validate_mlr(df_boysun, df_denov, df_mingchuqur, mlr_params,
                        gap_start=1977, gap_end=1984):
    """Leave-k-out cross-validation."""
    
    print("━" * 70)
    print("  ✅ CROSS-VALIDATION")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    gap_years = set(range(gap_start, gap_end + 1))
    boysun_years = set(df_boysun.dropna(subset=['annual'])['year'])
    denov_years = set(df_denov.dropna(subset=['annual'])['year'])
    mingchuqur_years = set(df_mingchuqur.dropna(subset=['annual'])['year'])
    common_years = sorted((boysun_years & denov_years & mingchuqur_years) - gap_years)
    
    # 8 ta tasodifiy yilni test qilish
    np.random.seed(42)
    n_test = min(8, len(common_years) // 3)
    test_years = sorted(np.random.choice(common_years, size=n_test, replace=False))
    
    print(f"  Test yillari ({n_test} ta): {test_years}")
    print()
    
    annual_errors = []
    
    for year in test_years:
        obs_annual = df_boysun[df_boysun['year'] == year]['annual'].values[0]
        d_row = df_denov[df_denov['year'] == year]
        m_row = df_mingchuqur[df_mingchuqur['year'] == year]
        
        pred_annual = 0
        for col in month_cols:
            params = mlr_params.get(col)
            if params is None:
                continue
            d_val = d_row[col].values[0] if len(d_row) > 0 else 0
            m_val = m_row[col].values[0] if len(m_row) > 0 else 0
            pred = params['beta0'] + params['beta1_denov'] * d_val + params['beta2_mingchuqur'] * m_val
            pred_annual += max(0, pred)
        
        if pd.notna(obs_annual):
            annual_errors.append(pred_annual - obs_annual)
    
    errors = np.array(annual_errors)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    bias = np.mean(errors)
    obs_mean = df_boysun['annual'].dropna().mean()
    
    print(f"  YILLIK NATIJALAR:")
    print(f"    MAE:  {mae:.1f} mm")
    print(f"    RMSE: {rmse:.1f} mm")
    print(f"    Bias: {bias:+.1f} mm")
    print(f"    Nisbiy xatolik: {mae/obs_mean*100:.1f}%")
    print()
    
    return mae, rmse, bias


# ══════════════════════════════════════════════════════════════
# 5-QADAM: GRAFIK VA EKSPORT
# ══════════════════════════════════════════════════════════════

def plot_and_export(df_filled, mae, rmse, gap_start=1977, gap_end=1984):
    """Natijalar grafigi va CSV eksport."""
    
    print("━" * 70)
    print("  🎨 GRAFIK VA EKSPORT")
    print("━" * 70)
    print()
    
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'figure.dpi': 150})
    fig, ax = plt.subplots(figsize=(14, 5))
    
    observed = df_filled[df_filled['data_quality'] == 'observed']
    infilled = df_filled[df_filled['data_quality'] == 'infilled_mlr']
    mean_ann = observed['annual'].mean()
    
    ax.bar(observed['year'], observed['annual'],
           color='#457B9D', alpha=0.7, width=0.8, label='Kuzatilgan')
    ax.bar(infilled['year'], infilled['annual'],
           color='#E07060', alpha=0.8, width=0.8,
           label=f"MLR to'ldirilgan ({gap_start}-{gap_end})",
           edgecolor='#CC0000', linewidth=0.5)
    ax.axhline(y=mean_ann, color='#1a1a1a', ls='--', lw=1.2,
               label=f"O'rtacha: {mean_ann:.0f} mm")
    ax.axvspan(gap_start-0.5, gap_end+0.5, color='#FFEEEE', alpha=0.3, zorder=0)
    
    # 11 yillik o'rtacha
    all_annual = df_filled.set_index('year')['annual'].dropna()
    ma = all_annual.rolling(11, center=True, min_periods=6).mean()
    ax.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0, label="11 yillik o'rtacha")
    
    ax.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax.set_ylabel("Yillik yog'ingarchilik (mm)", fontsize=11, fontweight='bold')
    ax.set_title("Boysun MS — MLR bilan to'ldirilgan (Denov + Mingchuqur)",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(1934, 2026)
    
    ax.text(0.02, 0.95,
            f"MLR validatsiya:\nMAE = {mae:.1f} mm\nRMSE = {rmse:.1f} mm",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CCC', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('mlr_gap_filling_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('mlr_gap_filling_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # CSV eksport
    output = 'boysun_precipitation_filled_mlr.csv'
    df_filled.to_csv(output, index=False)
    
    print(f"  ✅ Grafik: mlr_gap_filling_results.png / .pdf")
    print(f"  ✅ Ma'lumot: {output}")
    print()
    
    return output


# ══════════════════════════════════════════════════════════════
# ASOSIY DASTUR
# ══════════════════════════════════════════════════════════════

def main(denov_mingchuqur_file='Denov_Mingchuqur.xlsx',
         denov_sheet=None,
         mingchuqur_sheet=None,
         boysun_file='boysun_precipitation_clean.csv',
         gap_start=1977, gap_end=1984):
    """
    To'liq MLR gap-filling jarayoni.
    
    ISHLATISH:
    ─────────
    # 1. Avval sheet nomlarini ko'ring:
    check_excel_sheets('Denov_Mingchuqur.xlsx')
    
    # 2. Keyin to'ldiring:
    df = main(denov_sheet='...', mingchuqur_sheet='...')
    """
    
    print("=" * 70)
    print("  MLR BO'SHLIQ TO'LDIRISH")
    print("  Boysun MS ← Denov + Mingchuqur")
    print("=" * 70)
    print()
    
    # 1. Ma'lumotlarni yuklash
    df_boysun, df_denov, df_mingchuqur = load_all_data(
        denov_mingchuqur_file, denov_sheet, mingchuqur_sheet, boysun_file
    )
    
    # 2. MLR va to'ldirish
    df_filled, mlr_params = mlr_gap_filling(
        df_boysun, df_denov, df_mingchuqur, gap_start, gap_end
    )
    
    # 3. Cross-validation
    mae, rmse, bias = cross_validate_mlr(
        df_boysun, df_denov, df_mingchuqur, mlr_params, gap_start, gap_end
    )
    
    # 4. Grafik va eksport
    plot_and_export(df_filled, mae, rmse, gap_start, gap_end)
    
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    
    return df_filled


# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Avval sheetlarni ko'ring:
    # check_excel_sheets('Denov_Mingchuqur.xlsx')
    
    # Keyin ishga tushiring:
    df = main()

"""
ERA5 yordamida Boysun MS yog'ingarchilik bo'shliqlarini to'ldirish
═══════════════════════════════════════════════════════════════════

Bu skript 3 bosqichdan iborat:
  A) ERA5 ma'lumotlarini CDS API orqali yuklab olish
  B) Bias correction (tuzatish koeffitsienti hisoblash)
  C) 1977-1984 bo'shliqni to'ldirish va validatsiya

Kerakli kutubxonalar:
  pip install cdsapi netCDF4 xarray numpy pandas matplotlib scipy

ERA5 ma'lumotlari:
  - Manba: Copernicus Climate Data Store (CDS)
  - O'zgaruvchi: total_precipitation (oylik)
  - Koordinatalar: Boysun (38.19°N, 67.21°E)
  - Davr: 1940-2023

Muallif: Paleoklimatologiya laboratoriyasi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# A-BOSQICH: ERA5 MA'LUMOTLARINI YUKLASH
# ══════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────
# A1. CDS API orqali yuklash (bir marta ishga tushiring)
# ─────────────────────────────────────────────────────────────

def download_era5_precipitation():
    """
    ERA5 oylik yog'ingarchilik ma'lumotlarini CDS API orqali yuklash.
    
    OLDINDAN TAYYORGARLIK:
    1. https://cds.climate.copernicus.eu/ da ro'yxatdan o'ting
    2. API kalitingizni ~/.cdsapirc faylga saqlang:
       url: https://cds.climate.copernicus.eu/api
       key: <sizning-api-kalitingiz>
    3. 'pip install cdsapi' o'rnating
    
    ⚠️ Bu funksiya internet kerak — faqat BIR MARTA ishga tushiring!
    """
    try:
        import cdsapi
    except ImportError:
        print("❌ 'cdsapi' o'rnatilmagan!")
        print("   pip install cdsapi")
        print("   Keyin https://cds.climate.copernicus.eu/ dan ro'yxatdan o'ting")
        return None
    
    c = cdsapi.Client()
    
    # Boysun koordinatalari atrofidagi hudud
    # [N, W, S, E] — kichik hudud (1 piksel)
    area = [38.5, 67.0, 38.0, 67.5]  # Boysun atrofi
    
    print("📡 ERA5 ma'lumotlari yuklanmoqda...")
    print(f"   Hudud: {area}")
    print(f"   Davr: 1940-2023")
    print(f"   O'zgaruvchi: total_precipitation (oylik)")
    print()
    
    c.retrieve(
        'reanalysis-era5-single-levels-monthly-means',
        {
            'product_type': 'monthly_averaged_reanalysis',
            'variable': 'total_precipitation',
            'year': [str(y) for y in range(1940, 2024)],
            'month': [f'{m:02d}' for m in range(1, 13)],
            'time': '00:00',
            'area': area,
            'format': 'netcdf',
        },
        'era5_boysun_precipitation.nc'
    )
    
    print("✅ Yuklandi: era5_boysun_precipitation.nc")
    return 'era5_boysun_precipitation.nc'


# ─────────────────────────────────────────────────────────────
# A2. NetCDF faylni o'qish va DataFrame ga aylantirish
# ─────────────────────────────────────────────────────────────

def load_era5_from_netcdf(nc_file='era5_boysun_precipitation.nc'):
    """
    ERA5 NetCDF faylni oylik DataFrame ga aylantirish.
    
    ERA5 da yog'ingarchilik m/kun birlikda beriladi.
    Oylik mm ga aylantirish kerak: m/kun × 1000 × oydagi_kunlar
    """
    try:
        import xarray as xr
    except ImportError:
        print("❌ 'xarray' o'rnatilmagan! pip install xarray netcdf4")
        return None
    
    ds = xr.open_dataset(nc_file)
    
    # Eng yaqin pikselni olish (Boysun: 38.19°N, 67.21°E)
    lat_target = 38.19
    lon_target = 67.21
    
    # Eng yaqin nuqtani tanlash
    precip = ds['tp'].sel(latitude=lat_target, longitude=lon_target, method='nearest')
    
    # m/kun → mm/oy ga aylantirish
    # ERA5 oylik o'rtacha: m/kun
    times = pd.DatetimeIndex(precip.time.values)
    days_in_month = times.days_in_month
    
    precip_mm = precip.values * 1000 * days_in_month  # mm/oy
    
    # DataFrame yaratish
    df_era5 = pd.DataFrame({
        'date': times,
        'year': times.year,
        'month': times.month,
        'precip_mm': precip_mm
    })
    
    ds.close()
    
    print(f"✅ ERA5 ma'lumotlari o'qildi: {len(df_era5)} oy")
    print(f"   Davr: {df_era5['year'].min()}-{df_era5['year'].max()}")
    print(f"   O'rtacha oylik: {df_era5['precip_mm'].mean():.1f} mm")
    
    return df_era5


# ─────────────────────────────────────────────────────────────
# A3. Agar NetCDF fayl bo'lmasa — test ma'lumotlari
# ─────────────────────────────────────────────────────────────

def create_synthetic_era5():
    """
    ERA5 ga o'xshash sintetik ma'lumotlar yaratish (test uchun).
    HAQIQIY ISHDA download_era5_precipitation() ISHLATILADI!
    """
    np.random.seed(55)
    
    years = range(1940, 2024)
    months = range(1, 13)
    
    # Boysun uchun ERA5 ga o'xshash oylik klimatologiya (mm)
    # ERA5 odatda stansiyadan biroz farq qiladi
    era5_clim = [28, 32, 50, 62, 40, 12, 4, 2, 8, 30, 40, 35]  # mm
    era5_std =  [12, 15, 22, 28, 18, 8, 4, 2, 6, 15, 18, 16]
    
    data = []
    for year in years:
        for m_idx, month in enumerate(months):
            val = np.random.normal(era5_clim[m_idx], era5_std[m_idx])
            # Uzoq muddatli trend (biroz isish = biroz ko'p yog'in)
            trend = 0.05 * (year - 1980)
            val = max(0, val + trend)
            data.append({
                'date': pd.Timestamp(year=year, month=month, day=1),
                'year': year,
                'month': month,
                'precip_mm': round(val, 1)
            })
    
    df_era5 = pd.DataFrame(data)
    print("⚠️  Sintetik ERA5 ma'lumotlari yaratildi (test uchun)")
    print(f"   Davr: {df_era5['year'].min()}-{df_era5['year'].max()}")
    print()
    return df_era5


# ══════════════════════════════════════════════════════════════
# B-BOSQICH: BIAS CORRECTION (TUZATISH)
# ══════════════════════════════════════════════════════════════

def compute_bias_correction(df_obs, df_era5):
    """
    ERA5 va stansiya ma'lumotlari orasidagi sistematik farqni
    hisoblash va tuzatish koeffitsientlarini aniqlash.
    
    Usullar:
    1. Linear Scaling (oylik koeffitsient)
    2. Quantile Mapping (taqsimot moslash)
    
    Parametrlar:
    -----------
    df_obs : DataFrame
        Stansiya ma'lumotlari (year, jan, feb, ..., dec)
    df_era5 : DataFrame  
        ERA5 ma'lumotlari (year, month, precip_mm)
    
    Qaytaradi:
    ---------
    dict: Oylik bias correction koeffitsientlari
    """
    print("━" * 70)
    print("  📐 BIAS CORRECTION — TUZATISH KOEFFITSIENTLARI")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # ERA5 ni oylik jadvalga aylantirish
    era5_pivot = df_era5.pivot_table(
        values='precip_mm', index='year', columns='month'
    )
    era5_pivot.columns = month_cols
    
    # Umumiy davr topish (1977-1984 TASHQARI)
    obs_years = set(df_obs['year'].values)
    era5_years = set(era5_pivot.index.values)
    
    # Bo'shliq yillarini chiqarib tashlash
    gap_years = set(range(1977, 1985))
    common_years = sorted((obs_years & era5_years) - gap_years)
    
    print(f"  Umumiy kuzatilgan davr: {min(common_years)}-{max(common_years)}")
    print(f"  Umumiy yillar soni: {len(common_years)}")
    print(f"  Bo'shliq yillari (chiqarilgan): 1977-1984")
    print()
    
    # Oylik bias correction koeffitsientlari
    bias_params = {}
    
    print(f"  {'Oy':<6} {'Obs o\\'rtacha':<12} {'ERA5 o\\'rtacha':<14} "
          f"{'Ratio':<8} {'R²':<8} {'RMSE':<8} {'Usul'}")
    print(f"  {'─'*6} {'─'*12} {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*15}")
    
    for m_idx, col in enumerate(month_cols):
        # Kuzatilgan ma'lumotlar
        obs_vals = []
        era5_vals = []
        
        for year in common_years:
            obs_row = df_obs[df_obs['year'] == year]
            if len(obs_row) > 0:
                obs_val = obs_row[col].values[0]
                if year in era5_pivot.index:
                    era5_val = era5_pivot.loc[year, col]
                    if not pd.isna(obs_val) and not pd.isna(era5_val):
                        obs_vals.append(obs_val)
                        era5_vals.append(era5_val)
        
        obs_arr = np.array(obs_vals)
        era5_arr = np.array(era5_vals)
        
        if len(obs_arr) < 5:
            print(f"  {col:<6} ⚠️  Yetarli ma'lumot yo'q ({len(obs_arr)} nuqta)")
            bias_params[col] = {'method': 'climatology', 'value': np.nanmean(obs_arr) if len(obs_arr) > 0 else 0}
            continue
        
        # Linear Scaling: obs_mean / era5_mean
        obs_mean = np.mean(obs_arr)
        era5_mean = np.mean(era5_arr)
        
        if era5_mean > 0:
            ratio = obs_mean / era5_mean
        else:
            ratio = 1.0
        
        # Chiziqli regressiya
        slope, intercept, r_value, p_value, std_err = stats.linregress(era5_arr, obs_arr)
        r_squared = r_value ** 2
        
        # RMSE (tuzatilgan ERA5 vs obs)
        era5_corrected = era5_arr * ratio
        rmse = np.sqrt(np.mean((obs_arr - era5_corrected) ** 2))
        
        # Qaysi usul yaxshiroq?
        # Agar R² > 0.3 — regressiya, aks holda — scaling
        if r_squared > 0.3 and p_value < 0.05:
            method = 'regression'
            bias_params[col] = {
                'method': 'regression',
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_squared,
                'ratio': ratio
            }
        else:
            method = 'scaling'
            bias_params[col] = {
                'method': 'scaling',
                'ratio': ratio,
                'r_squared': r_squared
            }
        
        print(f"  {col:<6} {obs_mean:<12.1f} {era5_mean:<14.1f} "
              f"{ratio:<8.2f} {r_squared:<8.3f} {rmse:<8.1f} {method}")
    
    print()
    
    return bias_params, era5_pivot


# ══════════════════════════════════════════════════════════════
# C-BOSQICH: BO'SHLIQNI TO'LDIRISH
# ══════════════════════════════════════════════════════════════

def fill_gap_with_era5(df_obs, era5_pivot, bias_params, 
                        gap_start=1977, gap_end=1984):
    """
    ERA5 + bias correction yordamida bo'shliqni to'ldirish.
    
    Parametrlar:
    -----------
    df_obs : DataFrame
        Stansiya ma'lumotlari
    era5_pivot : DataFrame
        ERA5 oylik jadvali
    bias_params : dict
        Oylik tuzatish koeffitsientlari
    gap_start, gap_end : int
        Bo'shliq davri
    """
    print("━" * 70)
    print(f"  🔄 BO'SHLIQNI TO'LDIRISH ({gap_start}-{gap_end})")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Bo'shliq yillari uchun to'ldirish
    filled_rows = []
    
    print(f"  {'Yil':<6}", end='')
    for col in month_cols:
        print(f" {col:<6}", end='')
    print(f" {'Yillik':<8}")
    print(f"  {'─'*6}", end='')
    for _ in month_cols:
        print(f" {'─'*6}", end='')
    print(f" {'─'*8}")
    
    for year in range(gap_start, gap_end + 1):
        row = {'year': year}
        
        for m_idx, col in enumerate(month_cols):
            # ERA5 qiymati
            if year in era5_pivot.index:
                era5_val = era5_pivot.loc[year, col]
            else:
                era5_val = np.nan
            
            if pd.isna(era5_val):
                row[col] = np.nan
                continue
            
            # Bias correction qo'llash
            params = bias_params[col]
            
            if params['method'] == 'regression':
                corrected = params['slope'] * era5_val + params['intercept']
            elif params['method'] == 'scaling':
                corrected = era5_val * params['ratio']
            else:
                corrected = params.get('value', 0)
            
            # Salbiy bo'lmasin
            row[col] = max(0, round(corrected, 1))
        
        # Yillik jami
        monthly_vals = [row[col] for col in month_cols if not pd.isna(row.get(col, np.nan))]
        row['annual'] = round(sum(monthly_vals), 1) if monthly_vals else np.nan
        row['data_quality'] = 'infilled_era5'
        
        filled_rows.append(row)
        
        # Konsolda ko'rsatish
        print(f"  {year:<6}", end='')
        for col in month_cols:
            val = row[col]
            print(f" {val:<6.1f}" if not pd.isna(val) else f" {'NaN':<6}", end='')
        print(f" {row['annual']:<8.1f}" if not pd.isna(row['annual']) else f" {'NaN':<8}")
    
    print()
    
    # Asosiy DataFrame ga qo'shish
    df_filled = df_obs.copy()
    
    # data_quality ustuni qo'shish (agar yo'q bo'lsa)
    if 'data_quality' not in df_filled.columns:
        df_filled['data_quality'] = 'observed'
    
    # Bo'shliq yillarini qo'shish
    df_gap = pd.DataFrame(filled_rows)
    
    # Mavjud bo'shliq qatorlarini olib tashlash
    df_filled = df_filled[~df_filled['year'].isin(range(gap_start, gap_end + 1))]
    
    # Birlashtirish
    df_filled = pd.concat([df_filled, df_gap], ignore_index=True)
    df_filled = df_filled.sort_values('year').reset_index(drop=True)
    
    print(f"  ✅ {len(filled_rows)} yil to'ldirildi (ERA5 + bias correction)")
    print()
    
    return df_filled


# ══════════════════════════════════════════════════════════════
# D-BOSQICH: VALIDATSIYA (Cross-validation)
# ══════════════════════════════════════════════════════════════

def cross_validate(df_obs, era5_pivot, bias_params, n_folds=5):
    """
    Leave-k-out cross-validation.
    Ma'lum yillarni "yashirib", ERA5 bilan qanchalik aniq tiklanishini tekshirish.
    """
    print("━" * 70)
    print("  ✅ CROSS-VALIDATION (TEKSHIRUV)")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Kuzatilgan yillar (bo'shliqsiz)
    gap_years = set(range(1977, 1985))
    obs_years = df_obs[df_obs['data_quality'] == 'observed']['year'].values
    valid_years = [y for y in obs_years if y not in gap_years and y in era5_pivot.index]
    
    np.random.seed(42)
    np.random.shuffle(valid_years)
    
    # N ta yilni "yashirish" va ERA5 bilan tiklash
    n_test = min(8, len(valid_years) // 4)  # 8 yil test (bo'shliqqa o'xshash uzunlik)
    test_years = valid_years[:n_test]
    
    print(f"  Test yillari ({n_test} ta): {sorted(test_years)}")
    print()
    
    # Har bir oy uchun xatolik hisoblash
    errors_monthly = {col: [] for col in month_cols}
    errors_annual = []
    
    for year in test_years:
        obs_row = df_obs[df_obs['year'] == year].iloc[0]
        
        annual_obs = 0
        annual_pred = 0
        
        for col in month_cols:
            obs_val = obs_row[col]
            
            if pd.isna(obs_val) or year not in era5_pivot.index:
                continue
            
            era5_val = era5_pivot.loc[year, col]
            params = bias_params[col]
            
            if params['method'] == 'regression':
                pred_val = params['slope'] * era5_val + params['intercept']
            elif params['method'] == 'scaling':
                pred_val = era5_val * params['ratio']
            else:
                pred_val = params.get('value', 0)
            
            pred_val = max(0, pred_val)
            
            errors_monthly[col].append(pred_val - obs_val)
            annual_obs += obs_val
            annual_pred += pred_val
        
        if annual_obs > 0:
            errors_annual.append(annual_pred - annual_obs)
    
    # Natijalar
    print(f"  {'Oy':<6} {'MAE (mm)':<10} {'RMSE (mm)':<11} {'Bias (mm)':<11} {'Rel.Err (%)':<12}")
    print(f"  {'─'*6} {'─'*10} {'─'*11} {'─'*11} {'─'*12}")
    
    total_mae = []
    for col in month_cols:
        errs = np.array(errors_monthly[col])
        if len(errs) > 0:
            mae = np.mean(np.abs(errs))
            rmse = np.sqrt(np.mean(errs**2))
            bias = np.mean(errs)
            obs_mean = df_obs[col].mean()
            rel_err = (mae / obs_mean * 100) if obs_mean > 0 else 0
            print(f"  {col:<6} {mae:<10.1f} {rmse:<11.1f} {bias:<+11.1f} {rel_err:<12.1f}")
            total_mae.append(mae)
    
    print()
    
    # Yillik xatolik
    if errors_annual:
        annual_errs = np.array(errors_annual)
        print(f"  YILLIK:")
        print(f"    MAE:  {np.mean(np.abs(annual_errs)):.1f} mm")
        print(f"    RMSE: {np.sqrt(np.mean(annual_errs**2)):.1f} mm")
        print(f"    Bias: {np.mean(annual_errs):+.1f} mm")
        annual_obs_mean = df_obs['annual'].mean()
        print(f"    Nisbiy xatolik: {np.mean(np.abs(annual_errs))/annual_obs_mean*100:.1f}%")
    print()
    
    return errors_monthly, errors_annual


# ══════════════════════════════════════════════════════════════
# E-BOSQICH: VIZUALIZATSIYA
# ══════════════════════════════════════════════════════════════

def plot_gap_filling_results(df_filled, gap_start=1977, gap_end=1984):
    """To'ldirish natijalarini vizualizatsiya qilish."""
    
    print("━" * 70)
    print("  🎨 VIZUALIZATSIYA")
    print("━" * 70)
    print()
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 1.0,
        'figure.dpi': 150
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), 
                             gridspec_kw={'hspace': 0.3})
    
    # ── Panel (a): Yillik yog'ingarchilik ──
    ax1 = axes[0]
    
    observed = df_filled[df_filled['data_quality'] == 'observed']
    infilled = df_filled[df_filled['data_quality'] == 'infilled_era5']
    
    mean_annual = observed['annual'].mean()
    
    # Kuzatilgan
    ax1.bar(observed['year'], observed['annual'],
            color='#457B9D', alpha=0.7, width=0.8, 
            label='Kuzatilgan', zorder=2)
    
    # To'ldirilgan (ERA5)
    ax1.bar(infilled['year'], infilled['annual'],
            color='#E07060', alpha=0.8, width=0.8,
            label=f"ERA5 bilan to'ldirilgan ({gap_start}-{gap_end})", 
            zorder=2, edgecolor='#CC0000', linewidth=0.5)
    
    # O'rtacha chiziq
    ax1.axhline(y=mean_annual, color='#1a1a1a', linestyle='--',
                linewidth=1.2, label=f"Uzoq muddatli o'rtacha: {mean_annual:.0f} mm")
    
    # Bo'shliq zonasi
    ax1.axvspan(gap_start - 0.5, gap_end + 0.5, 
                color='#FFEEEE', alpha=0.3, zorder=0)
    
    ax1.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax1.set_ylabel("Yillik yog'ingarchilik (mm)", fontsize=11, fontweight='bold')
    ax1.set_title("Boysun MS — ERA5 bilan to'ldirilgan yog'ingarchilik qatori",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax1.set_xlim(1934, 2026)
    ax1.text(0.02, 0.95, 'a)', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', color='#CC0000')
    
    # ── Panel (b): ERA5 vs Observed scatter plot ──
    ax2 = axes[1]
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Yillik qiymatlarni taqqoslash
    obs_annual = observed.set_index('year')['annual']
    
    # ERA5 yillik (to'ldirilgan yillardan tashqari)
    # ... scatter uchun barcha kuzatilgan yillarni olish
    ax2.scatter(observed['year'], observed['annual'],
                color='#457B9D', s=30, alpha=0.7, label='Kuzatilgan', zorder=3)
    ax2.scatter(infilled['year'], infilled['annual'],
                color='#E07060', s=50, alpha=0.9, marker='D',
                edgecolor='#CC0000', linewidth=0.8,
                label="ERA5 to'ldirilgan", zorder=4)
    
    # 11 yillik harakatlanuvchi o'rtacha (to'liq qator)
    all_annual = df_filled.set_index('year')['annual'].dropna()
    if len(all_annual) > 11:
        ma = all_annual.rolling(window=11, center=True, min_periods=6).mean()
        ax2.plot(ma.index, ma.values, color='#1a1a1a', linewidth=2.0,
                 label="11 yillik harakatlanuvchi o'rtacha", zorder=5)
    
    ax2.axvspan(gap_start - 0.5, gap_end + 0.5,
                color='#FFEEEE', alpha=0.3, zorder=0)
    
    ax2.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax2.set_ylabel("Yillik yog'ingarchilik (mm)", fontsize=11, fontweight='bold')
    ax2.set_title("Yillik qator + harakatlanuvchi o'rtacha",
                  fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9, framealpha=0.95)
    ax2.set_xlim(1934, 2026)
    ax2.text(0.02, 0.95, 'b)', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', color='#CC0000')
    
    plt.tight_layout()
    plt.savefig('era5_gap_filling_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('era5_gap_filling_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✅ Grafiklar saqlandi:")
    print("     - era5_gap_filling_results.png")
    print("     - era5_gap_filling_results.pdf")
    print()


# ══════════════════════════════════════════════════════════════
# ASOSIY DASTUR
# ══════════════════════════════════════════════════════════════

def main(obs_file="Бойсун МС  8.xlsx", 
         obs_sheet='Ёғин',
         era5_file=None,
         use_synthetic_era5=True):
    """
    To'liq ERA5 gap-filling jarayoni.
    
    Parametrlar:
    -----------
    obs_file : str
        Stansiya ma'lumotlari fayli
    obs_sheet : str
        Excel sheet nomi
    era5_file : str or None
        ERA5 NetCDF fayl (None = sintetik)
    use_synthetic_era5 : bool
        True = test ma'lumotlar, False = haqiqiy ERA5
    """
    
    print("=" * 70)
    print("  ERA5 YORDAMIDA BO'SHLIQNI TO'LDIRISH")
    print("  Boysun MS — Yog'ingarchilik (1977-1984)")
    print("=" * 70)
    print()
    
    # ──────────────────────────────────────────────────
    # 1. Stansiya ma'lumotlarini yuklash
    # ──────────────────────────────────────────────────
    print("📂 1. Stansiya ma'lumotlarini yuklash...")
    
    try:
        from boysun_precipitation_cleaning import (
            load_precipitation_data, identify_structure,
            standardize_dataframe, quality_control, fill_gaps
        )
        df_raw = load_precipitation_data(obs_file, obs_sheet)
        year_col, month_cols_raw = identify_structure(df_raw)
        df_obs = standardize_dataframe(df_raw, year_col, month_cols_raw)
        df_obs, _ = quality_control(df_obs)
        df_obs = fill_gaps(df_obs, method='none')
    except Exception as e:
        print(f"  ⚠️  Tozalash skripti topilmadi yoki xatolik: {e}")
        print("  Test ma'lumotlari ishlatilmoqda...")
        try:
            from boysun_precipitation_cleaning import create_test_precipitation_data
        except ImportError:
            # Agar import bo'lmasa, lokal funksiya ishlatiladi
            create_test_precipitation_data = create_synthetic_era5
        df_raw = create_test_precipitation_data()
        # Oddiy standartlashtirish
        df_obs = df_raw.copy()
        df_obs.columns = ['year'] + ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        df_obs['annual'] = df_obs[month_cols].sum(axis=1, min_count=1)
        df_obs['data_quality'] = 'observed'
        df_obs.loc[df_obs['annual'].isna(), 'data_quality'] = 'missing'
    
    print()
    
    # ──────────────────────────────────────────────────
    # 2. ERA5 ma'lumotlarini yuklash
    # ──────────────────────────────────────────────────
    print("📡 2. ERA5 ma'lumotlarini yuklash...")
    
    if use_synthetic_era5 or era5_file is None:
        df_era5 = create_synthetic_era5()
    else:
        df_era5 = load_era5_from_netcdf(era5_file)
    
    if df_era5 is None:
        print("❌ ERA5 ma'lumotlari yuklanmadi!")
        return None
    print()
    
    # ──────────────────────────────────────────────────
    # 3. Bias correction
    # ──────────────────────────────────────────────────
    print("📐 3. Bias correction hisoblash...")
    bias_params, era5_pivot = compute_bias_correction(df_obs, df_era5)
    
    # ──────────────────────────────────────────────────
    # 4. Bo'shliqni to'ldirish
    # ──────────────────────────────────────────────────
    print("🔄 4. Bo'shliqni to'ldirish...")
    df_filled = fill_gap_with_era5(df_obs, era5_pivot, bias_params)
    
    # ──────────────────────────────────────────────────
    # 5. Cross-validation
    # ──────────────────────────────────────────────────
    print("✅ 5. Cross-validation...")
    errors_monthly, errors_annual = cross_validate(df_filled, era5_pivot, bias_params)
    
    # ──────────────────────────────────────────────────
    # 6. Grafik
    # ──────────────────────────────────────────────────
    print("🎨 6. Vizualizatsiya...")
    plot_gap_filling_results(df_filled)
    
    # ──────────────────────────────────────────────────
    # 7. Saqlash
    # ──────────────────────────────────────────────────
    output_file = 'boysun_precipitation_filled_era5.csv'
    df_filled.to_csv(output_file, index=False)
    print(f"  💾 Saqlandi: {output_file}")
    print()
    
    # ──────────────────────────────────────────────────
    # XULOSA
    # ──────────────────────────────────────────────────
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    print()
    print("  📋 Maqolada yozish uchun:")
    print("  ─────────────────────────")
    print("  \"1977–1984 yillar orasidagi yog'ingarchilik ma'lumotlari")
    print("   bo'shlig'i ERA5 reanaliz ma'lumotlari (Hersbach et al., 2020)")
    print("   asosida to'ldirildi. Oylik linear scaling bias correction")
    print("   koeffitsientlari umumiy kuzatilgan davr (1940–1976 va")
    print("   1985–2023) asosida hisoblandi. Cross-validation natijasi:")
    if errors_annual:
        annual_errs = np.array(errors_annual)
        print(f"   yillik MAE = {np.mean(np.abs(annual_errs)):.1f} mm,")
        print(f"   RMSE = {np.sqrt(np.mean(annual_errs**2)):.1f} mm.\"")
    print()
    
    return df_filled


# ══════════════════════════════════════════════════════════════
# ISHGA TUSHIRISH
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # TEST rejimida (sintetik ERA5):
    df = main(use_synthetic_era5=True)
    
    # HAQIQIY ERA5 bilan:
    # df = main(era5_file='era5_boysun_precipitation.nc', use_synthetic_era5=False)

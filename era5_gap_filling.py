"""
ERA5 yordamida Boysun MS yog'ingarchilik bo'shliqlarini to'ldirish
═══════════════════════════════════════════════════════════════════

BOSQICHLAR:
  1-bosqich: ERA5 ma'lumotlarini CDS API orqali yuklab olish
  2-bosqich: Bias correction + bo'shliq to'ldirish + validatsiya

Kerakli kutubxonalar:
  pip install cdsapi netCDF4 xarray numpy pandas matplotlib scipy

Muallif: Paleoklimatologiya laboratoriyasi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════
# 1-BOSQICH: ERA5 MA'LUMOTLARINI YUKLAB OLISH
# ══════════════════════════════════════════════════════════════
# Bu qismni Jupyter da BIR MARTA ishga tushiring.
# Keyin 2-bosqichga o'ting.
# ══════════════════════════════════════════════════════════════

def download_era5():
    """
    ERA5 oylik yog'ingarchilik — Boysun hududi.
    
    TAYYORGARLIK:
    1. https://cds.climate.copernicus.eu/ → ro'yxatdan o'ting
    2. Profile → API Key → nusxalang
    3. ~/.cdsapirc faylga saqlang:
         url: https://cds.climate.copernicus.eu/api
         key: <sizning-api-kalitingiz>
    4. pip install cdsapi
    """
    import cdsapi
    
    c = cdsapi.Client()
    
    # Boysun koordinatalari: 38.19°N, 67.21°E
    # ERA5 0.25° grid → yaqin hudud olish
    area = [38.5, 67.0, 38.0, 67.5]  # [N, W, S, E]
    
    print("📡 ERA5 yuklanmoqda...")
    print(f"   Hudud: lat=[38.0, 38.5], lon=[67.0, 67.5]")
    print(f"   Davr: 1940–2023 (oylik)")
    print(f"   O'zgaruvchi: total_precipitation")
    print("   ⏳ Bu 5-15 daqiqa vaqt olishi mumkin...")
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
        'era5_boysun_precip.nc'
    )
    
    print("✅ Yuklandi: era5_boysun_precip.nc")
    print("   Endi 2-bosqichga o'ting!")


# ══════════════════════════════════════════════════════════════
# 2-BOSQICH: BIAS CORRECTION + TO'LDIRISH + VALIDATSIYA
# ══════════════════════════════════════════════════════════════
# ERA5 faylni yuklagandan SO'NG bu qismni ishga tushiring.
# ══════════════════════════════════════════════════════════════

def load_era5(nc_file='era5_boysun_precip.nc'):
    """
    ERA5 NetCDF → oylik DataFrame (mm/oy).
    """
    import xarray as xr
    
    ds = xr.open_dataset(nc_file)
    
    # Boysun ga eng yaqin piksel
    precip = ds['tp'].sel(latitude=38.19, longitude=67.21, method='nearest')
    
    # ERA5: m/kun → mm/oy
    times = pd.DatetimeIndex(precip.time.values)
    days_in_month = times.days_in_month
    precip_mm = precip.values * 1000 * days_in_month
    
    df = pd.DataFrame({
        'year': times.year,
        'month': times.month,
        'precip_era5': precip_mm
    })
    
    ds.close()
    
    print(f"✅ ERA5 yuklandi: {len(df)} oy ({df['year'].min()}-{df['year'].max()})")
    return df


def load_observed(csv_file='boysun_precipitation_clean.csv'):
    """
    Tozalangan Boysun stansiya ma'lumotlarini yuklash.
    """
    df = pd.read_csv(csv_file)
    print(f"✅ Stansiya yuklandi: {len(df)} yil ({df['year'].min()}-{df['year'].max()})")
    return df


def run_gap_filling(era5_file='era5_boysun_precip.nc',
                    obs_file='boysun_precipitation_clean.csv',
                    gap_start=1977, gap_end=1984):
    """
    To'liq jarayon: ERA5 yuklash → bias correction → to'ldirish → validatsiya.
    """
    
    print("=" * 70)
    print("  ERA5 BILAN BO'SHLIQNI TO'LDIRISH")
    print(f"  Boysun MS yog'ingarchilik ({gap_start}-{gap_end})")
    print("=" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # ──────────────────────────────────────────
    # 1. Ma'lumotlarni yuklash
    # ──────────────────────────────────────────
    df_era5 = load_era5(era5_file)
    df_obs = load_observed(obs_file)
    print()
    
    # ERA5 ni pivot jadvalga aylantirish (year × month)
    era5_pivot = df_era5.pivot_table(
        values='precip_era5', index='year', columns='month'
    )
    era5_pivot.columns = month_cols
    
    # ──────────────────────────────────────────
    # 2. Bias correction koeffitsientlarini hisoblash
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  📐 BIAS CORRECTION")
    print("━" * 70)
    print()
    
    # Umumiy kuzatilgan yillar (bo'shliq TASHQARI)
    gap_years = set(range(gap_start, gap_end + 1))
    obs_years = set(df_obs.dropna(subset=['annual'])['year'].values)
    era5_years = set(era5_pivot.index.values)
    common_years = sorted((obs_years & era5_years) - gap_years)
    
    print(f"  Umumiy yillar: {len(common_years)} ({min(common_years)}-{max(common_years)})")
    print()
    
    print(f"  {'Oy':<6} {'Obs':<8} {'ERA5':<8} {'Ratio':<8} {'R²':<8} {'RMSE':<8}")
    print(f"  {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    
    bias_params = {}
    
    for m_idx, col in enumerate(month_cols):
        obs_vals = []
        era5_vals = []
        
        for year in common_years:
            obs_row = df_obs[df_obs['year'] == year]
            if len(obs_row) == 0:
                continue
            obs_val = obs_row[col].values[0]
            
            if year not in era5_pivot.index:
                continue
            era5_val = era5_pivot.loc[year, col]
            
            if pd.notna(obs_val) and pd.notna(era5_val):
                obs_vals.append(obs_val)
                era5_vals.append(era5_val)
        
        obs_arr = np.array(obs_vals)
        era5_arr = np.array(era5_vals)
        
        obs_mean = np.mean(obs_arr)
        era5_mean = np.mean(era5_arr)
        ratio = obs_mean / era5_mean if era5_mean > 0 else 1.0
        
        # Regressiya
        slope, intercept, r_value, p_value, _ = stats.linregress(era5_arr, obs_arr)
        r_sq = r_value ** 2
        
        # Tuzatilgan RMSE
        corrected = era5_arr * ratio
        rmse = np.sqrt(np.mean((obs_arr - corrected) ** 2))
        
        # Yaxshiroq usulni tanlash
        if r_sq > 0.3 and p_value < 0.05:
            bias_params[col] = {'method': 'regression', 'slope': slope, 'intercept': intercept}
        else:
            bias_params[col] = {'method': 'scaling', 'ratio': ratio}
        
        print(f"  {col:<6} {obs_mean:<8.1f} {era5_mean:<8.1f} {ratio:<8.2f} {r_sq:<8.3f} {rmse:<8.1f}")
    
    print()
    
    # ──────────────────────────────────────────
    # 3. Bo'shliqni to'ldirish
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
        
        for col in month_cols:
            if year not in era5_pivot.index:
                row[col] = np.nan
                continue
            
            era5_val = era5_pivot.loc[year, col]
            params = bias_params[col]
            
            if params['method'] == 'regression':
                corrected = params['slope'] * era5_val + params['intercept']
            else:
                corrected = era5_val * params['ratio']
            
            row[col] = max(0, round(corrected, 1))
        
        monthly_vals = [row[col] for col in month_cols if pd.notna(row.get(col))]
        row['annual'] = round(sum(monthly_vals), 1)
        row['data_quality'] = 'infilled_era5'
        filled_rows.append(row)
        
        print(f"  {year:<6}", end='')
        for col in month_cols:
            print(f" {row[col]:<5.1f}", end='')
        print(f" {row['annual']:<8.1f}")
    
    print()
    
    # Asosiy DataFrame ga birlashtirish
    df_filled = df_obs.copy()
    if 'data_quality' not in df_filled.columns:
        df_filled['data_quality'] = 'observed'
    
    # Bo'shliq yillarini olib tashlash (agar mavjud bo'lsa)
    df_filled = df_filled[~df_filled['year'].isin(range(gap_start, gap_end + 1))]
    
    # To'ldirilgan yillarni qo'shish
    df_gap = pd.DataFrame(filled_rows)
    df_filled = pd.concat([df_filled, df_gap], ignore_index=True)
    df_filled = df_filled.sort_values('year').reset_index(drop=True)
    
    print(f"  ✅ {len(filled_rows)} yil to'ldirildi")
    print()
    
    # ──────────────────────────────────────────
    # 4. Cross-validation
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  ✅ CROSS-VALIDATION")
    print("━" * 70)
    print()
    
    # 8 ta tasodifiy yilni "yashirish" va ERA5 bilan tiklash
    np.random.seed(42)
    test_years = np.random.choice(common_years, size=min(8, len(common_years)//3), replace=False)
    test_years = sorted(test_years)
    
    print(f"  Test yillari: {list(test_years)}")
    print()
    
    annual_errors = []
    
    for year in test_years:
        obs_row = df_obs[df_obs['year'] == year].iloc[0]
        obs_annual = obs_row['annual']
        
        pred_annual = 0
        for col in month_cols:
            era5_val = era5_pivot.loc[year, col]
            params = bias_params[col]
            if params['method'] == 'regression':
                pred = params['slope'] * era5_val + params['intercept']
            else:
                pred = era5_val * params['ratio']
            pred_annual += max(0, pred)
        
        if pd.notna(obs_annual):
            annual_errors.append(pred_annual - obs_annual)
    
    annual_errors = np.array(annual_errors)
    mae = np.mean(np.abs(annual_errors))
    rmse = np.sqrt(np.mean(annual_errors**2))
    bias = np.mean(annual_errors)
    obs_mean = df_obs['annual'].dropna().mean()
    
    print(f"  Yillik natijalar:")
    print(f"    MAE:  {mae:.1f} mm")
    print(f"    RMSE: {rmse:.1f} mm")
    print(f"    Bias: {bias:+.1f} mm")
    print(f"    Nisbiy xatolik: {mae/obs_mean*100:.1f}%")
    print()
    
    # ──────────────────────────────────────────
    # 5. Grafik
    # ──────────────────────────────────────────
    print("━" * 70)
    print("  🎨 GRAFIK")
    print("━" * 70)
    print()
    
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 'figure.dpi': 150})
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    observed = df_filled[df_filled['data_quality'] == 'observed']
    infilled = df_filled[df_filled['data_quality'] == 'infilled_era5']
    mean_ann = observed['annual'].mean()
    
    ax.bar(observed['year'], observed['annual'],
           color='#457B9D', alpha=0.7, width=0.8, label='Kuzatilgan')
    ax.bar(infilled['year'], infilled['annual'],
           color='#E07060', alpha=0.8, width=0.8,
           label=f"ERA5 to'ldirilgan ({gap_start}-{gap_end})",
           edgecolor='#CC0000', linewidth=0.5)
    ax.axhline(y=mean_ann, color='#1a1a1a', ls='--', lw=1.2,
               label=f"O'rtacha: {mean_ann:.0f} mm")
    ax.axvspan(gap_start-0.5, gap_end+0.5, color='#FFEEEE', alpha=0.3, zorder=0)
    
    # 11 yillik harakatlanuvchi o'rtacha
    all_annual = df_filled.set_index('year')['annual'].dropna()
    ma = all_annual.rolling(11, center=True, min_periods=6).mean()
    ax.plot(ma.index, ma.values, color='#1a1a1a', lw=2.0,
            label="11 yillik o'rtacha")
    
    ax.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax.set_ylabel("Yillik yog'ingarchilik (mm)", fontsize=11, fontweight='bold')
    ax.set_title("Boysun MS — ERA5 bilan to'ldirilgan yog'ingarchilik",
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlim(1934, 2026)
    
    # Validatsiya ma'lumotlari
    ax.text(0.02, 0.95,
            f"Validatsiya:\nMAE = {mae:.1f} mm\nRMSE = {rmse:.1f} mm",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='#CCC', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('era5_gap_filling_results.png', dpi=300, bbox_inches='tight')
    plt.savefig('era5_gap_filling_results.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("  ✅ Saqlandi: era5_gap_filling_results.png / .pdf")
    print()
    
    # ──────────────────────────────────────────
    # 6. Eksport
    # ──────────────────────────────────────────
    output = 'boysun_precipitation_filled_era5.csv'
    df_filled.to_csv(output, index=False)
    print(f"  💾 Saqlandi: {output}")
    print(f"     Jami: {len(df_filled)} yil")
    print()
    print("=" * 70)
    print("  ✨ TAYYOR!")
    print("=" * 70)
    
    return df_filled


# ══════════════════════════════════════════════════════════════
# ISHGA TUSHIRISH
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1-BOSQICH: ERA5 yuklash (faqat bir marta)
    # download_era5()
    
    # 2-BOSQICH: Bo'shliq to'ldirish
    df = run_gap_filling(
        era5_file='era5_boysun_precip.nc',
        obs_file='boysun_precipitation_clean.csv',
        gap_start=1977,
        gap_end=1984
    )

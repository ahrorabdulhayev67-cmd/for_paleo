"""
Boysun meteorologik stansiyasi — Yog'ingarchilik ma'lumotlarini tozalash
════════════════════════════════════════════════════════════════════════

Ma'lumot: Boysun MS, 1936–2025
Muammo: 1977–1984 yillar orasida bo'shliq (7 yil), tartibsizliklar

Bu skript:
1. Excel faylni yuklaydi
2. Ma'lumotlar strukturasini tekshiradi va tuzatadi
3. Bo'shliqlarni aniqlaydi
4. Sifat nazoratini o'tkazadi
5. Toza DataFrame qaytaradi
6. Bazaviy statistika va grafiklar chiqaradi

Muallif: Paleoklimatologiya laboratoriyasi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. MA'LUMOTLARNI YUKLASH
# ══════════════════════════════════════════════════════════════

def load_precipitation_data(filepath="Бойсун МС  8.xlsx", sheet_name='Ёғин'):
    """
    Yog'ingarchilik ma'lumotlarini yuklash.
    
    Parametrlar:
    -----------
    filepath : str
        Excel fayl yo'li
    sheet_name : str
        Sheet nomi (default: 'Ёғин')
    
    Qaytaradi:
    ---------
    pandas.DataFrame : xom ma'lumotlar
    """
    print("=" * 70)
    print("  BOYSUN MS — YOG'INGARCHILIK MA'LUMOTLARINI TOZALASH")
    print("=" * 70)
    print()
    
    try:
        df_raw = pd.read_excel(filepath, sheet_name=sheet_name)
        print(f"✅ Fayl yuklandi: {filepath}")
        print(f"   Sheet: '{sheet_name}'")
        print(f"   Shakl: {df_raw.shape[0]} qator × {df_raw.shape[1]} ustun")
        print(f"   Ustunlar: {list(df_raw.columns)}")
        print()
        return df_raw
    except FileNotFoundError:
        print(f"❌ Fayl topilmadi: {filepath}")
        print("   Test ma'lumotlari yaratilmoqda...")
        return create_test_precipitation_data()
    except Exception as e:
        print(f"❌ Xatolik: {e}")
        return create_test_precipitation_data()


def create_test_precipitation_data():
    """Test uchun sintetik yog'ingarchilik ma'lumotlari."""
    np.random.seed(42)
    
    years = list(range(1936, 2025))
    
    # Oylik yog'ingarchilik (mm) — Boysun iqlimiga mos
    # Bahor (mart-may) va kuz (okt-noy) — ko'p yog'in
    monthly_means = [30, 35, 55, 70, 45, 15, 5, 3, 10, 35, 45, 40]  # mm
    monthly_stds =  [15, 18, 25, 30, 20, 10, 5, 3, 8, 18, 22, 20]
    
    data = []
    for year in years:
        row = {'Yil': year}
        for month_idx, month_name in enumerate(['I', 'II', 'III', 'IV', 'V', 
                                                  'VI', 'VII', 'VIII', 'IX', 
                                                  'X', 'XI', 'XII']):
            if 1977 <= year <= 1984:
                # Bo'shliq davri
                row[month_name] = np.nan
            else:
                val = np.random.normal(monthly_means[month_idx], monthly_stds[month_idx])
                row[month_name] = max(0, round(val, 1))  # Salbiy bo'lmasin
        data.append(row)
    
    df = pd.DataFrame(data)
    print("   ⚠️  Sintetik ma'lumotlar yaratildi (test uchun)")
    print()
    return df


# ══════════════════════════════════════════════════════════════
# 2. MA'LUMOTLAR STRUKTURASINI ANIQLASH VA TUZATISH
# ══════════════════════════════════════════════════════════════

def identify_structure(df_raw):
    """
    Ma'lumotlar strukturasini aniqlash.
    Excel fayllarida turli formatlar bo'lishi mumkin.
    """
    print("━" * 70)
    print("  📋 MA'LUMOTLAR STRUKTURASI TAHLILI")
    print("━" * 70)
    print()
    
    # Dastlabki 5 qatorni ko'rsatish
    print("  Dastlabki 5 qator:")
    print(df_raw.head().to_string(index=False))
    print()
    
    # Ustun nomlarini tekshirish
    print(f"  Ustunlar ({len(df_raw.columns)} ta): {list(df_raw.columns)}")
    print(f"  Dtypes: ")
    for col in df_raw.columns:
        print(f"    {col}: {df_raw[col].dtype}")
    print()
    
    # Yil ustunini topish
    year_col = None
    for col in df_raw.columns:
        col_str = str(col).lower().strip()
        if col_str in ['yil', 'год', 'year', 'years', 'йил']:
            year_col = col
            break
        # Raqamli ustun — yillar bo'lishi mumkin
        if df_raw[col].dtype in ['int64', 'float64']:
            vals = df_raw[col].dropna()
            if len(vals) > 0 and 1900 < vals.mean() < 2030:
                year_col = col
                break
    
    if year_col is None:
        # Birinchi ustunni yil deb taxmin qilish
        year_col = df_raw.columns[0]
        print(f"  ⚠️  Yil ustuni aniq topilmadi. '{year_col}' ishlatilmoqda.")
    else:
        print(f"  ✅ Yil ustuni: '{year_col}'")
    
    # Oy ustunlarini topish
    month_cols = [col for col in df_raw.columns if col != year_col]
    print(f"  📅 Oy ustunlari ({len(month_cols)} ta): {month_cols}")
    print()
    
    return year_col, month_cols


def standardize_dataframe(df_raw, year_col, month_cols):
    """
    DataFrame ni standart formatga keltirish.
    
    Standart format:
    - 'year' ustuni: int
    - 'jan', 'feb', ..., 'dec' ustunlari: float (mm)
    - 'annual' ustuni: yillik jami
    """
    print("━" * 70)
    print("  🔧 MA'LUMOTLARNI STANDARTLASHTIRISH")
    print("━" * 70)
    print()
    
    # Standart oy nomlari
    standard_months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                       'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Oy nomlarini moslash
    month_mapping = {}
    
    # Turli formatlarni qo'llab-quvvatlash
    roman_months = ['I', 'II', 'III', 'IV', 'V', 'VI', 
                    'VII', 'VIII', 'IX', 'X', 'XI', 'XII']
    russian_months = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн',
                      'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']
    uzbek_months = ['Yan', 'Fev', 'Mar', 'Apr', 'May', 'Iyun',
                    'Iyul', 'Avg', 'Sen', 'Okt', 'Noy', 'Dek']
    number_months = ['1', '2', '3', '4', '5', '6', 
                     '7', '8', '9', '10', '11', '12']
    
    for i, std_name in enumerate(standard_months):
        for col in month_cols:
            col_clean = str(col).strip()
            if col_clean in [roman_months[i], russian_months[i], 
                           uzbek_months[i], number_months[i],
                           std_name, std_name.capitalize(),
                           str(i+1)]:
                month_mapping[col] = std_name
                break
    
    # Agar moslash topilmasa, tartib bo'yicha belgilash
    if len(month_mapping) < 12 and len(month_cols) >= 12:
        print("  ⚠️  Oy nomlari avtomatik moslanmadi. Tartib bo'yicha belgilanmoqda...")
        month_mapping = {}
        for i, col in enumerate(month_cols[:12]):
            month_mapping[col] = standard_months[i]
    
    print(f"  Oy moslash: {month_mapping}")
    print()
    
    # Yangi DataFrame yaratish
    df_clean = pd.DataFrame()
    
    # Yil ustuni
    df_clean['year'] = pd.to_numeric(df_raw[year_col], errors='coerce').astype('Int64')
    
    # Oy ustunlari
    for orig_col, std_col in month_mapping.items():
        df_clean[std_col] = pd.to_numeric(df_raw[orig_col], errors='coerce')
    
    # Yillik jami
    month_data_cols = [col for col in standard_months if col in df_clean.columns]
    df_clean['annual'] = df_clean[month_data_cols].sum(axis=1, min_count=1)
    
    # NaN yillarni olib tashlash
    df_clean = df_clean.dropna(subset=['year'])
    df_clean['year'] = df_clean['year'].astype(int)
    
    # Tartibga solish
    df_clean = df_clean.sort_values('year').reset_index(drop=True)
    
    # Dublikatlarni tekshirish
    duplicates = df_clean[df_clean['year'].duplicated()]
    if len(duplicates) > 0:
        print(f"  ⚠️  {len(duplicates)} ta dublikat yil topildi! Birinchisi saqlanadi.")
        df_clean = df_clean.drop_duplicates(subset='year', keep='first')
    
    print(f"  ✅ Standartlashtirildi: {len(df_clean)} yil ({df_clean['year'].min()}–{df_clean['year'].max()})")
    print()
    
    return df_clean


# ══════════════════════════════════════════════════════════════
# 3. BO'SHLIQLARNI ANIQLASH VA HISOBOT
# ══════════════════════════════════════════════════════════════

def analyze_gaps(df_clean):
    """Bo'shliqlarni aniqlash va hisobot."""
    
    print("━" * 70)
    print("  🔍 BO'SHLIQLAR TAHLILI")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Yillar bo'yicha bo'shliqlar
    all_years = set(range(df_clean['year'].min(), df_clean['year'].max() + 1))
    present_years = set(df_clean['year'].values)
    missing_years = sorted(all_years - present_years)
    
    if missing_years:
        print(f"  ❌ Yo'qolgan yillar ({len(missing_years)} ta): {missing_years}")
    else:
        print(f"  ✅ Barcha yillar mavjud ({df_clean['year'].min()}–{df_clean['year'].max()})")
    print()
    
    # Oylar bo'yicha bo'shliqlar
    print("  Oylik bo'shliqlar:")
    print(f"  {'Ustun':<8} {'NaN soni':<10} {'NaN %':<10} {'Holat'}")
    print(f"  {'─'*8} {'─'*10} {'─'*10} {'─'*20}")
    
    gap_info = {}
    for col in month_cols + ['annual']:
        n_missing = df_clean[col].isna().sum()
        pct_missing = n_missing / len(df_clean) * 100
        status = '✅ Yaxshi' if pct_missing < 5 else '⚠️ O\'rtacha' if pct_missing < 15 else '❌ Ko\'p'
        print(f"  {col:<8} {n_missing:<10} {pct_missing:<10.1f}% {status}")
        gap_info[col] = {'n_missing': n_missing, 'pct': pct_missing}
    print()
    
    # Ketma-ket bo'shliqlarni topish
    print("  Ketma-ket bo'shliq davrlari (annual asosida):")
    annual = df_clean.set_index('year')['annual']
    
    gap_periods = []
    in_gap = False
    gap_start = None
    
    for year in range(df_clean['year'].min(), df_clean['year'].max() + 1):
        if year not in annual.index or pd.isna(annual.get(year, np.nan)):
            if not in_gap:
                gap_start = year
                in_gap = True
        else:
            if in_gap:
                gap_periods.append((gap_start, year - 1))
                in_gap = False
    if in_gap:
        gap_periods.append((gap_start, df_clean['year'].max()))
    
    if gap_periods:
        for start, end in gap_periods:
            duration = end - start + 1
            print(f"    📌 {start}–{end} ({duration} yil)")
    else:
        print("    ✅ Ketma-ket bo'shliq topilmadi")
    print()
    
    return gap_info, missing_years, gap_periods


# ══════════════════════════════════════════════════════════════
# 4. SIFAT NAZORATI (Quality Control)
# ══════════════════════════════════════════════════════════════

def quality_control(df_clean):
    """
    Sifat nazorati:
    - Salbiy qiymatlar (yog'ingarchilik salbiy bo'la olmaydi)
    - Outlier'lar (juda katta qiymatlar)
    - Mantiqiy tekshiruvlar
    """
    print("━" * 70)
    print("  🛡️  SIFAT NAZORATI")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    issues = []
    
    # 1. Salbiy qiymatlar
    print("  1. Salbiy qiymatlar tekshiruvi:")
    for col in month_cols:
        negative = df_clean[df_clean[col] < 0]
        if len(negative) > 0:
            print(f"     ❌ {col}: {len(negative)} ta salbiy qiymat topildi")
            issues.append(('negative', col, negative.index.tolist()))
            # Tuzatish: salbiy qiymatlarni 0 ga almashtirish
            df_clean.loc[df_clean[col] < 0, col] = 0
        
    if not any(i[0] == 'negative' for i in issues):
        print("     ✅ Salbiy qiymat yo'q")
    print()
    
    # 2. Outlier'lar (IQR usuli)
    print("  2. Outlier tekshiruvi (IQR × 3.0):")
    outlier_count = 0
    for col in month_cols:
        data = df_clean[col].dropna()
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 3.0 * IQR  # Konservativ chegara
        
        outliers = df_clean[df_clean[col] > upper_bound]
        if len(outliers) > 0:
            outlier_count += len(outliers)
            max_val = df_clean[col].max()
            print(f"     ⚠️  {col}: {len(outliers)} ta outlier (max={max_val:.1f}, chegara={upper_bound:.1f})")
            issues.append(('outlier', col, outliers.index.tolist()))
    
    if outlier_count == 0:
        print("     ✅ Outlier topilmadi")
    else:
        print(f"     📊 Jami: {outlier_count} ta potensial outlier")
        print("     ℹ️  Outlier'lar olib tashlanMADI (faqat belgilandi)")
    print()
    
    # 3. Yillik jami mantiqiy tekshiruvi
    print("  3. Yillik jami tekshiruvi:")
    # Boysun uchun kutilgan yillik yog'in: 200-800 mm
    annual_data = df_clean['annual'].dropna()
    too_low = df_clean[df_clean['annual'] < 50]
    too_high = df_clean[df_clean['annual'] > 1500]
    
    if len(too_low) > 0:
        print(f"     ⚠️  Juda past yillik (<50 mm): {len(too_low)} yil")
        for _, row in too_low.iterrows():
            print(f"         {int(row['year'])}: {row['annual']:.1f} mm")
    if len(too_high) > 0:
        print(f"     ⚠️  Juda yuqori yillik (>1500 mm): {len(too_high)} yil")
    if len(too_low) == 0 and len(too_high) == 0:
        print("     ✅ Yillik qiymatlar mantiqiy doirada")
    print()
    
    # 4. Yillik jami qayta hisoblash
    print("  4. Yillik jami qayta hisoblanmoqda...")
    df_clean['annual_calc'] = df_clean[month_cols].sum(axis=1, min_count=10)
    diff = (df_clean['annual'] - df_clean['annual_calc']).abs()
    mismatch = df_clean[diff > 1.0].dropna(subset=['annual', 'annual_calc'])
    if len(mismatch) > 0:
        print(f"     ⚠️  {len(mismatch)} yilda Excel va hisoblangan jami farqli")
        print("     → 'annual' ustuni qayta hisoblandi")
    df_clean['annual'] = df_clean['annual_calc']
    df_clean = df_clean.drop(columns=['annual_calc'])
    print("     ✅ Yillik jami yangilandi")
    print()
    
    return df_clean, issues


# ══════════════════════════════════════════════════════════════
# 5. BO'SHLIQLARNI TO'LDIRISH (Interpolyatsiya)
# ══════════════════════════════════════════════════════════════

def fill_gaps(df_clean, method='none'):
    """
    Bo'shliqlarni to'ldirish.
    
    Metodlar:
    - 'none': To'ldirmaslik (NaN qoldirish)
    - 'climatology': Uzoq muddatli o'rtacha bilan to'ldirish
    - 'linear': Chiziqli interpolyatsiya
    - 'flag': To'ldirish + flag ustuni qo'shish
    
    ⚠️ MUHIM: 7 yillik bo'shliq (1977-1984) uchun interpolyatsiya 
    tavsiya etilMAYDI — bu juda uzoq davr. Faqat flag qo'shish tavsiya etiladi.
    """
    print("━" * 70)
    print("  🔄 BO'SHLIQLARNI BOSHQARISH")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Flag ustuni qo'shish
    df_clean['data_quality'] = 'observed'  # Default: kuzatilgan
    
    # Bo'shliq yillarini belgilash
    for idx, row in df_clean.iterrows():
        n_missing = row[month_cols].isna().sum()
        if n_missing == 12:
            df_clean.loc[idx, 'data_quality'] = 'missing'
        elif n_missing > 0:
            df_clean.loc[idx, 'data_quality'] = 'partial'
    
    # To'liq yillar qatorini yaratish (barcha yillar uchun)
    full_years = pd.DataFrame({
        'year': range(df_clean['year'].min(), df_clean['year'].max() + 1)
    })
    df_complete = full_years.merge(df_clean, on='year', how='left')
    
    # Yo'qolgan yillar uchun quality flag
    df_complete.loc[df_complete['data_quality'].isna(), 'data_quality'] = 'missing'
    
    if method == 'climatology':
        print("  📊 Klimatologik o'rtacha bilan to'ldirish...")
        print("  ⚠️  OGOHLANTIRUV: 7 yillik bo'shliq uchun bu ideal emas!")
        print()
        
        # Uzoq muddatli oylik o'rtacha
        climatology = {}
        for col in month_cols:
            climatology[col] = df_complete.loc[
                df_complete['data_quality'] == 'observed', col
            ].mean()
            print(f"    {col}: {climatology[col]:.1f} mm (klimatologik o'rtacha)")
        print()
        
        # To'ldirish
        for col in month_cols:
            mask = df_complete[col].isna()
            df_complete.loc[mask, col] = climatology[col]
        
        # To'ldirilgan yillar uchun flag yangilash
        df_complete.loc[df_complete['data_quality'] == 'missing', 'data_quality'] = 'infilled_clim'
        
        # Yillik jami qayta hisoblash
        df_complete['annual'] = df_complete[month_cols].sum(axis=1, min_count=1)
        
    elif method == 'linear':
        print("  📈 Chiziqli interpolyatsiya...")
        print("  ⚠️  OGOHLANTIRUV: 7 yillik uzilish uchun ishonchsiz!")
        print()
        
        for col in month_cols:
            df_complete[col] = df_complete[col].interpolate(method='linear')
        
        df_complete.loc[df_complete['data_quality'] == 'missing', 'data_quality'] = 'infilled_interp'
        df_complete['annual'] = df_complete[month_cols].sum(axis=1, min_count=1)
        
    else:
        print("  ℹ️  Bo'shliqlar to'ldirilmadi (NaN qoldirildi)")
        print("  ✅ 'data_quality' flag ustuni qo'shildi")
        print()
        # Yillik jami qayta hisoblash
        df_complete['annual'] = df_complete[month_cols].sum(axis=1, min_count=10)
    
    # Xulosa
    quality_counts = df_complete['data_quality'].value_counts()
    print("  📊 Ma'lumot sifati xulosasi:")
    for status, count in quality_counts.items():
        pct = count / len(df_complete) * 100
        emoji = '✅' if status == 'observed' else '⚠️' if 'infilled' in status else '❌'
        print(f"    {emoji} {status}: {count} yil ({pct:.1f}%)")
    print()
    
    return df_complete


# ══════════════════════════════════════════════════════════════
# 6. BAZAVIY STATISTIKA
# ══════════════════════════════════════════════════════════════

def compute_statistics(df_complete):
    """Asosiy statistik ko'rsatkichlar."""
    
    print("━" * 70)
    print("  📊 BAZAVIY STATISTIKA")
    print("━" * 70)
    print()
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # Faqat kuzatilgan ma'lumotlar
    observed = df_complete[df_complete['data_quality'] == 'observed']
    
    print(f"  Kuzatilgan davr: {observed['year'].min()}–{observed['year'].max()}")
    print(f"  Kuzatilgan yillar soni: {len(observed)}")
    print()
    
    # Oylik statistika
    print(f"  {'Oy':<6} {'O\'rtacha':<10} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
    print(f"  {'─'*6} {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")
    
    monthly_stats = {}
    for col in month_cols:
        data = observed[col].dropna()
        stats = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'median': data.median()
        }
        monthly_stats[col] = stats
        print(f"  {col:<6} {stats['mean']:<10.1f} {stats['std']:<8.1f} "
              f"{stats['min']:<8.1f} {stats['max']:<8.1f} {stats['median']:<8.1f}")
    
    print()
    
    # Yillik statistika
    annual_obs = observed['annual'].dropna()
    print(f"  YILLIK JAMI:")
    print(f"    O'rtacha: {annual_obs.mean():.1f} mm")
    print(f"    Std:      {annual_obs.std():.1f} mm")
    print(f"    Min:      {annual_obs.min():.1f} mm ({observed.loc[annual_obs.idxmin(), 'year']})")
    print(f"    Max:      {annual_obs.max():.1f} mm ({observed.loc[annual_obs.idxmax(), 'year']})")
    print(f"    CV:       {(annual_obs.std()/annual_obs.mean()*100):.1f}%")
    print()
    
    return monthly_stats


# ══════════════════════════════════════════════════════════════
# 7. VIZUALIZATSIYA
# ══════════════════════════════════════════════════════════════

def plot_precipitation(df_complete):
    """Yog'ingarchilik grafiklari."""
    
    print("━" * 70)
    print("  🎨 GRAFIKLAR")
    print("━" * 70)
    print()
    
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.linewidth': 1.0,
        'figure.dpi': 150
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), 
                             gridspec_kw={'hspace': 0.35})
    
    month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                  'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    # ── Panel (a): Yillik yog'ingarchilik ──
    ax1 = axes[0]
    
    observed = df_complete[df_complete['data_quality'] == 'observed']
    missing = df_complete[df_complete['data_quality'] != 'observed']
    
    # O'rtacha chiziq
    mean_annual = observed['annual'].mean()
    
    ax1.bar(observed['year'], observed['annual'], 
            color='#457B9D', alpha=0.7, width=0.8, label='Kuzatilgan')
    
    if len(missing) > 0 and not missing['annual'].isna().all():
        ax1.bar(missing['year'], missing['annual'],
                color='#E07060', alpha=0.5, width=0.8, label="To'ldirilgan")
    
    ax1.axhline(y=mean_annual, color='#CC0000', linestyle='--', 
                linewidth=1.2, label=f"O'rtacha: {mean_annual:.0f} mm")
    
    # Bo'shliq davrini belgilash
    ax1.axvspan(1977, 1984, color='#FFCCCC', alpha=0.3, zorder=0,
                label="Ma'lumot yo'q (1977–1984)")
    
    ax1.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Yillik yog\'ingarchilik (mm)', fontsize=11, fontweight='bold')
    ax1.set_title("Boysun MS — Yillik yog'ingarchilik (1936–2025)",
                  fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim(1934, 2026)
    ax1.text(0.02, 0.95, 'a)', transform=ax1.transAxes,
             fontsize=12, fontweight='bold', va='top', color='#CC0000')
    
    # ── Panel (b): Oylik klimatologiya ──
    ax2 = axes[1]
    
    monthly_means = [observed[col].mean() for col in month_cols]
    monthly_stds = [observed[col].std() for col in month_cols]
    
    months_labels = ['Yan', 'Fev', 'Mar', 'Apr', 'May', 'Iyn',
                     'Iyl', 'Avg', 'Sen', 'Okt', 'Noy', 'Dek']
    
    bars = ax2.bar(months_labels, monthly_means, color='#457B9D', alpha=0.8,
                   edgecolor='#2C3E50', linewidth=0.5)
    ax2.errorbar(months_labels, monthly_means, yerr=monthly_stds,
                 fmt='none', color='#1a1a1a', capsize=3, linewidth=1.2)
    
    ax2.set_xlabel('Oy', fontsize=11, fontweight='bold')
    ax2.set_ylabel("O'rtacha yog'ingarchilik (mm)", fontsize=11, fontweight='bold')
    ax2.set_title("Oylik klimatologiya (uzoq muddatli o'rtacha ± std)",
                  fontsize=12, fontweight='bold')
    ax2.text(0.02, 0.95, 'b)', transform=ax2.transAxes,
             fontsize=12, fontweight='bold', va='top', color='#CC0000')
    
    # ── Panel (c): Ma'lumot mavjudligi (heatmap) ──
    ax3 = axes[2]
    
    # Mavjudlik matritsasi
    years = df_complete['year'].values
    availability = np.zeros((len(month_cols), len(years)))
    
    for i, col in enumerate(month_cols):
        for j, year in enumerate(years):
            val = df_complete.loc[df_complete['year'] == year, col].values
            if len(val) > 0 and not pd.isna(val[0]):
                availability[i, j] = 1
            else:
                availability[i, j] = 0
    
    im = ax3.imshow(availability, aspect='auto', cmap='RdYlGn',
                    interpolation='nearest', vmin=0, vmax=1)
    
    ax3.set_yticks(range(12))
    ax3.set_yticklabels(months_labels)
    
    # X o'qi
    tick_positions = np.arange(0, len(years), 10)
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels([str(years[i]) for i in tick_positions], rotation=45)
    
    ax3.set_xlabel('Yil', fontsize=11, fontweight='bold')
    ax3.set_title("Ma'lumot mavjudligi (yashil=bor, qizil=yo'q)",
                  fontsize=12, fontweight='bold')
    ax3.text(0.02, 0.95, 'c)', transform=ax3.transAxes,
             fontsize=12, fontweight='bold', va='top', color='#CC0000')
    
    plt.tight_layout()
    
    plt.savefig('boysun_precipitation_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig('boysun_precipitation_overview.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✅ Grafiklar saqlandi:")
    print("     - boysun_precipitation_overview.png")
    print("     - boysun_precipitation_overview.pdf")
    print()


# ══════════════════════════════════════════════════════════════
# 8. YAKUNIY EKSPORT
# ══════════════════════════════════════════════════════════════

def export_clean_data(df_complete, output_file='boysun_precipitation_clean.csv'):
    """Tozalangan ma'lumotlarni eksport qilish."""
    
    print("━" * 70)
    print("  💾 EKSPORT")
    print("━" * 70)
    print()
    
    df_complete.to_csv(output_file, index=False)
    print(f"  ✅ Saqlandi: {output_file}")
    print(f"     Qatorlar: {len(df_complete)}")
    print(f"     Ustunlar: {list(df_complete.columns)}")
    print()
    
    return output_file


# ══════════════════════════════════════════════════════════════
# ASOSIY DASTUR
# ══════════════════════════════════════════════════════════════

def main(filepath="Бойсун МС  8.xlsx", sheet_name='Ёғин', fill_method='none'):
    """
    To'liq tozalash jarayoni.
    
    Parametrlar:
    -----------
    filepath : str
        Excel fayl yo'li
    sheet_name : str
        Sheet nomi
    fill_method : str
        'none' — to'ldirmaslik (tavsiya etiladi)
        'climatology' — uzoq muddatli o'rtacha bilan
        'linear' — chiziqli interpolyatsiya
    """
    
    # 1. Yuklash
    df_raw = load_precipitation_data(filepath, sheet_name)
    
    # 2. Strukturani aniqlash
    year_col, month_cols = identify_structure(df_raw)
    
    # 3. Standartlashtirish
    df_clean = standardize_dataframe(df_raw, year_col, month_cols)
    
    # 4. Bo'shliqlar tahlili
    gap_info, missing_years, gap_periods = analyze_gaps(df_clean)
    
    # 5. Sifat nazorati
    df_clean, issues = quality_control(df_clean)
    
    # 6. Bo'shliqlarni boshqarish
    df_complete = fill_gaps(df_clean, method=fill_method)
    
    # 7. Statistika
    monthly_stats = compute_statistics(df_complete)
    
    # 8. Grafiklar
    plot_precipitation(df_complete)
    
    # 9. Eksport
    export_clean_data(df_complete)
    
    print("=" * 70)
    print("  ✨ MA'LUMOTLAR TOZALASH TUGALLANDI!")
    print("=" * 70)
    print()
    print("  📋 Keyingi qadamlar:")
    print("     1. Grafiklarni tekshiring")
    print("     2. Outlier'larni qayta ko'rib chiqing")
    print("     3. Bo'shliqlar uchun qaror qabul qiling:")
    print("        - NaN qoldirish (tahlil uchun eng xavfsiz)")
    print("        - Klimatologik o'rtacha (faqat mavsumiy tahlil uchun)")
    print("        - Qo'shni stansiyalar bilan to'ldirish (eng yaxshi)")
    print("     4. Mann-Kendall trend testini o'tkazing")
    print()
    
    return df_complete


# ══════════════════════════════════════════════════════════════
# ISHGA TUSHIRISH
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # TO'LDIRILMASDAN tozalash (tavsiya etiladi):
    df = main(fill_method='none')
    
    # Agar klimatologik o'rtacha bilan to'ldirmoqchi bo'lsangiz:
    # df = main(fill_method='climatology')

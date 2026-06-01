"""
TON-2 Speleothem Paleoclimate Reconstruction Figure
Publication-ready multi-panel figure with annotations

Author: Generated for paleoclimate analysis
Date: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# SOZLAMALAR (Bu qismni o'zgartiring)
# =====================================================

# Ma'lumotlar fayli yo'li
DATA_FILE = 'TON2_data.csv'  # yoki 'TON2_data.xlsx'
FILE_TYPE = 'csv'  # 'csv' yoki 'excel'

# Ustun nomlari (sizning CSV/Excel faylingizdagi ustun nomlari)
COLUMNS = {
    'age': 'age_calBP',              # Yosh (cal yr BP)
    'd18O_calcite': 'd18O_calcite',  # δ¹⁸O calcite
    'd18O_water': 'delta_d18O_water', # Δδ¹⁸O water
    'temperature': 'temp_anomaly',    # Temperatura anomaliyasi (°C)
    'precipitation': 'MAP_anomaly',   # Yog'ingarchilik anomaliyasi (mm)
}

# Uncertainty qiymatlari (agar ma'lumotda bo'lmasa, default qiymatlar)
UNCERTAINTIES = {
    'd18O': 0.1,           # δ¹⁸O uchun ±0.1‰
    'temperature': 0.8,    # Temperatura uchun ±0.8°C
    'precipitation': 30,   # MAP uchun ±30 mm
}

# Muhim nuqtalar (age in ka BP, value)
ANNOTATIONS = {
    'LGM': {'age': 21, 'temp': -4.5, 'precip': -150},
    'MIS5e': {'age': 85, 'temp': 3.8, 'precip': 100},
    'Holocene': {'age': 8, 'temp': 2.0, 'precip': 180},
}

# Hiatus davrlari (age range in ka BP)
# Agar avtomatik topmoqchi bo'lsangiz, [] qoldiring
HIATUS_PERIODS = [
    (36, 59),    # Birinchi katta uzilish
    (101, 144),  # Ikkinchi katta uzilish
]

# Rang sxemasi
COLORS = {
    'd18O_calcite': '#2E86AB',    # Ko'k
    'd18O_water': '#A23B72',      # Binafsha
    'temperature': '#E63946',     # Qizil
    'precipitation': '#457B9D',   # Ochiq ko'k
    'hiatus': 'gray',             # Hiatus zona rangi
}

# Y o'qi chegaralari
Y_LIMITS = {
    'd18O_calcite': None,      # None = avtomatik
    'd18O_water': None,
    'temperature': (-6, 6),    # Simmetrik
    'precipitation': (-300, 300),  # Simmetrik
}

# Rasm o'lchamlari
FIGURE_SIZE = (10, 12)  # width, height in inches
DPI = 300

# Chiqish fayl nomlari
OUTPUT_FILES = {
    'png': 'TON2_reconstruction_annotated.png',
    'pdf': 'TON2_reconstruction_annotated.pdf',
}

# =====================================================
# MA'LUMOTLARNI YUKLASH
# =====================================================

def load_data(file_path, file_type='csv'):
    """Ma'lumotlarni yuklash"""
    try:
        if file_type.lower() == 'csv':
            df = pd.read_csv(file_path)
        elif file_type.lower() == 'excel':
            df = pd.read_excel(file_path)
        else:
            raise ValueError("file_type 'csv' yoki 'excel' bo'lishi kerak")
        
        print(f"✅ Ma'lumotlar yuklandi: {len(df)} qator")
        return df
    except FileNotFoundError:
        print(f"⚠️  Fayl topilmadi: {file_path}")
        print("📊 Test ma'lumotlar yaratilmoqda...")
        return create_test_data()
    except Exception as e:
        print(f"❌ Xatolik: {e}")
        print("📊 Test ma'lumotlar yaratilmoqda...")
        return create_test_data()

def create_test_data():
    """Test ma'lumotlar yaratish (haqiqiy ma'lumot bo'lmasa)"""
    age_bp = np.linspace(8000, 134000, 2007)
    
    # Realistik sintetik ma'lumotlar
    age_ka = age_bp / 1000
    
    df = pd.DataFrame({
        'age_calBP': age_bp,
        'd18O_calcite': -5 + np.sin(age_ka/10) * 2 + np.random.randn(len(age_ka)) * 0.3,
        'delta_d18O_water': -3 + np.sin(age_ka/10) * 1.5 + np.random.randn(len(age_ka)) * 0.2,
        'temp_anomaly': -2 + 4*np.sin(age_ka/15) + 2*np.cos(age_ka/25) + np.random.randn(len(age_ka)) * 0.5,
        'MAP_anomaly': 50 + 150*np.sin(age_ka/12) - 80*np.cos(age_ka/20) + np.random.randn(len(age_ka)) * 20,
    })
    
    return df

def detect_hiatus(age_ka, threshold_ka=2.0):
    """Avtomatik hiatus davrlarini topish"""
    age_diff = np.abs(np.diff(age_ka))
    hiatus_indices = np.where(age_diff > threshold_ka)[0]
    
    hiatus_periods = []
    for idx in hiatus_indices:
        start = min(age_ka[idx], age_ka[idx + 1])
        end = max(age_ka[idx], age_ka[idx + 1])
        hiatus_periods.append((start, end))
        print(f"  Hiatus topildi: {start:.1f} - {end:.1f} ka BP (uzilish: {end-start:.1f} ka)")
    
    return hiatus_periods

# =====================================================
# RASMNI YARATISH
# =====================================================

def create_publication_figure(df, hiatus_periods=None):
    """Publication-ready figure yaratish"""
    
    # Ma'lumotlarni tayyorlash
    age_ka = df[COLUMNS['age']].values / 1000  # ka ga o'tkazish
    
    # Style sozlamalari
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 1.2
    plt.rcParams['figure.dpi'] = DPI
    
    # Figure va axes yaratish
    fig, axes = plt.subplots(3, 1, figsize=FIGURE_SIZE, sharex=True)
    fig.subplots_adjust(hspace=0.05)
    
    # =====================================================
    # PANEL A: δ¹⁸O calcite va Δδ¹⁸O water
    # =====================================================
    ax1 = axes[0]
    
    # δ¹⁸O calcite
    d18O_calc = df[COLUMNS['d18O_calcite']].values
    line1 = ax1.plot(age_ka, d18O_calc, 
                     color=COLORS['d18O_calcite'], 
                     linewidth=1.5, 
                     label='δ¹⁸O calcite (‰ VPDB)', 
                     zorder=3)
    ax1.fill_between(age_ka, 
                     d18O_calc - UNCERTAINTIES['d18O'], 
                     d18O_calc + UNCERTAINTIES['d18O'],
                     color=COLORS['d18O_calcite'], 
                     alpha=0.2, 
                     zorder=2)
    
    # Δδ¹⁸O water (ikkinchi Y o'qi)
    ax1_twin = ax1.twinx()
    d18O_water = df[COLUMNS['d18O_water']].values
    line2 = ax1_twin.plot(age_ka, d18O_water, 
                          color=COLORS['d18O_water'], 
                          linewidth=1.5,
                          linestyle='--', 
                          label='Δδ¹⁸O water (‰ VSMOW)', 
                          zorder=3)
    ax1_twin.fill_between(age_ka, 
                          d18O_water - UNCERTAINTIES['d18O']/2, 
                          d18O_water + UNCERTAINTIES['d18O']/2,
                          color=COLORS['d18O_water'], 
                          alpha=0.15, 
                          zorder=2)
    
    # Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', frameon=True, 
              fancybox=False, edgecolor='black', fontsize=9)
    
    # O'qlar
    ax1.set_ylabel('δ¹⁸O calcite (‰ VPDB)', fontsize=11, fontweight='bold')
    ax1_twin.set_ylabel('Δδ¹⁸O water (‰ VSMOW)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    if Y_LIMITS['d18O_calcite']:
        ax1.set_ylim(Y_LIMITS['d18O_calcite'])
    if Y_LIMITS['d18O_water']:
        ax1_twin.set_ylim(Y_LIMITS['d18O_water'])
    
    # Panel label
    ax1.text(0.02, 0.95, 'a', transform=ax1.transAxes, 
             fontsize=14, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor='none', alpha=0.8))
    
    # =====================================================
    # PANEL B: Temperature reconstruction
    # =====================================================
    ax2 = axes[1]
    
    temp = df[COLUMNS['temperature']].values
    ax2.plot(age_ka, temp, 
             color=COLORS['temperature'], 
             linewidth=1.5, 
             zorder=3)
    ax2.fill_between(age_ka, 
                     temp - UNCERTAINTIES['temperature'], 
                     temp + UNCERTAINTIES['temperature'],
                     color=COLORS['temperature'], 
                     alpha=0.25, 
                     zorder=2)
    
    # Zero line
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    # Annotatsiyalar
    # LGM
    ax2.annotate('LGM\n-4.5°C', 
                xy=(ANNOTATIONS['LGM']['age'], ANNOTATIONS['LGM']['temp']), 
                xytext=(ANNOTATIONS['LGM']['age'] + 15, ANNOTATIONS['LGM']['temp'] - 2),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                fontsize=9, fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                         edgecolor='black', linewidth=1))
    
    # MIS 5e
    ax2.annotate('MIS 5e\n+3.8°C', 
                xy=(ANNOTATIONS['MIS5e']['age'], ANNOTATIONS['MIS5e']['temp']),
                xytext=(ANNOTATIONS['MIS5e']['age'] - 20, ANNOTATIONS['MIS5e']['temp'] + 1.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                fontsize=9, fontweight='bold', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=1))
    
    # Holocene
    ax2.annotate('Holocene\nOptimum', 
                xy=(ANNOTATIONS['Holocene']['age'], ANNOTATIONS['Holocene']['temp']),
                xytext=(ANNOTATIONS['Holocene']['age'] + 12, ANNOTATIONS['Holocene']['temp'] + 1.5),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                fontsize=9, fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=1))
    
    # O'qlar
    ax2.set_ylabel('Temperature anomaly (°C)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    if Y_LIMITS['temperature']:
        ax2.set_ylim(Y_LIMITS['temperature'])
    
    # Panel label
    ax2.text(0.02, 0.95, 'b', transform=ax2.transAxes,
             fontsize=14, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.8))
    
    # =====================================================
    # PANEL C: Precipitation reconstruction
    # =====================================================
    ax3 = axes[2]
    
    precip = df[COLUMNS['precipitation']].values
    ax3.plot(age_ka, precip, 
             color=COLORS['precipitation'], 
             linewidth=1.5, 
             zorder=3)
    ax3.fill_between(age_ka, 
                     precip - UNCERTAINTIES['precipitation'], 
                     precip + UNCERTAINTIES['precipitation'],
                     color=COLORS['precipitation'], 
                     alpha=0.25, 
                     zorder=2)
    
    # Zero line
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)
    
    # Annotatsiyalar
    # Holocene peak
    ax3.annotate('Holocene peak\n+180 mm', 
                xy=(ANNOTATIONS['Holocene']['age'], ANNOTATIONS['Holocene']['precip']),
                xytext=(ANNOTATIONS['Holocene']['age'] + 15, ANNOTATIONS['Holocene']['precip'] + 50),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                fontsize=9, fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=1))
    
    # LGM
    ax3.annotate('LGM\n-150 mm', 
                xy=(ANNOTATIONS['LGM']['age'], ANNOTATIONS['LGM']['precip']),
                xytext=(ANNOTATIONS['LGM']['age'] + 15, ANNOTATIONS['LGM']['precip'] - 50),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='black'),
                fontsize=9, fontweight='bold', ha='left',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=1))
    
    # O'qlar
    ax3.set_xlabel('Age (ka BP) → older', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MAP anomaly (mm)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    if Y_LIMITS['precipitation']:
        ax3.set_ylim(Y_LIMITS['precipitation'])
    
    # Panel label
    ax3.text(0.02, 0.95, 'c', transform=ax3.transAxes,
             fontsize=14, fontweight='bold', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='none', alpha=0.8))
    
    # =====================================================
    # HIATUS zonalarini qo'shish
    # =====================================================
    if hiatus_periods:
        for ax_idx, ax in enumerate(axes):
            for hiatus_start, hiatus_end in hiatus_periods:
                ax.axvspan(hiatus_start, hiatus_end, 
                          color=COLORS['hiatus'], 
                          alpha=0.15, 
                          zorder=0)
                
                # "HIATUS" yozuvi (faqat birinchi panelda)
                if ax_idx == 0:
                    mid_point = (hiatus_start + hiatus_end) / 2
                    ylim = ax.get_ylim()
                    y_pos = ylim[0] + (ylim[1] - ylim[0]) * 0.85
                    ax.text(mid_point, y_pos, 'HIATUS',
                           ha='center', va='top', fontsize=8, style='italic',
                           color=COLORS['hiatus'], fontweight='bold', 
                           rotation=90, alpha=0.7)
    
    # X o'qini inverted qilish (o'ngga - eski)
    ax3.set_xlim(age_ka.max(), age_ka.min())
    
    # =====================================================
    # Figure caption
    # =====================================================
    caption = """Fig. X. TON-2 speleothem quantitative paleoclimate reconstruction. (a) δ¹⁸O calcite 
(blue solid line, VPDB scale) and Δδ¹⁸O water (purple dashed line, VSMOW scale) with analytical 
uncertainties (shaded areas). (b) Temperature anomaly reconstruction relative to present day with 
key climatic periods annotated: Last Glacial Maximum (LGM, ~21 ka BP), Marine Isotope Stage 5e 
(MIS 5e, ~85 ka BP), and Holocene Optimum (~8 ka BP). (c) Mean annual precipitation (MAP) anomaly 
with Holocene peak and LGM minimum highlighted. Gray vertical bars indicate hiatus periods with 
missing stratigraphic sections. All ages are calibrated radiocarbon years before present (ka BP)."""
    
    fig.text(0.12, 0.01, caption, wrap=True, fontsize=8, 
            verticalalignment='bottom', style='italic')
    
    # Layout
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    return fig

# =====================================================
# ASOSIY DASTUR
# =====================================================

def main():
    """Asosiy funksiya"""
    print("=" * 60)
    print("TON-2 SPELEOTHEM PALEOCLIMATE FIGURE GENERATOR")
    print("=" * 60)
    print()
    
    # 1. Ma'lumotlarni yuklash
    print("📂 Ma'lumotlarni yuklash...")
    df = load_data(DATA_FILE, FILE_TYPE)
    print()
    
    # 2. Hiatus davrlarini topish yoki ishlatish
    print("🔍 Hiatus davrlarini tekshirish...")
    if HIATUS_PERIODS:
        hiatus_periods = HIATUS_PERIODS
        print(f"  {len(hiatus_periods)} ta hiatus davr belgilangan")
    else:
        age_ka = df[COLUMNS['age']].values / 1000
        hiatus_periods = detect_hiatus(age_ka, threshold_ka=2.0)
    
    if hiatus_periods:
        for i, (start, end) in enumerate(hiatus_periods, 1):
            print(f"  Hiatus {i}: {start:.1f} - {end:.1f} ka BP")
    else:
        print("  Hiatus topilmadi")
    print()
    
    # 3. Rasmni yaratish
    print("🎨 Publication figure yaratilmoqda...")
    fig = create_publication_figure(df, hiatus_periods)
    print("✅ Rasm yaratildi")
    print()
    
    # 4. Saqlash
    print("💾 Fayllarni saqlash...")
    for fmt, filename in OUTPUT_FILES.items():
        fig.savefig(filename, dpi=DPI, bbox_inches='tight')
        print(f"  ✅ {filename}")
    print()
    
    # 5. Ko'rsatish
    print("📊 Rasmni ko'rsatish...")
    plt.show()
    
    print()
    print("=" * 60)
    print("✨ TAYYOR! Rasmlar muvaffaqiyatli yaratildi!")
    print("=" * 60)
    print()
    print("📋 Keyingi qadamlar:")
    print("  1. Rasmlarni tekshiring")
    print("  2. Annotatsiya pozitsiyalarini sozlang (agar kerak bo'lsa)")
    print("  3. Maqolaga qo'shing")
    print()

if __name__ == "__main__":
    main()

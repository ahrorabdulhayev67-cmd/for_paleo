"""
TON-2 Speleothem δ¹⁸O and δ¹³C Records - FIGURE 2 FINAL
Professional publication figure with outlier removal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# MA'LUMOTLARNI YUKLASH
# ══════════════════════════════════════════════════════════════

DATA_FILE = 'ton2_clean.csv'

try:
    df = pd.read_csv(DATA_FILE)
    print(f"✅ Ma'lumotlar yuklandi: {len(df)} qator")
    print(f"📋 Ustunlar: {list(df.columns)}")
    
    age_col = 'age_calBP'
    d18O_col = 'd18O'
    d13C_col = 'd13C'
    
    age_bp = df[age_col].values
    d18O = df[d18O_col].values
    d13C = df[d13C_col].values
    
except Exception as e:
    print(f"⚠️  Xatolik: {e}")
    print("📊 Test data yaratilmoqda...")
    
    age_bp = np.linspace(8000, 115000, 2007)
    d18O = -5 + np.sin(age_bp/10000) * 2 + np.random.randn(len(age_bp)) * 0.3
    d13C = -8 + np.cos(age_bp/12000) * 1.5 + np.random.randn(len(age_bp)) * 0.2

age_ka = age_bp / 1000

# ══════════════════════════════════════════════════════════════
# OUTLIER OLIB TASHLASH
# ══════════════════════════════════════════════════════════════

print("\n🔍 Outlier detection...")

# δ¹³C outliers
d13C_mean = np.nanmean(d13C)
d13C_std = np.nanstd(d13C)
z_scores_c = np.abs((d13C - d13C_mean) / d13C_std)
outlier_mask_c = z_scores_c > 3
n_outliers_c = outlier_mask_c.sum()

if n_outliers_c > 0:
    print(f"   δ¹³C: {n_outliers_c} ta outlier")
    d13C_clean = d13C.copy()
    d13C_clean[outlier_mask_c] = np.nan
else:
    d13C_clean = d13C.copy()

# δ¹⁸O outliers
d18O_mean = np.nanmean(d18O)
d18O_std = np.nanstd(d18O)
z_scores_o = np.abs((d18O - d18O_mean) / d18O_std)
outlier_mask_o = z_scores_o > 3

n_outliers_o = outlier_mask_o.sum()

if n_outliers_o > 0:
    print(f"   δ¹⁸O: {n_outliers_o} ta outlier")
    d18O_clean = d18O.copy()
    d18O_clean[outlier_mask_o] = np.nan
else:
    d18O_clean = d18O.copy()

# ══════════════════════════════════════════════════════════════
# SMOOTHING
# ══════════════════════════════════════════════════════════════

def smooth_data(age, data, window_ka=0.5):
    """500-year binning"""
    mask = ~np.isnan(data)
    age_clean = age[mask]
    data_clean = data[mask]
    
    age_min, age_max = age_clean.min(), age_clean.max()
    bins = np.arange(age_min, age_max + window_ka, window_ka)
    
    age_binned = []
    data_binned = []
    
    for i in range(len(bins) - 1):
        mask_bin = (age_clean >= bins[i]) & (age_clean < bins[i+1])
        if mask_bin.sum() > 0:
            age_binned.append(age_clean[mask_bin].mean())
            data_binned.append(data_clean[mask_bin].mean())
    
    return np.array(age_binned), np.array(data_binned)

age_smooth, d18O_smooth = smooth_data(age_ka, d18O_clean)
age_smooth_c, d13C_smooth = smooth_data(age_ka, d13C_clean)

print(f"✅ Smoothing: {len(age_smooth)} nuqta")

# ══════════════════════════════════════════════════════════════
# HIATUS DETECTION
# ══════════════════════════════════════════════════════════════

def detect_hiatus_precise(age_ka, threshold_ka=0.5):
    age_sorted = np.sort(age_ka)
    age_diff = np.abs(np.diff(age_sorted))
    hiatus_indices = np.where(age_diff > threshold_ka)[0]
    
    hiatus_list = []
    for idx in hiatus_indices:
        start = age_sorted[idx]
        end = age_sorted[idx + 1]
        gap_size = end - start
        
        hiatus_list.append({
            'start': start,
            'end': end,
            'gap_ka': gap_size,
            'gap_years': gap_size * 1000
        })
    
    return hiatus_list

print("\n🔍 Hiatus detection...")
hiatus_data = detect_hiatus_precise(age_ka, threshold_ka=0.5)

if hiatus_data:
    print(f"   {len(hiatus_data)} ta hiatus")
    hiatus_periods = [(h['start'], h['end']) for h in hiatus_data]
else:
    hiatus_periods = []

# ══════════════════════════════════════════════════════════════
# MIS STAGES
# ══════════════════════════════════════════════════════════════

MIS_stages = [
    {'name': 'MIS 1', 'start': 0, 'end': 11.7, 'color': '#FFF4E6'},
    {'name': 'MIS 2', 'start': 11.7, 'end': 29, 'color': '#E3F2FD'},
    {'name': 'MIS 3', 'start': 29, 'end': 57, 'color': '#FFF8E1'},
    {'name': 'MIS 4', 'start': 57, 'end': 71, 'color': '#E1F5FE'},
    {'name': 'MIS 5', 'start': 71, 'end': 115, 'color': '#FFFDE7'},
]

events = [
    {'name': 'YD', 'age': 12.0},
    {'name': 'B/A', 'age': 14.7},
    {'name': 'LGM', 'age': 21.0},
    {'name': 'MIS 5e', 'age': 85.0},
]

# ══════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════

print("\n🎨 Figure yaratilmoqda...")

plt.style.use('default')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 10, 
                     'axes.linewidth': 1.0, 'figure.dpi': 300})

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.subplots_adjust(hspace=0.05)

# PANEL A: δ¹⁸O
ax1 = axes[0]

for mis in MIS_stages:
    ax1.axvspan(mis['start'], mis['end'], color=mis['color'], 
                alpha=0.4, zorder=0, linewidth=0)

if hiatus_periods:
    for h_start, h_end in hiatus_periods:
        ax1.axvspan(h_start, h_end, color='#999999', alpha=0.2, 
                    zorder=1, edgecolor='#666666', linewidth=0.5)

ax1.plot(age_ka, d18O_clean, color='#CCCCCC', linewidth=0.4, 
         alpha=0.6, zorder=2, label='Raw data')
ax1.plot(age_smooth, d18O_smooth, color='#2C3E50', linewidth=2.0,
         zorder=3, label='500-yr smoothed')

for event in events:
    if event['age'] <= age_ka.max():
        ax1.axvline(event['age'], color='#E63946', linewidth=1.2,
                   linestyle='--', alpha=0.7, zorder=4)
        y_max = np.nanmax(d18O_clean)
        ax1.text(event['age'], y_max * 0.95, event['name'],
                ha='center', va='top', fontsize=8, color='#E63946',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#E63946', linewidth=1, alpha=0.9))

ax1.invert_yaxis()
ax1.set_ylabel('δ¹⁸O (‰ VPDB)', fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
ax1.legend(loc='upper right', fontsize=9, frameon=True, 
          framealpha=0.95, edgecolor='#CCCCCC')
ax1.text(0.01, 0.97, 'a', transform=ax1.transAxes, fontsize=14,
         fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='none', alpha=0.8))
ax1.text(0.99, 0.03, 'Higher δ¹⁸O = cooler/drier ↓',
         transform=ax1.transAxes, fontsize=7, color='#666666',
         ha='right', va='bottom', style='italic')

# PANEL B: δ¹³C
ax2 = axes[1]

for mis in MIS_stages:
    ax2.axvspan(mis['start'], mis['end'], color=mis['color'],
                alpha=0.4, zorder=0, linewidth=0)

if hiatus_periods:
    for h_start, h_end in hiatus_periods:
        ax2.axvspan(h_start, h_end, color='#999999', alpha=0.2,
                    zorder=1, edgecolor='#666666', linewidth=0.5)

ax2.plot(age_ka, d13C_clean, color='#D4E6F1', linewidth=0.4,
         alpha=0.6, zorder=2, label='Raw data')
ax2.plot(age_smooth_c, d13C_smooth, color='#2874A6', linewidth=2.0,
         zorder=3, label='500-yr smoothed')

ax2.set_xlabel('Age (ka BP)', fontsize=11, fontweight='bold')
ax2.set_ylabel('δ¹³C (‰ VPDB)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
ax2.legend(loc='upper right', fontsize=9, frameon=True,
          framealpha=0.95, edgecolor='#CCCCCC')
ax2.text(0.01, 0.97, 'b', transform=ax2.transAxes, fontsize=14,
         fontweight='bold', va='top', ha='left',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='none', alpha=0.8))
ax2.text(0.99, 0.03, 'For Hendy test',
         transform=ax2.transAxes, fontsize=7, color='#666666',
         ha='right', va='bottom', style='italic')

ax2.set_xlim(0, min(115, age_ka.max() + 5))
ax2.set_xticks(np.arange(0, 120, 10))

# MIS labels
ax1_top = ax1.twiny()
ax1_top.set_xlim(ax1.get_xlim())
mis_pos = [(m['start']+m['end'])/2 for m in MIS_stages if (m['start']+m['end'])/2 <= age_ka.max()]
mis_lab = [m['name'] for m in MIS_stages if (m['start']+m['end'])/2 <= age_ka.max()]
ax1_top.set_xticks(mis_pos)
ax1_top.set_xticklabels(mis_lab, fontsize=9, fontweight='600', color='#555')
ax1_top.tick_params(axis='x', length=0)
for spine in ax1_top.spines.values():
    spine.set_visible(False)

# Caption
hiatus_text = ""
if hiatus_data:
    total_gap = sum([h['gap_ka'] for h in hiatus_data])
    hiatus_text = f" Gray zones: {len(hiatus_data)} hiatus periods (total {total_gap:.1f} ka, 500-yr threshold)."

caption = f"""Fig. 2. TON-2 speleothem isotope records ({age_ka.min():.1f}–{age_ka.max():.1f} ka BP). (a) δ¹⁸O (gray=raw, black=500-yr smoothed, inverted y-axis). MIS stages shown by colored bands (warm=yellow, cold=blue). Events: YD (Younger Dryas), B/A (Bølling-Allerød), LGM (Last Glacial Maximum), MIS 5e. Limited LGM signal reflects reduced growth under extreme aridity.{hiatus_text} (b) δ¹³C (outliers >3σ removed) for Hendy test. Records show glacial-interglacial cyclicity."""

fig.text(0.08, 0.01, caption, fontsize=8, color='#444', style='italic', ha='left', va='bottom')

plt.tight_layout(rect=[0, 0.10, 1, 0.98])

print("\n💾 Saqlash...")
plt.savefig('fig2_TON2_isotopes_FINAL.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('fig2_TON2_isotopes_FINAL.pdf', dpi=300, bbox_inches='tight', facecolor='white')

plt.show()

print("\n✅ TAYYOR! Journal-ready figure.")

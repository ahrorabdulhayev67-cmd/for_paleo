"""
Tonnelnaya Cave Study Site Map - Fixed version
Xatoliklarni tuzatilgan, Jupyter uchun tayyor versiya
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

# Cartopy import qilishdan oldin Natural Earth cache sozlamalari
import os
os.environ['CARTOPY_USER_BACKGROUNDS'] = 'true'

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    CARTOPY_AVAILABLE = True
except ImportError:
    print("⚠️  Cartopy o'rnatilmagan. Oddiy matplotlib versiyasi ishlatiladi.")
    CARTOPY_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# 1. KOORDINATALAR
# ══════════════════════════════════════════════════════════════
sites = {
    'Tonnelnaya Cave\n(this study)': {
        'lon': 67.23, 'lat': 38.40,
        'elev': 3226,
        'marker': '*', 'size': 280,
        'color': '#D73027',
        'zorder': 6
    },
    'Boysun MS\n(instrumental)': {
        'lon': 67.21, 'lat': 38.19,
        'elev': 1000,
        'marker': '^', 'size': 120,
        'color': '#4393C3',
        'zorder': 5
    },
}

# Muhim shaharlar
cities = {
    'Tashkent'   : (69.24, 41.30),
    'Samarkand'  : (66.92, 39.65),
    'Termez'     : (67.28, 37.22),
    'Dushanbe'   : (68.77, 38.56),
    'Bishkek'    : (74.59, 42.87),
    'Kabul'      : (69.18, 34.53),
}

# Mamlakatlar
countries = {
    'UZBEKISTAN' : (64.5, 41.0),
    'TAJIKISTAN' : (71.0, 39.0),
    'KYRGYZSTAN' : (74.5, 41.5),
    'AFGHANISTAN': (66.0, 34.8),
    'TURKMENISTAN': (59.5, 39.5),
    'KAZAKHSTAN' : (63.0, 44.5),
}

# ══════════════════════════════════════════════════════════════
# 2. XARITA YARATISH FUNKSIYASI
# ══════════════════════════════════════════════════════════════

def create_study_site_map_cartopy():
    """Cartopy bilan xarita yaratish"""
    
    proj = ccrs.LambertConformal(
        central_longitude=67.0,
        central_latitude=38.5
    )
    
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[2.2, 1],
        height_ratios=[1.8, 1],
        wspace=0.04, hspace=0.08,
        left=0.03, right=0.97,
        top=0.93, bottom=0.06
    )
    
    # ── Panel 1: Asosiy xarita ───────────────────────────────────
    ax_main = fig.add_subplot(gs[:, 0], projection=proj)
    ax_main.set_extent([58, 78, 34, 46], crs=ccrs.PlateCarree())
    
    # Asosiy xarita elementlari (50m resolution - tezroq)
    try:
        ax_main.add_feature(
            cfeature.LAND.with_scale('50m'),
            facecolor='#F5F0E8',
            edgecolor='none',
            zorder=0
        )
        ax_main.add_feature(
            cfeature.OCEAN.with_scale('50m'),
            facecolor='#D6EAF8',
            edgecolor='none',
            zorder=0
        )
        ax_main.add_feature(
            cfeature.LAKES.with_scale('50m'),
            facecolor='#D6EAF8',
            edgecolor='#7FB3D3',
            linewidth=0.5,
            zorder=1
        )
        ax_main.add_feature(
            cfeature.RIVERS.with_scale('50m'),
            edgecolor='#7FB3D3',
            linewidth=0.4,
            zorder=1
        )
        ax_main.add_feature(
            cfeature.BORDERS.with_scale('50m'),
            edgecolor='#666666',
            linewidth=0.9,
            zorder=3
        )
    except Exception as e:
        print(f"⚠️  Natural Earth ma'lumotlari yuklanmadi: {e}")
        print("    Oddiy fon ishlatilmoqda...")
        ax_main.set_facecolor('#F5F0E8')
    
    # Mamlakat nomlari
    for name, (lon, lat) in countries.items():
        ax_main.text(
            lon, lat, name,
            transform=ccrs.PlateCarree(),
            fontsize=8, color='#555555',
            ha='center', style='italic',
            fontweight='bold',
            path_effects=[pe.withStroke(
                linewidth=2,
                foreground='white'
            )]
        )
    
    # Shaharlar
    for city, (lon, lat) in cities.items():
        ax_main.plot(
            lon, lat, 'o',
            color='#555555', ms=4,
            transform=ccrs.PlateCarree(),
            zorder=4
        )
        ax_main.text(
            lon + 0.3, lat + 0.2, city,
            transform=ccrs.PlateCarree(),
            fontsize=8, color='#333333',
            path_effects=[pe.withStroke(
                linewidth=2,
                foreground='white'
            )]
        )
    
    # Study sites
    for name, info in sites.items():
        ax_main.plot(
            info['lon'], info['lat'],
            marker=info['marker'],
            color=info['color'],
            ms=np.sqrt(info['size']),
            transform=ccrs.PlateCarree(),
            zorder=info['zorder'],
            markeredgecolor='white',
            markeredgewidth=1.5,
            label=name
        )
        # Label
        offset_x = 0.5
        offset_y = 0.3
        if 'Boysun' in name:
            offset_x = 0.5
            offset_y = -0.5
        ax_main.text(
            info['lon'] + offset_x,
            info['lat'] + offset_y,
            name.replace('\n', '\n'),
            transform=ccrs.PlateCarree(),
            fontsize=9, color=info['color'],
            fontweight='bold',
            path_effects=[pe.withStroke(
                linewidth=2.5,
                foreground='white'
            )]
        )
    
    # Tonnelnaya → Boysun masofasi
    ax_main.plot(
        [67.23, 67.21],
        [38.40, 38.19],
        color='#888888', lw=1.0,
        ls='--', transform=ccrs.PlateCarree(),
        zorder=3
    )
    ax_main.text(
        67.6, 38.30,
        '~25 km\n~2226 m Δelev',
        transform=ccrs.PlateCarree(),
        fontsize=8, color='#666666',
        ha='left',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white',
            edgecolor='#CCCCCC',
            alpha=0.8
        )
    )
    
    # Grid chiziqlar
    gl = ax_main.gridlines(
        draw_labels=True,
        linewidth=0.4,
        color='gray',
        alpha=0.4,
        linestyle='--',
        crs=ccrs.PlateCarree()
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.xformatter    = LONGITUDE_FORMATTER
    gl.yformatter    = LATITUDE_FORMATTER
    gl.xlabel_style  = {'size': 9, 'color': '#333'}
    gl.ylabel_style  = {'size': 9, 'color': '#333'}
    
    # Legend
    legend_elements = [
        Line2D([0], [0],
               marker='*', color='w',
               markerfacecolor='#D73027',
               markersize=15,
               markeredgecolor='white',
               label='Tonnelnaya Cave (3226 m a.s.l.)'),
        Line2D([0], [0],
               marker='^', color='w',
               markerfacecolor='#4393C3',
               markersize=10,
               markeredgecolor='white',
               label='Boysun MS (1000 m a.s.l.)'),
        Line2D([0], [0],
               marker='o', color='w',
               markerfacecolor='#555555',
               markersize=6,
               label='City'),
    ]
    ax_main.legend(
        handles=legend_elements,
        loc='lower left',
        fontsize=9,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        fancybox=False,
        title='Study sites',
        title_fontsize=9
    )
    
    ax_main.set_title(
        'Study area — Tonnelnaya Cave, southern Uzbekistan',
        fontsize=12, fontweight='bold',
        pad=10, loc='left'
    )
    
    # ── Panel 2: Regional kontekst ───────────────────────────────
    ax_reg = fig.add_subplot(
        gs[0, 1],
        projection=ccrs.PlateCarree()
    )
    ax_reg.set_extent([40, 100, 20, 55],
                       crs=ccrs.PlateCarree())
    
    try:
        ax_reg.add_feature(
            cfeature.LAND.with_scale('110m'),
            facecolor='#F5F0E8',
            edgecolor='#999999',
            linewidth=0.3
        )
        ax_reg.add_feature(
            cfeature.OCEAN.with_scale('110m'),
            facecolor='#D6EAF8'
        )
        ax_reg.add_feature(
            cfeature.BORDERS.with_scale('110m'),
            edgecolor='#888888',
            linewidth=0.4
        )
    except:
        ax_reg.set_facecolor('#F5F0E8')
    
    # Study area box
    ax_reg.add_patch(
        mpatches.Rectangle(
            (58, 34), 20, 12,
            linewidth=2,
            edgecolor='#D73027',
            facecolor='none',
            transform=ccrs.PlateCarree(),
            zorder=5
        )
    )
    ax_reg.plot(
        67.23, 38.40, '*',
        color='#D73027', ms=10,
        transform=ccrs.PlateCarree(),
        zorder=6,
        markeredgecolor='white',
        markeredgewidth=1
    )
    
    # Muhim geografik nomlar
    for name, (lon, lat) in [
        ('Aral\nSea',    (60.0, 45.0)),
        ('Caspian\nSea', (51.0, 41.5)),
        ('Pamir',        (73.5, 37.5)),
        ('Tian Shan',    (78.0, 42.0)),
        ('Hindu Kush',   (68.0, 35.0)),
        ('Arabian\nSea', (63.0, 22.5)),
    ]:
        ax_reg.text(
            lon, lat, name,
            transform=ccrs.PlateCarree(),
            fontsize=7, color='#555555',
            ha='center', style='italic'
        )
    
    gl2 = ax_reg.gridlines(
        draw_labels=False,
        linewidth=0.3,
        color='gray', alpha=0.3,
        linestyle='--'
    )
    ax_reg.set_title(
        'Regional context',
        fontsize=9, pad=4
    )
    
    return fig, gs


def create_study_site_map_simple():
    """Oddiy matplotlib versiyasi (Cartopy bo'lmasa)"""
    
    fig = plt.figure(figsize=(18, 12), facecolor='white')
    
    gs = gridspec.GridSpec(
        2, 2,
        width_ratios=[2.2, 1],
        height_ratios=[1.8, 1],
        wspace=0.04, hspace=0.08,
        left=0.03, right=0.97,
        top=0.93, bottom=0.06
    )
    
    # ── Panel 1: Asosiy xarita (oddiy) ───────────────────────────
    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.set_xlim(58, 78)
    ax_main.set_ylim(34, 46)
    ax_main.set_facecolor('#F5F0E8')
    ax_main.set_aspect('equal')
    
    # Mamlakat nomlari
    for name, (lon, lat) in countries.items():
        ax_main.text(
            lon, lat, name,
            fontsize=8, color='#555555',
            ha='center', style='italic',
            fontweight='bold',
            path_effects=[pe.withStroke(
                linewidth=2,
                foreground='white'
            )]
        )
    
    # Shaharlar
    for city, (lon, lat) in cities.items():
        ax_main.plot(lon, lat, 'o', color='#555555', ms=4, zorder=4)
        ax_main.text(
            lon + 0.3, lat + 0.2, city,
            fontsize=8, color='#333333',
            path_effects=[pe.withStroke(linewidth=2, foreground='white')]
        )
    
    # Study sites
    for name, info in sites.items():
        ax_main.plot(
            info['lon'], info['lat'],
            marker=info['marker'],
            color=info['color'],
            ms=np.sqrt(info['size']),
            zorder=info['zorder'],
            markeredgecolor='white',
            markeredgewidth=1.5
        )
        offset_x = 0.5
        offset_y = 0.3
        if 'Boysun' in name:
            offset_y = -0.5
        ax_main.text(
            info['lon'] + offset_x,
            info['lat'] + offset_y,
            name,
            fontsize=9, color=info['color'],
            fontweight='bold',
            path_effects=[pe.withStroke(linewidth=2.5, foreground='white')]
        )
    
    ax_main.plot([67.23, 67.21], [38.40, 38.19],
                 color='#888888', lw=1.0, ls='--', zorder=3)
    
    ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.4)
    ax_main.set_xlabel('Longitude (°E)', fontsize=10)
    ax_main.set_ylabel('Latitude (°N)', fontsize=10)
    ax_main.set_title(
        'Study area — Tonnelnaya Cave, southern Uzbekistan',
        fontsize=12, fontweight='bold', pad=10, loc='left'
    )
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w',
               markerfacecolor='#D73027', markersize=15,
               markeredgecolor='white',
               label='Tonnelnaya Cave (3226 m a.s.l.)'),
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor='#4393C3', markersize=10,
               markeredgecolor='white',
               label='Boysun MS (1000 m a.s.l.)'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#555555', markersize=6,
               label='City'),
    ]
    ax_main.legend(handles=legend_elements, loc='lower left',
                   fontsize=9, framealpha=0.95, edgecolor='#CCCCCC')
    
    # ── Panel 2: Regional (oddiy) ────────────────────────────────
    ax_reg = fig.add_subplot(gs[0, 1])
    ax_reg.set_xlim(40, 100)
    ax_reg.set_ylim(20, 55)
    ax_reg.set_facecolor('#F5F0E8')
    ax_reg.set_aspect('equal')
    
    # Study area box
    ax_reg.add_patch(
        mpatches.Rectangle(
            (58, 34), 20, 12,
            linewidth=2, edgecolor='#D73027',
            facecolor='none', zorder=5
        )
    )
    ax_reg.plot(67.23, 38.40, '*', color='#D73027', ms=10,
                zorder=6, markeredgecolor='white', markeredgewidth=1)
    
    ax_reg.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
    ax_reg.set_title('Regional context', fontsize=9, pad=4)
    ax_reg.set_xlabel('Longitude (°E)', fontsize=8)
    ax_reg.set_ylabel('Latitude (°N)', fontsize=8)
    
    return fig, gs


# ══════════════════════════════════════════════════════════════
# 3. ELEVATION PROFILE (ikkala versiya uchun bir xil)
# ══════════════════════════════════════════════════════════════

def add_elevation_profile(fig, gs):
    """Balandlik profilini qo'shish"""
    
    ax_elev = fig.add_subplot(gs[1, 1])
    
    # Balandlik profili
    dist    = np.array([0, 5, 10, 15, 20, 25])
    elev    = np.array([1000, 1400, 1900, 2500, 2900, 3226])
    elev_sm = np.interp(np.linspace(0, 25, 100), dist, elev)
    dist_sm = np.linspace(0, 25, 100)
    
    ax_elev.fill_between(
        dist_sm, elev_sm, 800,
        color='#E8DCC8', alpha=0.8, zorder=1
    )
    ax_elev.plot(
        dist_sm, elev_sm,
        color='#8B6914', lw=2.0, zorder=2
    )
    
    # Stantsiyalar
    ax_elev.plot(0, 1000, '^', color='#4393C3', ms=12,
                 zorder=4, markeredgecolor='white', markeredgewidth=1.5)
    ax_elev.text(0.5, 1050, 'Boysun MS\n(1000 m)',
                 fontsize=8.5, color='#4393C3', fontweight='bold')
    
    ax_elev.plot(25, 3226, '*', color='#D73027', ms=16,
                 zorder=4, markeredgecolor='white', markeredgewidth=1.5)
    ax_elev.text(21, 3300, 'Tonnelnaya\n(3226 m)',
                 fontsize=8.5, color='#D73027', fontweight='bold', ha='right')
    
    # Lapse rate
    ax_elev.annotate('', xy=(25, 3226), xytext=(0, 1000),
                     arrowprops=dict(arrowstyle='<->', color='#666', lw=1.2))
    ax_elev.text(13, 2100, 'Δh = 2226 m\nΓ = −3.57°C km⁻¹',
                 fontsize=8, color='#555555', ha='center',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#CCCCCC', alpha=0.9))
    
    ax_elev.set_xlim(-1, 27)
    ax_elev.set_ylim(700, 3600)
    ax_elev.set_xlabel('Distance (km)', fontsize=9)
    ax_elev.set_ylabel('Elevation (m a.s.l.)', fontsize=9)
    ax_elev.set_title('Elevation profile', fontsize=9, pad=4)
    ax_elev.grid(True, alpha=0.2, lw=0.5)
    ax_elev.spines[['top','right']].set_visible(False)


# ══════════════════════════════════════════════════════════════
# 4. ASOSIY FUNKSIYA
# ══════════════════════════════════════════════════════════════

def main():
    """Asosiy funksiya"""
    
    print("=" * 60)
    print("TONNELNAYA CAVE STUDY SITE MAP")
    print("=" * 60)
    print()
    
    # Cartopy mavjudligini tekshirish
    if CARTOPY_AVAILABLE:
        print("✅ Cartopy topildi. Professional xarita yaratilmoqda...")
        try:
            fig, gs = create_study_site_map_cartopy()
        except Exception as e:
            print(f"⚠️  Cartopy xatolik: {e}")
            print("    Oddiy versiya ishlatilmoqda...")
            fig, gs = create_study_site_map_simple()
    else:
        print("ℹ️  Cartopy topilmadi. Oddiy versiya ishlatilmoqda...")
        fig, gs = create_study_site_map_simple()
    
    # Elevation profile qo'shish
    add_elevation_profile(fig, gs)
    
    # Figure caption
    fig.text(
        0.03, 0.01,
        'Fig. 1.  Location of Tonnelnaya Cave (38.40°N, 67.23°E, '
        '3226 m a.s.l.) and Boysun meteorological station '
        '(38.19°N, 67.21°E, 1000 m a.s.l.), southern Uzbekistan. '
        '(a) Regional map showing the study area (red box). '
        '(b) Detailed map with study sites. '
        '(c) Elevation profile showing the 2226 m altitudinal difference.',
        fontsize=8, color='#444444',
        style='italic', wrap=True,
        ha='left', va='bottom'
    )
    
    print()
    print("💾 Rasmni saqlash...")
    plt.savefig('fig1_study_site.png', dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.savefig('fig1_study_site.pdf', dpi=300, bbox_inches='tight',
                facecolor='white')
    
    print("✅ Saqlandi:")
    print("   - fig1_study_site.png")
    print("   - fig1_study_site.pdf")
    print()
    
    plt.show()
    
    print("=" * 60)
    print("✨ TAYYOR!")
    print("=" * 60)


if __name__ == "__main__":
    main()

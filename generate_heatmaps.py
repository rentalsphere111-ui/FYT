import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import seaborn as sns

# 1. Load Data
print("Loading shot data...")
df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")

# Filter for Direct Free Kicks
df_dfk = df[df['situation'] == 'DirectFreekick'].copy()

# Filter for the Start vs End of the Decade
df_2014 = df_dfk[df_dfk['season'] == '2014/2015']
df_2023 = df_dfk[df_dfk['season'] == '2023/2024']

print(f"2014/15 DFK Shots: {len(df_2014)}")
print(f"2023/24 DFK Shots: {len(df_2023)}")

# 3. Setup the Pitch
# We create a simple figure with 2 subplots side-by-side
pitch = VerticalPitch(
    pitch_type='opta', 
    pitch_color='#f8f9fa', 
    line_color='#2c3e50',
    line_zorder=2,
    half=True # Only show attacking half
)

# grid() returns a dictionary of axes. 'pitch' contains the plotting areas.
fig, axs = pitch.grid(nrows=1, ncols=2, title_height=0.08, figheight=10, grid_height=0.82)

# --- Plot 1: 2014/15 (Red KDE) ---
pitch.kdeplot(
    df_2014['X'] * 100, 
    df_2014['Y'] * 100, 
    ax=axs['pitch'][0],  # Left Plot
    fill=True,
    levels=100, 
    cmap='Reds', 
    alpha=0.85,
    thresh=0.01
)
# Add scatter points to show exact DFK locations
pitch.scatter(
    df_2014['X'] * 100, 
    df_2014['Y'] * 100, 
    ax=axs['pitch'][0],
    color='#c0392b',
    edgecolors='white',
    s=25,
    alpha=0.6,
    zorder=3
)
axs['pitch'][0].set_title("2014/15 Season\n(441 Attempts | 27 Goals)", fontsize=16, fontweight='bold', pad=15, color='#2c3e50')

# --- Plot 2: 2023/24 (Blue KDE) ---
pitch.kdeplot(
    df_2023['X'] * 100, 
    df_2023['Y'] * 100, 
    ax=axs['pitch'][1],  # Right Plot
    fill=True,
    levels=100, 
    cmap='Blues', 
    alpha=0.85,
    thresh=0.01
)
pitch.scatter(
    df_2023['X'] * 100, 
    df_2023['Y'] * 100, 
    ax=axs['pitch'][1],
    color='#2980b9',
    edgecolors='white',
    s=25,
    alpha=0.6,
    zorder=3
)
axs['pitch'][1].set_title("2023/24 Season\n(282 Attempts | 11 Goals)", fontsize=16, fontweight='bold', pad=15, color='#2c3e50')

# Title
fig.suptitle("Spatial Density Comparison of Direct Free Kicks\nPremier League: 2014/15 vs 2023/24", fontsize=18, fontweight='bold', color='#1a252f', y=0.98)

# 4. Save
plt.savefig("Dissertation_Chart_4_HeatmapComparison.png", dpi=300, bbox_inches='tight')
print("\nSUCCESS! Heatmap saved as 'Dissertation_Chart_4_HeatmapComparison.png'")
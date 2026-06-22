import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

print("--- Loading Shot Data ---")
df = pd.read_csv("Premier_League_Shots_14_24_Direct.csv")

# Filter for Direct Free Kicks
df_dfk = df[df['situation'] == 'DirectFreekick'].copy()
df_dfk['is_goal'] = df_dfk['result'].apply(lambda x: 1 if x == 'Goal' else 0)

# Calculate shot coordinates in meters (pitch is 105m x 68m in Understat format X, Y)
df_dfk['x_m'] = df_dfk['X'] * 105
df_dfk['y_m'] = df_dfk['Y'] * 68
df_dfk['distance'] = np.sqrt((105 - df_dfk['x_m'])**2 + (34 - df_dfk['y_m'])**2)

# Summarize by season
summary = df_dfk.groupby('season').agg(
    total_shots=('result', 'count'),
    total_goals=('is_goal', 'sum'),
    avg_distance=('distance', 'mean'),
    conversion_rate=('is_goal', lambda x: x.mean() * 100)
).reset_index()

print("--- Free Kick Summary by Season ---")
print(summary)

# Plot: 10-Year Decline of Direct Free Kick Attempts and Goals
fig, ax1 = plt.subplots(figsize=(10, 6))

color = '#1f77b4' # Muted premium blue
ax1.set_xlabel('Season', fontweight='bold', labelpad=10)
ax1.set_ylabel('DFK Attempts', color=color, fontweight='bold')
bars = ax1.bar(summary['season'], summary['total_shots'], color=color, alpha=0.3, label='DFK Attempts')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticklabels(summary['season'], rotation=45)

# Add line for goals on a dual axis
ax2 = ax1.twinx()
color = '#d62728' # Muted premium red
ax2.set_ylabel('DFK Goals Scored', color=color, fontweight='bold')
line = ax2.plot(summary['season'], summary['total_goals'], color=color, marker='o', linewidth=3, markersize=8, label='DFK Goals')
ax2.tick_params(axis='y', labelcolor=color)

# Add values on top of the bars and lines
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=9, color='#1f77b4', fontweight='semibold')

for x, y in zip(summary['season'], summary['total_goals']):
    ax2.annotate(f'{y}',
                xy=(x, y),
                xytext=(0, 7),  # 7 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, color='#d62728', fontweight='bold')

plt.title('The Decline of Direct Free-Kick (DFK) Attempts & Goals\nPremier League (2014 - 2024)', fontsize=15, fontweight='bold', pad=15)
fig.tight_layout()

# Save the chart
plt.savefig("Dissertation_Chart_1_Decline.png", dpi=300, bbox_inches='tight')
print("\nChart saved as 'Dissertation_Chart_1_Decline.png'")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Figure and axis setup
fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_aspect('equal')
ax.set_xlabel('Die 1 (first roll)')
ax.set_ylabel('Die 2 (second roll)')
ax.set_xticks(np.arange(1, 7))
ax.set_yticks(np.arange(1, 7))
ax.set_title('Counting with Two Dice (Multiplication Principle)')

# Light background
ax.add_patch(Rectangle((0.5, 0.5), 6, 6, facecolor='white', edgecolor='none', zorder=0))

# Highlight one column (fix Die 1 = 3) and one row (fix Die 2 = 4)
col_x = 3
row_y = 4
ax.add_patch(Rectangle((col_x - 0.5, 0.5), 1, 6, facecolor=(0.65, 0.82, 1.0), alpha=0.6, edgecolor='none', zorder=1))
ax.add_patch(Rectangle((0.5, row_y - 0.5), 6, 1, facecolor=(0.70, 1.0, 0.70), alpha=0.55, edgecolor='none', zorder=1))

# Grid lines
for i in range(7):
    # verticals
    ax.plot([0.5 + i, 0.5 + i], [0.5, 6.5], color='0.7', lw=1)
    # horizontals
    ax.plot([0.5, 6.5], [0.5 + i, 0.5 + i], color='0.7', lw=1)

# Bold outer border
ax.add_patch(Rectangle((0.5, 0.5), 6, 6, fill=False, ec='0.2', lw=2))

# Label each ordered outcome (i, j)
for x in range(1, 7):
    for y in range(1, 7):
        ax.text(x, y, f"({x},{y})", ha='center', va='center', fontsize=8.2, color='0.35', zorder=3)

# Emphasize the intersection (3,4) as one specific combined outcome
ax.add_patch(Rectangle((col_x - 0.5, row_y - 0.5), 1, 1, fill=False, ec='crimson', lw=2.2, zorder=4))
ax.plot([col_x], [row_y], marker='o', color='crimson', ms=6, zorder=5)

# Annotations explaining the multiplication principle
# Column (fix Die 1)
ax.text(col_x, 6.72, 'Fix Die 1 = 3 → 6 choices for Die 2', ha='center', va='bottom', fontsize=10, color='tab:blue')
ax.annotate('', xy=(col_x, 6.48), xytext=(col_x, 0.52),
            arrowprops=dict(arrowstyle='-|>', color='tab:blue', lw=2), zorder=6)

# Row (fix Die 2)
ax.text(0.55, row_y + 0.42, 'Fix Die 2 = 4 → 6 choices for Die 1', ha='left', va='bottom', fontsize=10, color='darkgreen')
ax.annotate('', xy=(6.48, row_y), xytext=(0.52, row_y),
            arrowprops=dict(arrowstyle='-|>', color='darkgreen', lw=2), zorder=6)

# Annotation for one specific outcome
ax.annotate('One specific combined outcome', xy=(col_x, row_y), xytext=(4.9, 1.3),
            textcoords='data', color='crimson', fontsize=10,
            arrowprops=dict(arrowstyle='->', color='crimson', lw=1.6))

# Summary box
summary_text = 'Total outcomes = 6 × 6 = 36\nProbability of any specific outcome = 1/36'
ax.text(0.52, 6.9, summary_text, ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle='round,pad=0.4', fc='w', ec='0.4'))

# Tight layout and save
fig.subplots_adjust(top=0.88, right=0.96, left=0.12, bottom=0.12)
output_filename = 'dice_multiplication_principle.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f'Figure saved as {output_filename}')

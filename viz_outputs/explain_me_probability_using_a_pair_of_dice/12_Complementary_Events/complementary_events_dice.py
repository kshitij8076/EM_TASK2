import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# Data setup: two fair dice (faces 1..6)
sides = np.arange(1, 7)
# s[i, j] = (die1 = i+1) + (die2 = j+1)
s = sides[:, None] + sides[None, :]
mask_sum7 = (s == 7).astype(int)  # 1 where sum=7, 0 otherwise

# Probabilities
favorable = int(mask_sum7.sum())        # 6
unfavorable = 36 - favorable             # 30
p_sum7 = favorable / 36.0                # 1/6
p_not7 = 1.0 - p_sum7                    # 5/6

# Colors
color_event = "#2ca02c"   # green for Sum=7
color_comp = "#d9d9d9"    # light gray for Not 7
cmap = ListedColormap([color_comp, color_event])

fig = plt.figure(figsize=(11, 5.5), dpi=200)

# Left subplot: Sample space grid highlighting complementary events
ax1 = fig.add_subplot(1, 2, 1)
# Create cell boundary grid for pcolormesh
x = np.arange(7)
y = np.arange(7)
X, Y = np.meshgrid(x, y)
# Draw the grid with categories
pc = ax1.pcolormesh(X, Y, mask_sum7, cmap=cmap, edgecolors='white', linewidth=1.5, shading='flat')
ax1.set_aspect('equal')
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 6)
ax1.set_xticks(np.arange(6) + 0.5)
ax1.set_yticks(np.arange(6) + 0.5)
ax1.set_xticklabels([str(i) for i in sides])
ax1.set_yticklabels([str(i) for i in sides])
ax1.set_xlabel('Die 2 face')
ax1.set_ylabel('Die 1 face')
ax1.set_title('Sample Space (36 outcomes) — Green cells: sum = 7', fontsize=11)

# Annotate the green cells with '7' to emphasize the event
for i in range(6):
    for j in range(6):
        if s[i, j] == 7:
            ax1.text(j + 0.5, i + 0.5, '7', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

# Legend for left plot
legend_patches = [
    mpatches.Patch(color=color_event, label=f'Sum = 7 (6 outcomes)'),
    mpatches.Patch(color=color_comp, label=f'Not 7 (30 outcomes)')
]
ax1.legend(handles=legend_patches, loc='upper right', frameon=True)

# Right subplot: Stacked bar to show complement adds to 1
ax2 = fig.add_subplot(1, 2, 2)
ax2.bar(0, p_sum7, width=0.6, color=color_event, label='Sum = 7')
ax2.bar(0, p_not7, bottom=p_sum7, width=0.6, color=color_comp, label='Not 7')
ax2.set_xlim(-0.8, 0.8)
ax2.set_ylim(0, 1.0)
ax2.set_xticks([])
ax2.set_ylabel('Probability')
ax2.set_title('Complementary probabilities add to 1', fontsize=11)

# Annotate segment values and fractions
ax2.text(0, p_sum7 / 2, '6/36 = 1/6', ha='center', va='center', color='white', fontsize=11, fontweight='bold')
ax2.text(0, p_sum7 + p_not7 / 2, '30/36 = 5/6', ha='center', va='center', color='black', fontsize=11, fontweight='bold')

# Equation text emphasizing complement rule
ax2.text(0, 1.03, 'P(A) + P(A^c) = 1', ha='center', va='bottom', fontsize=12)

# Subplot overall title
fig.suptitle('Complementary Events with Two Dice: A = "sum is 7", A^c = "sum is not 7"', fontsize=13)

fig.tight_layout(rect=[0, 0, 1, 0.92])

# Save figure to working directory
output_filename = 'complementary_events_dice.png'
plt.savefig(output_filename, dpi=200)
plt.close(fig)
print(f"Saved figure to {output_filename}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec

# Set up figure layout
fig = plt.figure(figsize=(10, 8))
# GridSpec: [top bar | main | spacer] x [left bar | main | text]
gs = GridSpec(3, 3, width_ratios=[0.22, 0.78, 0.02], height_ratios=[0.22, 0.78, 0.26], hspace=0.15, wspace=0.08)

ax_top = fig.add_subplot(gs[0, 1])
ax_left = fig.add_subplot(gs[1, 0])
ax_main = fig.add_subplot(gs[1, 1])
ax_text = fig.add_subplot(gs[2, 1])

# Probabilities
p_single = 1/6
probs_grid = np.full((6, 6), 1/36)

# Main 6x6 sample space grid (all equally likely)
im = ax_main.imshow(probs_grid, origin='lower', cmap='Blues', extent=[0.5, 6.5, 0.5, 6.5], vmin=0, vmax=1/36)
ax_main.set_xticks(np.arange(1, 7))
ax_main.set_yticks(np.arange(1, 7))
ax_main.set_xlabel('Second die outcome (j)')
ax_main.set_ylabel('First die outcome (i)')

# Draw cell borders for clarity
ax_main.set_xticks(np.arange(0.5, 7.5, 1), minor=True)
ax_main.set_yticks(np.arange(0.5, 7.5, 1), minor=True)
ax_main.grid(which='minor', color='white', linestyle='-', linewidth=2)
ax_main.tick_params(which='minor', bottom=False, left=False)

# Highlight specific ordered outcome (i=2 on first die, j=5 on second die)
i_target, j_target = 2, 5
highlight = patches.Rectangle((j_target - 0.5, i_target - 0.5), 1, 1, linewidth=3, edgecolor='#e31a1c', facecolor='none')
ax_main.add_patch(highlight)
ax_main.text(j_target, i_target, '1/36', ha='center', va='center', fontsize=12, color='#e31a1c', fontweight='bold')
ax_main.annotate('P(2 and 5) = 1/36', xy=(j_target, i_target), xytext=(4.5, 4.8),
                 arrowprops=dict(arrowstyle='->', color='#e31a1c', lw=2), color='#e31a1c', fontsize=12, ha='left')

# Top bar chart: probabilities for second die outcomes
x = np.arange(1, 7)
heights = np.full(6, p_single)
bars_top = ax_top.bar(x, heights, color='#c7e9b4', edgecolor='black')
# Highlight j=5 (index 4)
bars_top[4].set_color('#fdae6b')
bars_top[4].set_edgecolor('#e6550d')
ax_top.set_xlim(0.5, 6.5)
ax_top.set_ylim(0, 0.25)
ax_top.set_ylabel('P(outcome)')
ax_top.set_xticks(np.arange(1, 7))
ax_top.tick_params(axis='x', labelbottom=False)
ax_top.set_yticks([0, p_single])
ax_top.set_yticklabels(['0', '1/6'])
ax_top.set_title('Second die: P(j)', fontsize=11)

# Left bar chart: probabilities for first die outcomes (horizontal)
y = np.arange(1, 7)
bars_left = ax_left.barh(y, np.full(6, p_single), color='#c7e9b4', edgecolor='black')
# Highlight i=2 (index 1)
bars_left[1].set_color('#fdae6b')
bars_left[1].set_edgecolor('#e6550d')
ax_left.set_ylim(0.5, 6.5)
ax_left.set_xlim(0, 0.25)
ax_left.set_xlabel('P(outcome)')
ax_left.set_yticks(np.arange(1, 7))
ax_left.tick_params(axis='y', labelleft=False)
ax_left.set_xticks([0, p_single])
ax_left.set_xticklabels(['0', '1/6'])
ax_left.set_title('First die: P(i)', fontsize=11, loc='right')

# Explanatory text panel
ax_text.axis('off')
explain = (
    'Multiplication Rule for Independent Events:\n'
    'The two dice are independent, so probabilities multiply.\n'
    'P(2 on first die and 5 on second die) = P(2) × P(5) = (1/6) × (1/6) = 1/36.\n'
    'All 36 ordered pairs (i, j) in the 6×6 sample space are equally likely.'
)
ax_text.text(0.5, 0.6, explain, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

# Overall title
fig.suptitle('Multiplication Rule for Independent Events (Two Dice)', fontsize=16, y=0.98)

# Save figure
out_name = 'multiplication_rule_two_dice.png'
plt.savefig(out_name, dpi=200, bbox_inches='tight')
print(f'Saved figure to {out_name}')

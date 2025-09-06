import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch

# Define sample space for two fair dice
outcomes = [(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)]

# Events
A = [(d1, d2) for (d1, d2) in outcomes if d1 + d2 == 7]       # Sum is 7
B = [(d1, d2) for (d1, d2) in outcomes if d1 == d2]            # Doubles

A_set, B_set = set(A), set(B)
intersection = list(A_set & B_set)
union = list(A_set | B_set)

# Counts and probabilities
N = 36
count_A = len(A)
count_B = len(B)
count_I = len(intersection)
count_U = len(union)

pA = count_A / N
pB = count_B / N
pI = count_I / N
pU = pA + pB - pI

# Colors
color_A = '#4C78A8'   # blue
color_B = '#F58518'   # orange
color_I = '#54A24B'   # green (for intersection if any)
edge_color = '#4D4D4D'

fig, ax = plt.subplots(figsize=(9, 8))
fig.patch.set_facecolor('white')

# Draw grid and color-coded cells
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)
ax.set_aspect('equal')

# Background grid
for x in range(7):
    ax.plot([x, x], [0, 6], color='#DDDDDD', linewidth=1)
for y in range(7):
    ax.plot([0, 6], [y, y], color='#DDDDDD', linewidth=1)

# Draw cells
for d1 in range(1, 7):
    for d2 in range(1, 7):
        xy = (d2 - 1, d1 - 1)
        inA = (d1, d2) in A_set
        inB = (d1, d2) in B_set
        if inA and inB:
            face = color_I
            label = 'A∩B'
        elif inA:
            face = color_A
            label = ''
        elif inB:
            face = color_B
            label = ''
        else:
            face = 'white'
            label = ''
        rect = Rectangle(xy, 1, 1, facecolor=face, edgecolor=edge_color, linewidth=1)
        ax.add_patch(rect)
        # Annotate ordered pair
        ax.text(d2 - 0.5, d1 - 0.5, f"({d1},{d2})", ha='center', va='center', fontsize=9, color='#333333')

# Axis ticks and labels
ax.set_xticks(np.arange(0.5, 6.5, 1.0))
ax.set_yticks(np.arange(0.5, 6.5, 1.0))
ax.set_xticklabels([str(i) for i in range(1, 7)])
ax.set_yticklabels([str(i) for i in range(1, 7)])
ax.set_xlabel('Die 2 (columns)')
ax.set_ylabel('Die 1 (rows)')

# Legend
legend_patches = [
    Patch(facecolor=color_A, edgecolor=edge_color, label='A: sum is 7 (6 outcomes)'),
    Patch(facecolor=color_B, edgecolor=edge_color, label='B: doubles (6 outcomes)'),
]
if count_I > 0:
    legend_patches.append(Patch(facecolor=color_I, edgecolor=edge_color, label='A ∩ B (overlap)'))
leg = ax.legend(handles=legend_patches, loc='upper right', frameon=True, framealpha=0.95)

# Titles and explanatory text
fig.suptitle('Addition Rule for Probability with Two Dice', fontsize=16, y=0.98)

# Display the rule and plug-in values in a text box
rule_text = (
    'Let A: sum is 7,  B: doubles\n'
    'P(A or B) = P(A) + P(B) - P(A ∩ B)\n'
    f'           = {count_A}/{N} + {count_B}/{N} - {count_I}/{N}\n'
    f'           = {count_U}/{N} = {pU:.3f} = 1/3'
)
ax.text(0.02, 6.35, rule_text, fontsize=12, ha='left', va='top', family='monospace')

# Annotate that overlap is empty for this example
if count_I == 0:
    ax.text(5.98, -0.35, 'No overlap here: A ∩ B = ∅', fontsize=11, ha='right', va='top', color='#444444')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
output_path = 'addition_rule_two_dice.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f'Saved figure to {output_path}')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Define events
simple_event = (2, 3)  # a single specific outcome
compound_event = [(2, 6), (3, 5), (4, 4), (5, 3), (6, 2)]  # outcomes summing to 8

p_simple = 1 / 36
p_compound = len(compound_event) / 36

# Create figure and layout
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[1.45, 1], wspace=0.08)
ax = fig.add_subplot(gs[0, 0])
ax_text = fig.add_subplot(gs[0, 1])

# Draw 6x6 outcome grid (sample space)
for i in range(1, 7):
    for j in range(1, 7):
        ax.add_patch(
            patches.Rectangle(
                (i - 0.5, j - 0.5), 1, 1,
                facecolor="#f3f3f3", edgecolor="#cfcfcf", linewidth=1
            )
        )

# Highlight compound event cells (sum = 8)
for (i, j) in compound_event:
    ax.add_patch(
        patches.Rectangle(
            (i - 0.5, j - 0.5), 1, 1,
            facecolor="#6baed6", edgecolor="#2171b5", linewidth=2
        )
    )

# Highlight the simple event cell (2,3)
i, j = simple_event
ax.add_patch(
    patches.Rectangle(
        (i - 0.5, j - 0.5), 1, 1,
        facecolor="#ef3b2c", edgecolor="#a50f15", linewidth=2
    )
)

# Axes formatting
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_xticks(np.arange(1, 7))
ax.set_yticks(np.arange(1, 7))
ax.set_xlabel("Die 1 outcome", fontsize=11)
ax.set_ylabel("Die 2 outcome", fontsize=11)
ax.set_aspect('equal')
ax.set_title("Simple versus Compound Events with Two Dice", fontsize=14, pad=12)
ax.tick_params(labelsize=10)

# Helpful annotations
ax.annotate(
    "Simple event (2,3)",
    xy=(2, 3), xycoords='data',
    xytext=(1.1, 6.6), textcoords='data',
    arrowprops=dict(arrowstyle='->', color="#a50f15", lw=1.8),
    color="#a50f15", fontsize=10, ha='left', va='center'
)
ax.annotate(
    "Compound event: sum = 8",
    xy=(4, 4), xycoords='data',
    xytext=(4.8, 6.7), textcoords='data',
    arrowprops=dict(arrowstyle='->', color="#2171b5", lw=1.8),
    color="#084594", fontsize=10, ha='left', va='center'
)
ax.text(0.52, -0.24, "Total outcomes = 36 (6 × 6)", transform=ax.transAxes, fontsize=10, ha='left', color="#555555")

# Right panel: concise teaching notes and legend
ax_text.axis('off')

y0 = 0.95
ax_text.text(0.0, y0, "Simple vs Compound Events", fontsize=13, fontweight='bold', color="#222222")
ax_text.text(0.02, y0 - 0.10, "Simple event: exactly one outcome (e.g., (2,3)).", fontsize=11)
ax_text.text(0.02, y0 - 0.18, "Compound event: several outcomes (e.g., sum = 8).", fontsize=11)

ax_text.text(0.0, y0 - 0.30, "Counting method (same for both):", fontsize=11, fontweight='bold')
ax_text.text(0.02, y0 - 0.38, "Probability = number of favorable outcomes / 36", fontsize=11, color="#333333")

ax_text.text(0.0, y0 - 0.52, "Example compound event: sum = 8", fontsize=11, fontweight='bold')
ax_text.text(0.02, y0 - 0.60, "Outcomes: (2,6), (3,5), (4,4), (5,3), (6,2) → 5 outcomes", fontsize=11)

ax_text.text(0.0, y0 - 0.74, f"P(simple: (2,3)) = 1/36 ≈ {p_simple:.4f}", fontsize=11)
ax_text.text(0.0, y0 - 0.82, f"P(compound: sum = 8) = 5/36 ≈ {p_compound:.4f}", fontsize=11)

# Color legend markers
# Simple event color box
ax_text.add_patch(patches.Rectangle((0.00, y0 - 0.77), 0.035, 0.035, transform=ax_text.transAxes,
                                    facecolor="#ef3b2c", edgecolor="#a50f15", linewidth=1))
ax_text.text(0.045, y0 - 0.755, "Simple event outcome", transform=ax_text.transAxes, va='center', fontsize=10)
# Compound event color box
ax_text.add_patch(patches.Rectangle((0.00, y0 - 0.85), 0.035, 0.035, transform=ax_text.transAxes,
                                    facecolor="#6baed6", edgecolor="#2171b5", linewidth=1))
ax_text.text(0.045, y0 - 0.835, "Compound event outcomes", transform=ax_text.transAxes, va='center', fontsize=10)

# Save figure
outfile = "simple_vs_compound_events_dice.png"
plt.tight_layout()
plt.savefig(outfile, dpi=200)
print(f"Saved figure to {outfile}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# Data for dice outcomes
values = np.arange(1, 7)
outcomes = [(i, j) for i in values for j in values]

# Define the event: sum = 7
event_sum = 7
event_outcomes = [(i, event_sum - i) for i in values if 1 <= event_sum - i <= 6]
example_outcome = (2, 5)

# Compute probability details
total_outcomes = len(outcomes)
num_favorable = len(event_outcomes)
probability = num_favorable / total_outcomes

# Create figure with two panels: left (grid), right (explanation)
fig = plt.figure(figsize=(11, 6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1], wspace=0.25)
ax = fig.add_subplot(gs[0, 0])
ax_text = fig.add_subplot(gs[0, 1])

# Left panel: grid of outcomes
ax.set_title("Pair of Dice: Outcomes vs Event (sum = 7)", fontsize=14, pad=10)
ax.set_xlim(0.5, 6.5)
ax.set_ylim(0.5, 6.5)
ax.set_xticks(range(1, 7))
ax.set_yticks(range(1, 7))
ax.set_xlabel("Die 1")
ax.set_ylabel("Die 2")
ax.set_aspect("equal")
ax.grid(which="major", linestyle="--", alpha=0.25)

# Lightly shade event cells (sum = 7)
event_color = "#4C78A8"
for (i, j) in event_outcomes:
    rect = Rectangle((i - 0.5, j - 0.5), 1, 1,
                     facecolor=event_color, edgecolor=event_color,
                     alpha=0.22, linewidth=1.2, zorder=0)
    ax.add_patch(rect)

# Plot all outcomes as faint points
xs = [i for (i, _) in outcomes]
ys = [j for (_, j) in outcomes]
ax.scatter(xs, ys, s=26, c="#B8B8B8", edgecolors="#B8B8B8", linewidths=0.5, zorder=1, label=None)

# Overlay event outcomes with stronger markers
ex = [i for (i, _) in event_outcomes]
ey = [j for (_, j) in event_outcomes]
ax.scatter(ex, ey, s=60, c=event_color, edgecolors=event_color, linewidths=0.8, zorder=2)

# Highlight a single example outcome (2,5)
ax.scatter([example_outcome[0]], [example_outcome[1]], s=180,
           facecolor="#FFD56B", edgecolor="#2E2E2E", linewidths=1.5, zorder=3)
ax.annotate("Example outcome (2,5)", xy=example_outcome, xycoords="data",
            xytext=(3.7, 5.8), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="#2E2E2E", lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CCCCCC", alpha=0.95),
            fontsize=11)

# Legend
legend_handles = [
    Patch(facecolor=event_color, edgecolor=event_color, alpha=0.22, label="Event: sum = 7"),
    Line2D([0], [0], marker='o', color='none', markerfacecolor="#B8B8B8", markeredgecolor="#B8B8B8",
           markersize=7, label="Other outcomes"),
    Line2D([0], [0], marker='o', color='none', markerfacecolor="#FFD56B", markeredgecolor="#2E2E2E",
           markersize=10, label="Example outcome (2,5)")
]
ax.legend(handles=legend_handles, loc="upper right", frameon=True, framealpha=0.95)

# Right panel: explanatory text
ax_text.axis('off')
ax_text.set_title("Event, Outcome, and Probability", fontsize=14, pad=10)

# Build strings for explanation
event_list_str = ", ".join([f"({i},{j})" for (i, j) in event_outcomes])
calc_lines = [
    "Definitions:",
    "• Outcome: one specific pair (i, j) from rolling two dice (e.g., (2,5)).",
    f"• Event: set of outcomes with sum = 7: {{{event_list_str}}} (6 outcomes).",
    "",
    "Sample space:",
    "• 6 × 6 = 36 equally likely outcomes.",
    "• Each single outcome has probability 1/36 ≈ 0.0278.",
    "",
    "Probability of the event 'sum = 7':",
    f"• Favorable outcomes = {num_favorable}",
    f"• Total outcomes = {total_outcomes}",
    f"• P(sum = 7) = {num_favorable}/{total_outcomes} = 1/6 ≈ {probability:.3f}"
]

# Place text neatly with spacing
y = 0.95
for line in calc_lines:
    ax_text.text(0.02, y, line, fontsize=11, va='top', ha='left')
    y -= 0.09 if line == "" else 0.08

# Save figure
outfile = "dice_event_outcome_probability.png"
plt.savefig(outfile, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure to {outfile}")

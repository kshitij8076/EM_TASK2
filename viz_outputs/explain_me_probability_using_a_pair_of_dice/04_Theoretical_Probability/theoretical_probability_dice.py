import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch


def make_figure(save_path="theoretical_probability_two_dice.png"):
    # Colors
    base_cell_color = "#e8eef7"   # light blue for equally likely outcomes
    highlight_cell_color = "#f9d7c5"  # light orange for the event sum=7
    bar_color = "#7fa2d6"
    bar_highlight_color = "#e67e22"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    ax1, ax2 = axes

    # --------------------- Left panel: 6x6 sample space ---------------------
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 6)
    ax1.set_aspect('equal')

    # Draw cells and annotate
    for i in range(1, 7):       # Die 1
        for j in range(1, 7):   # Die 2
            s = i + j
            is_sum7 = (s == 7)
            rect = Rectangle((i - 1, j - 1), 1, 1,
                              facecolor=highlight_cell_color if is_sum7 else base_cell_color,
                              edgecolor="#888888", linewidth=0.8)
            ax1.add_patch(rect)
            # Annotate ordered pair at the center of the cell
            ax1.text(i - 0.5, j - 0.5, f"({i},{j})",
                     ha='center', va='center', fontsize=8, color="#222222")

    # Ticks and labels centered on cells
    ax1.set_xticks(np.arange(0.5, 6.5, 1.0))
    ax1.set_yticks(np.arange(0.5, 6.5, 1.0))
    ax1.set_xticklabels(range(1, 7))
    ax1.set_yticklabels(range(1, 7))
    ax1.set_xlabel("Die 1 result")
    ax1.set_ylabel("Die 2 result")
    ax1.set_title("Sample space (36 equally likely outcomes)")

    # Legend explaining theoretical probability counting
    legend_handles = [
        Patch(facecolor=base_cell_color, edgecolor="#888888", label="Any single outcome: 1/36"),
        Patch(facecolor=highlight_cell_color, edgecolor="#888888", label="Event: sum = 7 (6 outcomes)")
    ]
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=9, frameon=True, framealpha=0.95)

    # Text showing probability calculation
    ax1.text(0.05, 0.15, "P(sum = 7) = favorable outcomes / 36 = 6/36 = 1/6",
             transform=ax1.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, ec='#cccccc'))

    # Hide default grid; cell borders already show the grid
    for spine in ax1.spines.values():
        spine.set_edgecolor('#444444')

    # --------------------- Right panel: distribution of sums ---------------------
    sums = np.arange(2, 13)
    # Count outcomes for each sum using theoretical counting
    counts = np.array([sum(1 for i in range(1, 7) for j in range(1, 7) if i + j == s) for s in sums])
    probs = counts / 36.0

    colors = [bar_highlight_color if s == 7 else bar_color for s in sums]
    bars = ax2.bar(sums, probs, color=colors, edgecolor="#444444")

    ax2.set_xlabel("Sum of two dice")
    ax2.set_ylabel("Probability")
    ax2.set_title("Theoretical distribution of sums")
    ax2.set_xticks(sums)
    ax2.set_ylim(0, max(probs) * 1.25)

    # Annotate each bar with count/36 (and 1/6 for 7)
    for rect, s, c in zip(bars, sums, counts):
        height = rect.get_height()
        label = f"{c}/36"
        if s == 7:
            label += " = 1/6"
        ax2.text(rect.get_x() + rect.get_width() / 2, height + 0.005, label,
                 ha='center', va='bottom', fontsize=9)

    # Subtle reference line at 1/6 to connect both panels
    ax2.axhline(1/6, color='#999999', linestyle='--', linewidth=1)
    ax2.text(12.6, 1/6, " 1/6", va='center', ha='left', color='#666666', fontsize=9)

    # Suptitle
    fig.suptitle("Theoretical Probability with Two Fair Dice: Count outcomes in the sample space to compute P(events)", fontsize=13)

    # Save figure
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
    print("Saved figure to 'theoretical_probability_two_dice.png'")

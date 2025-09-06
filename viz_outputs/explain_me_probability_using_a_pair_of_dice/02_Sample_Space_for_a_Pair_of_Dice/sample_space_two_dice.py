import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle, Patch


def make_figure(savepath="sample_space_two_dice.png"):
    # Create a 6x6 grid; highlight an example event (sum = 7)
    arr = np.zeros((6, 6), dtype=int)
    for i in range(1, 7):
        for j in range(1, 7):
            if i + j == 7:
                arr[j - 1, i - 1] = 1  # rows: second die (y), cols: first die (x)

    cmap = ListedColormap(["#ffffff", "#fde4c8"])  # white for all, light orange for sum=7

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=False)
    ax.set_facecolor("white")

    # Display background coloring
    ax.imshow(
        arr,
        cmap=cmap,
        origin="lower",
        extent=[0.5, 6.5, 0.5, 6.5],
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )

    # Draw grid lines
    for x in np.arange(0.5, 6.6, 1.0):
        ax.axvline(x, color="0.85", lw=1)
        ax.axhline(x, color="0.85", lw=1)

    # Draw border around the 6x6 grid
    ax.add_patch(Rectangle((0.5, 0.5), 6.0, 6.0, fill=False, edgecolor="black", linewidth=1.6))

    # Label each cell with the ordered pair (first die, second die)
    for i in range(1, 7):
        for j in range(1, 7):
            ax.text(i, j, f"({i},{j})", ha="center", va="center", fontsize=9, color="#1f1f1f")

    # Axis setup
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    ax.set_xlabel("First die outcome")
    ax.set_ylabel("Second die outcome")
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 6.5)
    ax.set_aspect("equal")

    # Hide default spines (we added a custom border)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title with key idea
    ax.set_title(
        "Sample Space for a Pair of Dice (Ordered Pairs)\n6 × 6 = 36 equally likely outcomes",
        fontsize=14,
        pad=12,
    )

    # Legend for the highlighted event (sum = 7)
    event_patch = Patch(facecolor="#fde4c8", edgecolor="#cfcfcf", label="Example event: sum = 7")
    ax.legend(handles=[event_patch], loc="upper right", frameon=True)

    # Save figure
    plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_figure()

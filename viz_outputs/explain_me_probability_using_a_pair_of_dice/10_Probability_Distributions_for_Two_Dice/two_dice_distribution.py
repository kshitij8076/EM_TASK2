import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter


def main():
    # Compute sums for two fair six-sided dice
    sides = np.arange(1, 7)
    sum_matrix = np.add.outer(sides, sides)  # 6x6 matrix of sums (2..12)
    sums = np.arange(2, 13)
    counts = np.array([(sum_matrix == s).sum() for s in sums])
    probs = counts / 36.0

    # Set up consistent colormap across both plots
    cmap = plt.cm.viridis
    norm = Normalize(vmin=2, vmax=12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # --- Left: Bar chart of probability distribution ---
    ax = axes[0]
    colors = cmap(norm(sums))
    bars = ax.bar(sums, probs, color=colors, edgecolor='black', linewidth=0.8)

    ax.set_title('Sum distribution for two fair six-sided dice')
    ax.set_xlabel('Sum of two dice')
    ax.set_ylabel('Probability')
    ax.set_xticks(sums)
    ax.set_ylim(0, 0.19)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

    # Annotate number of ways above each bar
    for s, p, n, bar in zip(sums, probs, counts, bars):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            p + 0.005,
            f"{n} {'way' if n == 1 else 'ways'}",
            ha='center', va='bottom', fontsize=9
        )

    # Highlight the most likely sum (7)
    idx7 = np.where(sums == 7)[0][0]
    ax.axvline(7, color='red', linestyle='--', linewidth=1)
    ax.annotate(
        'Most likely sum = 7\n(6 ways \u2248 16.7%)',
        xy=(7, probs[idx7]),
        xytext=(8.5, 0.16),
        arrowprops=dict(arrowstyle='->', color='red'),
        ha='left', va='center', fontsize=9, color='red'
    )

    # Show that probabilities sum to 1
    ax.text(
        0.02, 0.95,
        f'Total probability = {probs.sum():.1f}',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='gray')
    )

    # --- Right: 6x6 outcome grid (heatmap of sums) ---
    ax2 = axes[1]
    im = ax2.imshow(sum_matrix, cmap=cmap, norm=norm, origin='lower')
    ax2.set_title('36 equally likely outcomes (Die A vs Die B)')
    ax2.set_xlabel('Die B face')
    ax2.set_ylabel('Die A face')
    ax2.set_xticks(np.arange(6))
    ax2.set_yticks(np.arange(6))
    ax2.set_xticklabels(sides)
    ax2.set_yticklabels(sides)

    # Grid lines between cells
    ax2.set_xticks(np.arange(-0.5, 6, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax2.grid(which='minor', color='white', linestyle='-', linewidth=1)
    ax2.tick_params(which='minor', bottom=False, left=False)

    # Annotate each cell with the sum value, color-adjusted for readability
    for i in range(6):
        for j in range(6):
            s = int(sum_matrix[i, j])
            r, g, b, _ = cmap(norm(s))
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = 'black' if luminance > 0.6 else 'white'
            ax2.text(j, i, str(s), ha='center', va='center', fontsize=9, color=text_color)

    # Shared colorbar to link colors with sums across both plots
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(sum_matrix)
    cbar = fig.colorbar(sm, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label('Sum of two dice')

    # Save figure
    outname = 'two_dice_probability_distribution.png'
    plt.savefig(outname, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    main()

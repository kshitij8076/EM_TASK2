import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

def draw_sample_space(ax):
    base = '#f2f2f2'
    highlight = '#2ca02c'
    # Draw 6x6 outcome grid
    for i in range(1, 7):  # Die 1
        for j in range(1, 7):  # Die 2
            s = i + j
            fc = highlight if s == 7 else base
            rect = Rectangle((i - 0.5, j - 0.5), 1, 1, facecolor=fc, edgecolor='0.55', linewidth=1.0)
            ax.add_patch(rect)
            text_color = 'white' if s == 7 else '#555555'
            fontweight = 'bold' if s == 7 else 'normal'
            ax.text(i, j, f"{s}", ha='center', va='center', fontsize=9 if s != 7 else 11,
                    color=text_color, fontweight=fontweight)
    ax.set_xlim(0.5, 6.5)
    ax.set_ylim(0.5, 6.5)
    ax.set_xticks(range(1, 7))
    ax.set_yticks(range(1, 7))
    ax.set_xlabel('Die 1')
    ax.set_ylabel('Die 2')
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('Sample space: 6×6 = 36 equally likely outcomes', fontsize=12, pad=8)

    legend_handles = [
        Patch(facecolor=highlight, edgecolor='0.55', label='sum = 7'),
        Patch(facecolor=base, edgecolor='0.55', label='other sums')
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=9, frameon=True)

    ax.text(0.0, -0.12, 'Each cell = one ordered outcome (1/36). Diagonal cells share the same sum.',
            transform=ax.transAxes, fontsize=9)


def draw_sum_distribution(ax):
    sums = np.arange(2, 13)
    counts = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    probs = counts / 36.0
    base_bar = '#9ecae1'
    highlight = '#2ca02c'

    colors = [highlight if s == 7 else base_bar for s in sums]
    ax.bar(sums, probs, color=colors, edgecolor='0.3', linewidth=1)

    ax.set_xticks(sums)
    ax.set_xlim(1.5, 12.5)
    ax.set_ylim(0, probs.max() + 0.035)
    ax.set_xlabel('Sum of two dice')
    ax.set_ylabel('Probability')
    ax.set_title('Theoretical probability of each sum', fontsize=12, pad=8)

    # Annotate each bar with count/36
    for s, c, p in zip(sums, counts, probs):
        ax.text(s, p + 0.006, f'{c}/36', ha='center', va='bottom', fontsize=9)

    # Gridlines for readability
    ax.yaxis.grid(True, linestyle=':', color='0.85')
    ax.set_axisbelow(True)

    # Highlight and annotate the most likely sum (7)
    y7 = probs[sums.tolist().index(7)]
    ax.annotate('Most likely sum\n6 outcomes = 1/6',
                xy=(7, y7), xytext=(9.6, y7 + 0.03),
                arrowprops=dict(arrowstyle='->', color=highlight, lw=1.5),
                fontsize=10, color=highlight, ha='left')


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    draw_sample_space(axes[0])
    draw_sum_distribution(axes[1])

    fig.suptitle('Probability with a Pair of Dice: equally likely outcomes and sum probabilities',
                 fontsize=14, y=0.98)
    fig.text(0.5, 0.01, 'Randomness means any one roll is unpredictable, but probabilities let us predict patterns over many rolls.',
             ha='center', fontsize=10)

    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    out = 'dice_probability_explained.png'
    fig.savefig(out, dpi=200)
    print(f'Saved figure to {out}')


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def length_contraction(beta):
    """Return L/L0 for a given beta = v/c."""
    beta = np.asarray(beta)
    if np.any(beta < 0) or np.any(beta >= 1):
        raise ValueError("beta must satisfy 0 <= beta < 1")
    return np.sqrt(1 - beta**2)


def main():
    # Data for the curve
    beta = np.linspace(0.0, 0.999, 600)
    L_over_L0 = length_contraction(beta)

    # Selected illustrative speeds
    selected = [0.5, 0.8, 0.95]
    selected_vals = [length_contraction(b) for b in selected]

    # Figure and style
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 11
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax = axes[0]

    # Left panel: contraction curve
    ax.plot(beta, L_over_L0, color="#1f77b4", linewidth=2)
    ax.axhline(1.0, color="0.7", linestyle="--", linewidth=1, label="No contraction")

    # Mark selected betas
    colors = ["#2ca02c", "#d62728", "#9467bd"]
    for b, y, c in zip(selected, selected_vals, colors):
        ax.axvline(b, color=c, linestyle=":", linewidth=1)
        ax.plot([b], [y], marker="o", color=c)
        ax.annotate(
            f"v = {b:.2f}c\nL/L₀ = {y:.2f}",
            xy=(b, y), xycoords='data',
            xytext=(15, 15), textcoords='offset points',
            arrowprops=dict(arrowstyle='->', color=c, lw=1),
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=c, lw=1),
            color='black'
        )

    ax.set_xlabel("Speed fraction β = v/c")
    ax.set_ylabel("Contracted length (L/L₀)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_title("Length contraction factor vs speed")

    # Show the formula on the plot
    ax.text(0.02, 0.15, r"$L = L_0 \, \sqrt{1 - \beta^2}$", transform=ax.transAxes,
            fontsize=12, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.8'))

    # Right panel: visual rod comparison at v = 0.8c
    ax2 = axes[1]
    ax2.set_title("Proper length vs contracted length (v = 0.80c)")
    ax2.axis('off')

    # Define lengths in arbitrary units
    L0 = 1.0
    beta_demo = 0.8
    L_demo = length_contraction(beta_demo) * L0

    # Drawing parameters
    y0 = 0.65  # y center for proper length bar
    y1 = 0.35  # y center for contracted bar
    h = 0.12   # bar height

    # Set plotting limits to include some margin
    ax2.set_xlim(-0.05, 1.15 * L0)
    ax2.set_ylim(0, 1)

    # Bars
    proper_rect = Rectangle((0, y0 - h/2), L0, h, facecolor="#4daf4a", edgecolor="black")
    contracted_rect = Rectangle((0, y1 - h/2), L_demo, h, facecolor="#e41a1c", edgecolor="black")
    ax2.add_patch(proper_rect)
    ax2.add_patch(contracted_rect)

    # Double-headed measurement arrows
    # Proper length arrow
    ax2.annotate("", xy=(0, y0 + 0.18), xytext=(L0, y0 + 0.18),
                 arrowprops=dict(arrowstyle="<->", lw=1.4, color="black"))
    ax2.text(L0/2, y0 + 0.21, "L₀ (proper length)", ha="center", va="bottom")

    # Contracted length arrow
    ax2.annotate("", xy=(0, y1 - 0.18), xytext=(L_demo, y1 - 0.18),
                 arrowprops=dict(arrowstyle="<->", lw=1.4, color="black"))
    ax2.text(L_demo/2, y1 - 0.22, f"L = {L_demo:.2f} L₀", ha="center", va="top")

    # Labels next to bars
    ax2.text(L0 + 0.02, y0, "At rest (proper frame)", va="center")
    ax2.text(L_demo + 0.02, y1, f"Moving at {beta_demo:.2f}c", va="center")

    # Context note
    ax2.text(0, 0.03, "Contraction occurs only along the direction of motion.", fontsize=10)

    # Overall title
    fig.suptitle("Length Contraction in Special Relativity", fontsize=14)

    # Save figure
    out_name = "length_contraction.png"
    plt.savefig(out_name, bbox_inches='tight')
    print(f"Saved figure to {out_name}")


if __name__ == "__main__":
    main()

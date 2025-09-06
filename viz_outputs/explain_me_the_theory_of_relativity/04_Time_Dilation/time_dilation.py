import numpy as np
import matplotlib.pyplot as plt


def gamma(beta):
    beta = np.asarray(beta)
    return 1.0 / np.sqrt(1.0 - beta**2)


def main():
    # Data for left plot (continuous curve)
    beta_curve = np.linspace(0.0, 0.99, 600)
    gamma_curve = gamma(beta_curve)

    # Selected illustrative speeds (v/c) and their dilation factors
    betas = np.array([0.0, 0.5, 0.8, 0.95, 0.99])
    gammas = gamma(betas)

    # Figure layout
    plt.figure(figsize=(11, 5.5))
    gs = plt.GridSpec(1, 2, width_ratios=[1.2, 1.0], wspace=0.25)

    # Left subplot: dilation factor vs speed
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(beta_curve, gamma_curve, color="#1f77b4", lw=2)
    ax1.set_xlabel("speed as a fraction of light speed, v/c")
    ax1.set_ylabel("time dilation factor, t'/t = b3")
    ax1.set_title("Dilation grows nonlinearly as speed approaches c")
    ax1.grid(True, alpha=0.25)

    # Mark and annotate selected points
    sample_betas = betas[1:]  # exclude 0 for vertical lines clarity
    sample_gammas = gammas[1:]
    colors = ["#2ca02c", "#ff7f0e", "#d62728", "#9467bd"]
    for b, g, col in zip(sample_betas, sample_gammas, colors):
        ax1.axvline(b, color=col, ls="--", lw=1.2, alpha=0.7)
        ax1.plot([b], [g], marker="o", color=col)
        ax1.text(b + 0.01, g + 0.15, f"v/c={b:.2f}\nb3={g:.2f}", color=col, fontsize=9,
                 va="bottom", ha="left")

    # Right subplot: visualizing one "tick" duration stretching with speed
    ax2 = plt.subplot(gs[0, 1])

    # Build horizontal bars: one tick at rest (1 s) and dilated ticks at speeds
    labels = ["rest (v=0)", "v=0.5c", "v=0.8c", "v=0.95c", "v=0.99c"]
    durations = gammas  # since t' = b3 * t with t=1 s at rest

    y_positions = np.arange(len(labels))[::-1]  # top to bottom
    bar_colors = ["#1f77b4"] + colors  # rest + speeds

    # Set x-scale to include the largest dilation comfortably
    xmax = float(np.ceil(durations.max() * 1.1))

    for y, d, lab, col in zip(y_positions, durations, labels, bar_colors):
        ax2.hlines(y=y, xmin=0, xmax=d, color=col, lw=10, alpha=0.9)
        ax2.plot(d, y, marker="|", color="black")
        ax2.text(d + 0.08, y, f"{d:.2f} s", va="center", ha="left", fontsize=10)

    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(labels)
    ax2.set_xlim(0, xmax)
    ax2.set_xlabel("duration of one tick as observed in the lab (seconds)")
    ax2.set_title("Moving clocks tick slower: each tick takes longer by b3")
    ax2.grid(True, axis="x", alpha=0.25)

    # Styling
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Overall title with the time dilation equation
    plt.suptitle("Time Dilation\n$ t' = \\dfrac{t}{\\sqrt{1 - v^2/c^2}} $  (with t = 1 s at rest, so t' = b3 t)", y=1.02, fontsize=15)

    # Explanatory footnote
    plt.figtext(0.02, 0.01,
                "As relative speed increases, the dilation factor b3 grows, so intervals between ticks grow.\n"
                "This has been confirmed with fast-moving particles (e.g., muons) and in particle accelerators.",
                fontsize=9)

    outname = "time_dilation_figure.png"
    plt.tight_layout()
    plt.savefig(outname, dpi=200)
    print(f"Saved figure to {outname}")


if __name__ == "__main__":
    main()

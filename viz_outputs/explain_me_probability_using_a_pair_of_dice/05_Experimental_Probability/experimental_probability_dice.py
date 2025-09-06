import numpy as np
import matplotlib.pyplot as plt

# Generate a figure illustrating experimental probability with two dice.
# Left: Experimental vs theoretical probabilities for sums (N=100 trials).
# Right: Convergence of experimental probability of sum=7 to theoretical 1/6 with increasing trials.

def simulate_sums(n, rng):
    rolls = rng.integers(1, 7, size=(n, 2))  # two fair dice
    return rolls.sum(axis=1)

def main():
    rng = np.random.default_rng(42)  # reproducible

    # Theoretical distribution for sums 2..12 from two fair dice
    sums = np.arange(2, 13)
    theo_counts = np.array([1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1], dtype=float)
    p_theory = theo_counts / 36.0

    # Part 1: A small experiment (N=100) to show experimental vs theoretical
    N_bar = 100
    s_bar = simulate_sums(N_bar, rng)
    counts_exp = np.bincount(s_bar, minlength=13)[2:13]
    p_exp = counts_exp / N_bar

    # Part 2: Convergence of experimental probability for sum=7
    N_run = 2000
    s_run = simulate_sums(N_run, rng)
    is_seven = (s_run == 7).astype(float)
    running_p = np.cumsum(is_seven) / np.arange(1, N_run + 1)

    # Approximate 95% confidence band around theoretical p using normal approx
    p = 1.0 / 6.0
    n_vals = np.arange(1, N_run + 1)
    se = np.sqrt(p * (1 - p) / n_vals)
    z = 1.96
    upper = np.clip(p + z * se, 0, 1)
    lower = np.clip(p - z * se, 0, 1)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, constrained_layout=True)
    ax0, ax1 = axes

    # Left subplot: bar chart vs theoretical
    ax0.bar(sums, p_exp, width=0.8, color="#4C78A8", edgecolor="black", label=f"Experimental (N={N_bar})")
    ax0.plot(sums, p_theory, color="black", marker="o", linestyle="--", label="Theoretical")

    # Annotate bar heights
    for x, y in zip(sums, p_exp):
        ax0.text(x, y + 0.006, f"{y:.2f}", ha="center", va="bottom", fontsize=8)

    # Highlight experimental P(sum=7)
    idx7 = 7 - 2
    count7 = int(counts_exp[idx7])
    p7 = p_exp[idx7]
    ax0.text(0.5, 0.98, f"For N={N_bar}: P(7) = {count7}/{N_bar} = {p7:.2f}",
             transform=ax0.transAxes, ha="center", va="top", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="#aaaaaa"))

    ax0.set_xticks(sums)
    ax0.set_xlabel("Sum of two dice")
    ax0.set_ylabel("Probability")
    ax0.set_title("Experimental vs Theoretical Distribution")
    ax0.set_ylim(0, max(0.2, p_exp.max() + 0.06))
    ax0.grid(axis='y', alpha=0.3)
    ax0.legend(frameon=False)

    # Right subplot: running experimental probability for sum=7
    ax1.plot(n_vals, running_p, color="#E45756", linewidth=1.6, label="Experimental P(sum=7)")
    ax1.axhline(p, color="black", linestyle="--", linewidth=1.0, label="Theoretical 1/6 ≈ 0.167")
    ax1.fill_between(n_vals, lower, upper, color="#72B7B2", alpha=0.3, label="Approx 95% band")

    ax1.set_xlabel("Number of trials")
    ax1.set_ylabel("Probability")
    ax1.set_title("Convergence of Experimental Probability to Theoretical")
    ax1.set_xlim(1, N_run)
    ax1.set_ylim(0, 0.4)
    ax1.grid(alpha=0.3)
    ax1.legend(frameon=False)

    fig.suptitle("Experimental Probability with a Pair of Dice", fontsize=14, y=1.03)

    out_path = "experimental_probability_dice.png"
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()

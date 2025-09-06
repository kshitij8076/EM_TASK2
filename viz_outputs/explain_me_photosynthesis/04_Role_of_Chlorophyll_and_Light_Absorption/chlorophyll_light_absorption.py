import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Circle, FancyArrowPatch


def gauss(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def main():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.frameon": False
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1]})
    fig.set_facecolor("white")

    # --- Left panel: Absorption vs wavelength ---
    w = np.linspace(380, 750, 1000)
    absorption_raw = 0.95 * gauss(w, 430, 22) + 0.85 * gauss(w, 662, 28)
    absorption = absorption_raw / absorption_raw.max()

    # Reflectance peak in green region (plants look green)
    reflectance = 0.15 + 0.8 * np.exp(-0.5 * ((w - 550) / 45) ** 2)
    reflectance = np.clip(reflectance, 0, 1)

    # Shaded wavelength bands
    ax1.axvspan(400, 500, color="royalblue", alpha=0.06)
    ax1.axvspan(500, 570, color="limegreen", alpha=0.06)
    ax1.axvspan(620, 700, color="crimson", alpha=0.06)

    # Curves
    ax1.plot(w, absorption, color="forestgreen", lw=3, label="Absorption (chlorophyll a)")
    ax1.plot(w, reflectance, color="dimgray", lw=2, ls="--", label="Relative reflectance")

    # Annotations for peaks and reflection
    x_blue = 430
    y_blue = np.interp(x_blue, w, absorption)
    ax1.annotate("Blue light\nstrongly absorbed",
                 xy=(x_blue, y_blue), xytext=(395, 0.82),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    x_red = 662
    y_red = np.interp(x_red, w, absorption)
    ax1.annotate("Red light\nstrongly absorbed",
                 xy=(x_red, y_red), xytext=(690, 0.85),
                 ha="left",
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    x_green = 550
    y_green = np.interp(x_green, w, reflectance)
    ax1.annotate("Green mostly\nreflected",
                 xy=(x_green, y_green), xytext=(520, 0.35),
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.2))

    # Formatting
    ax1.set_title("Light absorption by chlorophyll")
    ax1.set_xlabel("Wavelength (nm)")
    ax1.set_ylabel("Relative intensity")
    ax1.set_xlim(380, 750)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, ls=":", alpha=0.4)
    ax1.legend(loc="upper right")

    # --- Right panel: Schematic of chloroplast thylakoids and light reactions ---
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Chlorophyll initiates light reactions in thylakoids")

    # Chloroplast outer body
    chloroplast = Ellipse((5, 5), width=8.8, height=5.2,
                          facecolor="#cfeecd", edgecolor="forestgreen", lw=2)
    ax2.add_patch(chloroplast)

    # Granum: stack of thylakoid discs
    y0 = 5
    for i in range(-3, 3):
        y = y0 + i * 0.38
        disc = FancyBboxPatch((3.6, y), 2.8, 0.34,
                              boxstyle="round,pad=0.02,rounding_size=0.12",
                              facecolor="#1b5e20", edgecolor="#0f3a12")
        ax2.add_patch(disc)

    # Label for chlorophyll location
    ax2.annotate("Chlorophyll in\nthylakoid membranes",
                 xy=(5.0, 5.0), xytext=(5.3, 7.4), ha="left",
                 arrowprops=dict(arrowstyle="->", lw=1.4, color="black"))

    # Incoming photons (yellow arrows)
    photon_arrows = [((1.2, 8.8), (4.1, 6.2)),
                     ((1.0, 7.3), (4.0, 5.5)),
                     ((1.6, 6.4), (4.0, 4.8))]
    for (x0, y0), (x1, y1) in photon_arrows:
        ax2.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='->',
                                      mutation_scale=14, lw=2.2, color="#f7d21f"))
    ax2.text(0.7, 9.2, "photons", color="#9a8700", fontsize=11)
    ax2.text(0.6, 9.8, "\u2600", fontsize=16, color="#f7d21f")  # sun symbol

    # Stars where photons are captured
    ax2.scatter([4.05, 4.05, 4.05], [6.2, 5.5, 4.8], marker='*', s=180,
                color="#ffd000", edgecolors="#e6a800", zorder=5)

    # Water molecule (substrate of splitting)
    water = Circle((3.2, 4.4), radius=0.26, facecolor="#4aa3df", edgecolor="#1c6aa7", lw=1.2)
    ax2.add_patch(water)
    ax2.text(3.2, 4.4, "H2O", ha='center', va='center', fontsize=10, color='white')

    # Arrow showing water splitting -> O2 release
    ax2.add_patch(FancyArrowPatch((3.45, 4.6), (2.2, 5.0), arrowstyle='->',
                                  mutation_scale=12, lw=1.4, color="#1c6aa7"))
    ax2.text(1.9, 5.15, "O2", color="#1c6aa7", fontsize=11)

    # Energy carriers produced
    ax2.add_patch(FancyArrowPatch((6.7, 5.0), (9.0, 5.8), arrowstyle='->',
                                  mutation_scale=12, lw=1.6, color="#b30000"))
    ax2.text(9.05, 5.8, "ATP, NADPH", color="#b30000", fontsize=11, va='center')

    # Caption-like explanatory note
    ax2.text(5.0, 1.0,
             "Absorbed light energy excites electrons in chlorophyll,\n"
             "driving water splitting (\u2192 O2) and forming ATP and NADPH for the Calvin cycle.",
             ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    out_name = "chlorophyll_light_absorption.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_name}")


if __name__ == "__main__":
    main()

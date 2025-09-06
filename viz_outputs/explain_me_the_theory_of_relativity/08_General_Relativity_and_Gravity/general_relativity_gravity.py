import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


def warp_points(X, Y, cx=0.0, cy=0.0, A=1.0, sigma=1.5):
    """
    Radial inward warp around (cx, cy) to visually mimic spacetime curvature.
    X, Y: arrays of coordinates
    A: warp amplitude
    sigma: scale of curvature region
    Returns warped coordinates (Xw, Yw)
    """
    dx = X - cx
    dy = Y - cy
    r = np.hypot(dx, dy) + 1e-9
    dr = A * np.exp(-(r**2) / (2.0 * sigma**2))
    new_r = np.maximum(r - dr, 0.02)
    scale = new_r / r
    Xw = cx + dx * scale
    Yw = cy + dy * scale
    return Xw, Yw


def plot_curved_spacetime(ax):
    # Parameters for the mass (lens)
    cx, cy = 0.0, 0.0
    A, sigma = 1.0, 1.6

    # Draw warped grid lines
    x_min, x_max = -5.0, 5.0
    y_min, y_max = -3.8, 3.8
    nx, ny = 11, 9

    grid_color = (0.75, 0.75, 0.75)
    for x0 in np.linspace(x_min, x_max, nx):
        Y = np.linspace(y_min, y_max, 600)
        X = np.full_like(Y, x0)
        Xw, Yw = warp_points(X, Y, cx, cy, A=A, sigma=sigma)
        ax.plot(Xw, Yw, color=grid_color, lw=0.8, zorder=1)

    for y0 in np.linspace(y_min, y_max, ny):
        X = np.linspace(x_min, x_max, 800)
        Y = np.full_like(X, y0)
        Xw, Yw = warp_points(X, Y, cx, cy, A=A, sigma=sigma)
        ax.plot(Xw, Yw, color=grid_color, lw=0.8, zorder=1)

    # Massive body at the center
    mass = Circle((cx, cy), 0.55, facecolor="#4d4d4d", edgecolor="black", lw=1.0, zorder=5)
    ax.add_patch(mass)
    ax.text(cx, cy - 0.95, "Massive body\n(curves spacetime)", ha="center", va="top", fontsize=9)

    # Light ray bending (geodesic) skirting the mass
    x = np.linspace(x_min, x_max, 1000)
    b = 2.2      # impact parameter
    A_defl = 1.15
    s = 1.25
    y = b - A_defl / (1.0 + (x / s) ** 2)
    ax.plot(x, y, color="#e67e22", lw=2.4, zorder=6, label="Light path (geodesic)")

    # Arrow indicating direction of light
    x_arrow = -3.8
    y_arrow = b - A_defl / (1.0 + (x_arrow / s) ** 2)
    arrow = FancyArrowPatch((x_arrow - 0.9, y_arrow + 0.02), (x_arrow, y_arrow),
                            arrowstyle='-|>', mutation_scale=12, color="#e67e22", lw=2.0, zorder=7)
    ax.add_patch(arrow)
    ax.text(-4.7, b + 0.35, "Incoming light", color="#e67e22", fontsize=9)
    ax.text(1.7, 1.1, "Deflection\n(gravitational lensing)", color="#e67e22", fontsize=9, ha="left")

    # A distant source and its bent path illustration (minimal)
    ax.scatter([x_min + 0.2], [b], s=30, color="#2c3e50", zorder=8)
    ax.text(x_min + 0.25, b + 0.25, "Distant\nsource", fontsize=8, ha="left", va="bottom")

    ax.set_title("Curved spacetime near mass bends light", fontsize=11)
    ax.set_aspect('equal')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)


def plot_precession(ax):
    # Sun at the focus (origin)
    sun = Circle((0.0, 0.0), 0.12, facecolor="#f1c40f", edgecolor="#ad8c00", lw=1.0, zorder=5)
    ax.add_patch(sun)
    ax.text(0.0, -0.35, "Sun", ha="center", va="top", fontsize=9)

    # Precessing orbits (exaggerated for visibility)
    e = 0.32           # eccentricity
    p = 1.2            # semi-latus rectum
    thetas = np.linspace(0, 2*np.pi, 1200)
    shifts = np.linspace(0, 0.65, 5)  # perihelion shift per orbit (exaggerated)

    base_color = np.array([31, 119, 180]) / 255.0  # matplotlib C0
    for i, phi in enumerate(shifts):
        r = p / (1 + e * np.cos(thetas - phi))
        x = r * np.cos(thetas)
        y = r * np.sin(thetas)
        alpha = 0.25 + 0.15 * i
        ax.plot(x, y, color=base_color, lw=2.0, alpha=alpha, zorder=3)
        # Mark perihelion point for each orbit
        r_min = p / (1 + e)
        x_peri = r_min * np.cos(phi)
        y_peri = r_min * np.sin(phi)
        ax.scatter([x_peri], [y_peri], s=18, color=base_color, zorder=6)
        if i < len(shifts) - 1:
            # Arrow from this perihelion to the next to indicate precession
            r_min_next = p / (1 + e)
            x_next = r_min_next * np.cos(shifts[i+1])
            y_next = r_min_next * np.sin(shifts[i+1])
            arr = FancyArrowPatch((x_peri, y_peri), (x_next, y_next),
                                  arrowstyle='-|>', mutation_scale=10,
                                  color=base_color, lw=1.5, alpha=0.7, zorder=6)
            ax.add_patch(arr)

    ax.text(1.6, -0.2, "Perihelion\nshifts forward", fontsize=9, ha="left", va="top")
    ax.set_title("Precession of Mercury's orbit (explained by GR)", fontsize=11)
    ax.set_aspect('equal')
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-2.6, 2.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)


def main():
    fig = plt.figure(figsize=(12, 5.2), dpi=150)
    gs = fig.add_gridspec(1, 2, wspace=0.1)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    plot_curved_spacetime(ax1)
    plot_precession(ax2)

    fig.suptitle("General Relativity and Gravity: Mass curves spacetime; motion follows curvature", fontsize=13, y=0.98)
    outname = "general_relativity_gravity.png"
    plt.savefig(outname, bbox_inches='tight')
    print(f"Saved figure to {outname}")


if __name__ == "__main__":
    main()

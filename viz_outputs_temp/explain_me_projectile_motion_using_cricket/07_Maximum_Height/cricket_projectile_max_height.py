import numpy as np
import matplotlib.pyplot as plt


def main():
    # Parameters (feel free to tweak for different shots)
    g = 9.81  # gravity (m/s^2)
    v0 = 35.0  # initial speed of the cricket shot (m/s)
    theta_deg = 42.0  # launch angle (degrees)
    theta = np.deg2rad(theta_deg)

    # Kinematics
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, 500)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2

    # Only display the portion above ground
    y_plot = np.maximum(y, 0)

    # Maximum height and its location
    t_peak = v0 * np.sin(theta) / g
    x_peak = v0 * np.cos(theta) * t_peak
    H_max = (v0 * np.sin(theta))**2 / (2 * g)
    R = (v0**2) * np.sin(2 * theta) / g  # range (for scale/ground line)

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5))

    # Trajectory
    ax.plot(x, y_plot, color="#1b8a5a", lw=2.5, label="Ball trajectory")

    # Ground line
    ax.plot([0, R], [0, 0], color="#6d4c41", lw=3, alpha=0.6)

    # Maximum height marker and guide line
    ax.axhline(H_max, color="#888888", ls="--", lw=1, alpha=0.8)
    ax.plot(x_peak, H_max, "o", color="#d43f3a", ms=8, label="Maximum height")

    # Annotation for H_max value
    ax.annotate(
        f"H_max = {H_max:.1f} m",
        xy=(x_peak, H_max),
        xytext=(x_peak * 0.55, H_max * 1.05),
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontsize=10,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cccccc")
    )

    # Angle arc at launch
    arc_r = max(R * 0.09, 6.0)  # small arc radius for visibility
    arc_theta = np.linspace(0, theta, 80)
    ax.plot(arc_r * np.cos(arc_theta), arc_r * np.sin(arc_theta), color="#1f77b4", lw=1.4)
    ax.text(
        arc_r * 0.75 * np.cos(theta / 2),
        arc_r * 0.75 * np.sin(theta / 2),
        f"θ = {theta_deg:.0f}°",
        color="#1f77b4",
        fontsize=10,
        ha="center",
        va="center"
    )

    # Initial velocity vector (scaled for display)
    v_scale = arc_r * 0.9
    ax.annotate(
        "Initial velocity v0",
        xy=(0, 0),
        xytext=(v_scale * np.cos(theta), v_scale * np.sin(theta)),
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#1f77b4"),
        color="#1f77b4",
        fontsize=10,
        ha="center"
    )

    # Apex annotation highlighting vy = 0 at the top
    ax.annotate(
        "Vertical velocity vy = 0\nat the top",
        xy=(x_peak, H_max),
        xytext=(x_peak * 1.05, H_max * 0.6),
        arrowprops=dict(arrowstyle="->", color="#333333"),
        fontsize=9,
        color="#333333"
    )

    # Explanatory formula box
    formula_text = (
        "Maximum height:\n"
        "H_max = (v0^2 sin^2θ) / (2g)\n"
        f"= {H_max:.1f} m\n\n"
        "Time to top:\n"
        "t_up = (v0 sinθ) / g\n"
        f"= {t_peak:.2f} s"
    )
    ax.text(
        R * 0.62,
        H_max * 0.15,
        formula_text,
        fontsize=10,
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f9f9f9", ec="#cccccc")
    )

    # Styling and labels
    ax.set_xlim(-R * 0.05, R * 1.05)
    ax.set_ylim(0, H_max * 1.35)
    ax.set_xlabel("Horizontal distance x (m)")
    ax.set_ylabel("Height y (m)")
    ax.set_title("Cricket Shot: Maximum Height of the Projectile")
    ax.grid(True, ls=":", color="#bbbbbb", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    outname = "cricket_maximum_height.png"
    fig.savefig(outname, dpi=200)
    print(f"Saved figure to {outname}")


if __name__ == "__main__":
    main()

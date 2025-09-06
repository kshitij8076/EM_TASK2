import numpy as np
import matplotlib.pyplot as plt


def main():
    # Parameters
    g = 9.8  # m/s^2
    v0 = 35.0  # initial speed (m/s), typical hard hit/throw
    angle_deg = 50  # launch angle relative to horizontal (degrees)
    theta = np.deg2rad(angle_deg)
    y0 = 0.0

    # Time of flight under gravity (from y0 back to ground y=0)
    T = (2 * v0 * np.sin(theta)) / g
    t = np.linspace(0, T, 500)

    # Trajectory with gravity
    x = v0 * np.cos(theta) * t
    y = y0 + v0 * np.sin(theta) * t - 0.5 * g * t**2

    # Hypothetical straight-line path without gravity (same initial velocity)
    x_ng = v0 * np.cos(theta) * t
    y_ng = y0 + v0 * np.sin(theta) * t

    # Apex (highest point) where vy = 0
    t_apex = v0 * np.sin(theta) / g
    x_apex = v0 * np.cos(theta) * t_apex
    y_apex = y0 + v0 * np.sin(theta) * t_apex - 0.5 * g * t_apex**2

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot trajectories
    ax.plot(x, y, color="#2d8659", lw=3, label="With gravity (g = 9.8 m/s²)")
    ax.plot(x_ng, y_ng, color="#ff8c00", lw=2.5, ls="--", label="Without gravity")

    # Ground line
    x_max = max(x.max(), x_ng.max())
    y_top = max(y.max(), y_ng.max())
    ax.axhline(0, color="gray", lw=1)

    # Mark release point
    ax.plot([0], [0], marker="o", color="#2d8659", ms=6)
    ax.text(0, -3, "Bat meets ball", ha="left", va="top", fontsize=10)

    # Apex marker and annotation
    ax.plot([x_apex], [y_apex], marker="o", color="#0a3", ms=6)
    ax.annotate(
        "Highest point (vy = 0)",
        xy=(x_apex, y_apex),
        xytext=(x_apex + 0.06 * x_max, y_apex + 8),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=11,
    )

    # Vertical velocity component arrows to show gravity's effect
    scale = 0.15  # converts m/s to plotted meters for arrow length
    times = [0.15 * T, t_apex, 0.85 * T]
    for ti in times:
        xi = v0 * np.cos(theta) * ti
        yi = y0 + v0 * np.sin(theta) * ti - 0.5 * g * ti**2
        vyi = v0 * np.sin(theta) - g * ti
        y_end = yi + scale * vyi
        # Draw vertical velocity arrow (up early, zero at apex, down late)
        if abs(vyi) > 1e-6:
            ax.annotate(
                "",
                xy=(xi, y_end),
                xytext=(xi, yi),
                arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=2),
            )
        else:
            ax.plot([xi], [yi], marker="s", color="#1f77b4", ms=5)

    # Labels near velocity arrows
    xi1 = v0 * np.cos(theta) * times[0]
    yi1 = y0 + v0 * np.sin(theta) * times[0] - 0.5 * g * times[0] ** 2
    ax.text(
        xi1 + 0.02 * x_max,
        yi1 + 3,
        "Upward vy\nslows down",
        color="#1f77b4",
        fontsize=10,
    )
    xi3 = v0 * np.cos(theta) * times[2]
    yi3 = y0 + v0 * np.sin(theta) * times[2] - 0.5 * g * times[2] ** 2
    ax.text(
        xi3 + 0.02 * x_max,
        yi3 - 8,
        "Downward vy\nspeeds up",
        color="#1f77b4",
        fontsize=10,
        va="top",
    )

    # Gravity arrow and label
    ax.annotate(
        "Gravity pulls down\n(g = 9.8 m/s²)",
        xy=(0.06 * x_max, 0.85 * y_top),
        xytext=(0.06 * x_max, 0.85 * y_top - 0.28 * y_top),
        arrowprops=dict(arrowstyle="-|>", lw=2, color="black"),
        ha="center",
        va="center",
        fontsize=11,
    )

    # Annotation for no-gravity straight line
    idx = int(0.7 * len(x_ng))
    ax.annotate(
        "Without gravity:\nstraight-line flight,\nnever returns to ground",
        xy=(x_ng[idx], y_ng[idx]),
        xytext=(x_max * 0.55, y_top * 0.6),
        arrowprops=dict(arrowstyle="->", color="#ff8c00"),
        fontsize=10,
        color="#ff8c00",
    )

    # Aesthetics
    ax.set_xlim(-2, x_max * 1.05)
    ax.set_ylim(-5, y_top * 1.15)
    ax.set_xlabel("Horizontal distance (m)", fontsize=12)
    ax.set_ylabel("Height (m)", fontsize=12)
    ax.set_title("Role of Gravity in Projectile Motion (Cricket)", fontsize=14)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.25, ls="--")

    fig.tight_layout()
    fig.savefig("cricket_gravity_projectile.png", dpi=200)
    fig.savefig("cricket_gravity_projectile.pdf")


if __name__ == "__main__":
    main()

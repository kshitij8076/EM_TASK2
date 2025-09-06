import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


def trajectory_y(x, theta_rad, v0, g):
    return x * np.tan(theta_rad) - (g * x**2) / (2 * v0**2 * np.cos(theta_rad)**2)


def main():
    # Physical parameters
    g = 9.81  # m/s^2
    v0 = 25.0  # m/s (initial speed of the kick)
    angles_deg = [25, 40, 55]  # launch angles to compare

    # Prepare figure
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10
    })
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Plot ground line (football field baseline)
    ax.axhline(0, color="dimgray", lw=1)

    # Compute and plot trajectories
    colors = ["tab:blue", "tab:green", "tab:red"]
    max_range = 0.0
    max_height_overall = 0.0
    for deg, c in zip(angles_deg, colors):
        th = np.deg2rad(deg)
        R = (v0**2 * np.sin(2 * th)) / g  # horizontal range
        x = np.linspace(0, R, 300)
        y = trajectory_y(x, th, v0, g)
        ax.plot(x, y, color=c, lw=2, label=f"θ = {deg}°  (Range ≈ {R:.1f} m)")

        # Apex
        x_apex = R / 2.0
        y_apex = (v0**2 * np.sin(th)**2) / (2 * g)
        ax.plot([x_apex], [y_apex], marker="o", color=c, ms=5)

        max_range = max(max_range, R)
        max_height_overall = max(max_height_overall, y_apex)

    # Annotate one apex for clarity (use the middle angle)
    th_mid = np.deg2rad(angles_deg[1])
    R_mid = (v0**2 * np.sin(2 * th_mid)) / g
    x_apex_mid, y_apex_mid = R_mid / 2.0, (v0**2 * np.sin(th_mid)**2) / (2 * g)
    ax.annotate(
        "apex (highest point)",
        xy=(x_apex_mid, y_apex_mid),
        xytext=(x_apex_mid + 6, y_apex_mid + 3),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        ha="left",
        va="bottom",
    )

    # Illustrate initial kick direction (v0, θ)
    # Draw an arrow representing initial direction for the middle angle
    arrow_len = 10.0  # meters (schematic)
    ax.annotate(
        "",
        xy=(arrow_len * np.cos(th_mid), arrow_len * np.sin(th_mid)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", lw=2, color="black"),
    )

    # Angle arc at origin
    arc_radius = 6.0
    arc = Arc((0, 0), width=2 * arc_radius, height=2 * arc_radius, angle=0,
              theta1=0, theta2=np.rad2deg(th_mid), color="gray", lw=1.5)
    ax.add_patch(arc)
    ax.text(arc_radius * 0.9 * np.cos(th_mid / 2), arc_radius * 0.9 * np.sin(th_mid / 2),
            r"$\theta$", ha="center", va="center")
    ax.text(arrow_len * np.cos(th_mid) * 1.02, arrow_len * np.sin(th_mid) * 1.02,
            r"$v_0$", ha="left", va="bottom")

    # Place a small football marker at the kick point
    ax.plot(0, 0, marker="o", color="#8B4513", ms=8)
    ax.text(0.4, 0.4, "Kick", color="#8B4513", ha="left", va="bottom")

    # Equation box: y(x) after eliminating time t
    eq_text = (r"$y = x\tan\theta - \dfrac{g\,x^2}{2\,v_0^{2}\,\cos^{2}\theta}$" +
               "\n" +
               r"(Obtained by eliminating time $t$ from $x(t)$ and $y(t)$)")
    ax.text(0.02, 0.97, eq_text, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7", edgecolor="#BBBBBB"))

    # Labels and aesthetics
    ax.set_title("Projectile Motion of a Football: Parabolic Trajectories")
    ax.set_xlabel("Horizontal distance x (m)")
    ax.set_ylabel("Height y (m)")
    ax.grid(True, which="both", ls=":", lw=0.8, color="#CCCCCC")

    # Limits
    ax.set_xlim(0, max_range * 1.05)
    ax.set_ylim(0, max(1.0, max_height_overall * 1.25))

    # Legend
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    out_name = "parabolic_trajectory_projectile_football.png"
    fig.savefig(out_name, dpi=300)
    print(f"Saved figure to {out_name}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt


def projectile_points(v0, theta_rad, g=9.81, num=300):
    t_flight = 2 * v0 * np.sin(theta_rad) / g
    t = np.linspace(0, t_flight, num)
    x = v0 * np.cos(theta_rad) * t
    y = v0 * np.sin(theta_rad) * t - 0.5 * g * t**2
    return t, x, y, t_flight


def main():
    # Parameters (feel free to tweak for different shots)
    g = 9.81  # m/s^2
    v0 = 35.0  # initial speed (m/s)
    theta_deg = 50.0  # launch angle (degrees)
    theta = np.deg2rad(theta_deg)

    # Trajectory and key quantities
    t, x, y, t_flight = projectile_points(v0, theta, g=g, num=600)
    t_apex = v0 * np.sin(theta) / g
    H_max = (v0**2) * (np.sin(theta)**2) / (2 * g)
    R = (v0**2) * np.sin(2 * theta) / g
    x_apex = v0 * np.cos(theta) * t_apex

    # Sample positions to show time dependence (exclude endpoints)
    n_samples = 4
    t_samples = np.linspace(0, t_flight, n_samples + 2)[1:-1]
    x_s = v0 * np.cos(theta) * t_samples
    y_s = v0 * np.sin(theta) * t_samples - 0.5 * g * t_samples**2

    # Figure setup
    plt.rcParams.update({
        "font.size": 11,
        "axes.grid": True,
        "grid.color": "#e6e6e6",
        "grid.linestyle": "-",
        "grid.linewidth": 0.8,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ground line
    ax.axhline(0, color="0.2", lw=1)

    # Trajectory
    ax.plot(x, y, color="forestgreen", lw=2.5, label="Ball path")

    # Sampled positions
    ax.scatter(x_s, y_s, s=50, color="crimson", edgecolor="white", zorder=5, label="Sampled positions")
    for ti, xi, yi in zip(t_samples, x_s, y_s):
        dy = 0.04 * max(H_max, 1.0)
        va = "bottom" if ti <= t_apex else "top"
        ax.text(xi, yi + (dy if va == "bottom" else -dy), f"t = {ti:.1f} s", ha="center", va=va,
                bbox=dict(facecolor="white", alpha=0.85, boxstyle="round,pad=0.2"), fontsize=9)

    # Velocity decomposition arrows at launch
    # Scales chosen for clarity (do not reflect exact magnitudes)
    len_scale = 0.18 * R
    # v0 arrow
    ax.annotate("", xy=(len_scale * np.cos(theta), len_scale * np.sin(theta)), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="tab:blue"))
    ax.text(len_scale * np.cos(theta) * 1.05, len_scale * np.sin(theta) * 1.05,
            r"$v_0$", color="tab:blue", ha="left", va="bottom")
    # v0*cos(theta) arrow (horizontal)
    ax.annotate("", xy=(0.22 * R, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2, color="tab:orange"))
    ax.text(0.11 * R, 0.06 * max(H_max, 1.0), r"$v_\!0\cos\theta$", color="tab:orange", ha="center")
    # v0*sin(theta) arrow (vertical)
    ax.annotate("", xy=(0, 0.5 * H_max if H_max > 0 else 5.0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2, color="tab:purple"))
    ax.text(0.02 * R, 0.25 * (H_max if H_max > 0 else 5.0), r"$v_\!0\sin\theta$", color="tab:purple",
            rotation=90, va="center")

    # Gravity indication near apex
    ax.annotate("", xy=(x_apex, H_max - 0.35 * H_max), xytext=(x_apex, H_max + 0.15 * H_max),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="black"))
    ax.text(x_apex + 0.01 * R, H_max, r"$g$ downward", va="center", ha="left")

    # Show H_max and Range R with dashed helpers
    ax.hlines(H_max, 0, x_apex, colors="gray", linestyles="--", linewidth=1)
    ax.text(0.01 * R, H_max + 0.02 * H_max, r"$H_{\max}$", ha="left", va="bottom")
    ax.vlines(R, 0, 0.12 * max(H_max, 1.0), colors="gray", linestyles="--", linewidth=1)
    ax.text(R, 0.14 * max(H_max, 1.0), r"$R$", ha="center", va="bottom")

    # Explanatory text boxes
    eq_text = (r"Kinematic equations (cricket ball):\n"
               r"$x(t) = v_0\cos\theta\, t$\n"
               r"$y(t) = v_0\sin\theta\, t - \tfrac{1}{2} g t^2$\n"
               r"$v_x(t) = v_0\cos\theta$ (constant)\n"
               r"$v_y(t) = v_0\sin\theta - g t$")
    ax.text(0.02, 0.98, eq_text, transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95), fontsize=10)

    summary_text = (f"Given: v0 = {v0:.0f} m/s, "
                    + r"$\theta$" + f" = {theta_deg:.0f}°\n"
                    + f"Time of flight ≈ {t_flight:.2f} s\n"
                    + f"Range R ≈ {R:.1f} m\n"
                    + f"Max height H ≈ {H_max:.1f} m")
    ax.text(0.98, 0.98, summary_text, transform=ax.transAxes, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.95), fontsize=10)

    # Additional hints
    ax.text(0.60 * R, 0.08 * max(H_max, 1.0), "Horizontal motion:\nconstant speed",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # Labels and limits
    ax.set_title("Kinematic Equations for a Lofted Cricket Shot")
    ax.set_xlabel("Horizontal distance x (m)")
    ax.set_ylabel("Vertical height y (m)")

    ax.set_xlim(-0.02 * R, 1.05 * R)
    ax.set_ylim(-0.02 * max(H_max, 1.0), 1.25 * max(H_max, 1.0))

    ax.legend(loc="lower right", frameon=True)

    out_name = "projectile_motion_cricket_kinematics.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=200, bbox_inches="tight")
    # plt.show()
    print(f"Saved figure to {out_name}")


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc


def trajectory(v0, theta_deg, g=9.81, y0=1.0, n=300):
    theta = np.radians(theta_deg)
    vy0 = v0 * np.sin(theta)
    vx0 = v0 * np.cos(theta)
    tf = (vy0 + np.sqrt(vy0**2 + 2 * g * y0)) / g
    t = np.linspace(0, tf, n)
    x = vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    # Apex
    t_peak = vy0 / g
    t_peak = np.clip(t_peak, 0, tf)
    x_peak = vx0 * t_peak
    y_peak = y0 + vy0 * t_peak - 0.5 * g * t_peak**2
    return x, y, tf, x[-1], (x_peak, y_peak)


def main():
    # Physics/setup
    g = 9.81  # m/s^2
    v0 = 25.0  # m/s (typical big hit in cricket, ignoring air resistance)
    y0 = 1.0   # m (approximate bat contact height)
    angles = [25, 35, 45, 55, 65]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    # Compute trajectories
    curves = []
    max_x = 0
    max_y = 0
    apex_45 = None
    for th, c in zip(angles, colors):
        x, y, tf, xr, apex = trajectory(v0, th, g=g, y0=y0)
        curves.append((th, c, x, y, xr, apex))
        max_x = max(max_x, xr)
        max_y = max(max_y, np.max(y))
        if th == 45:
            apex_45 = apex

    # Figure
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    # Ground line
    ax.axhline(0, color="black", lw=1)

    # Boundary marker (typical ~65 m)
    boundary_x = 65
    ax.axvline(boundary_x, color="gray", lw=1.2, ls="--")
    ax.text(boundary_x + 0.8, 0.5, "Boundary ~65 m", rotation=90, va="bottom", ha="left", color="gray")

    # Plot trajectories
    for th, c, x, y, xr, apex in curves:
        ax.plot(x, y, color=c, lw=2.2, label=f"θ = {th}°")

    # Highlight apex for 45°
    if apex_45 is not None:
        ax.plot(apex_45[0], apex_45[1], 'o', color="#2ca02c")
        ax.annotate(
            "Peak height (θ=45°)",
            xy=apex_45, xytext=(apex_45[0] + 6, apex_45[1] + 3),
            arrowprops=dict(arrowstyle="->", color="#2ca02c"),
            color="#2ca02c"
        )

    # Labels and title
    ax.set_title("Cricket projectile: Initial velocity and angle of projection (no air resistance)")
    ax.set_xlabel("Distance along ground (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(alpha=0.25)

    # Limits
    ax.set_xlim(0, max(max_x * 1.08, boundary_x + 5))
    ax.set_ylim(0, max_y * 1.15)

    # Legend
    leg = ax.legend(title="Launch angle θ", loc="upper right", frameon=True)
    leg._legend_box.align = "left"

    # Explanatory text
    ax.text(0.02, 0.97,
            f"Same launch speed v0 = {v0:.0f} m/s; higher θ → higher peak, lower range\nLower θ → flatter shot, longer range (max near 45°)",
            transform=ax.transAxes, va="top", ha="left", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

    # Inset: velocity components diagram
    inset = fig.add_axes([0.63, 0.53, 0.32, 0.38])  # [left, bottom, width, height]
    inset.set_xlim(0, 1)
    inset.set_ylim(0, 1)
    inset.set_aspect('equal')
    inset.axis('off')

    theta_demo_deg = 45
    theta_demo = np.radians(theta_demo_deg)
    L = 0.78  # length scale inside inset
    x_end = L * np.cos(theta_demo)
    y_end = L * np.sin(theta_demo)

    # Draw component arrows
    # v0 vector
    inset.arrow(0, 0, x_end, y_end, width=0.008, head_width=0.05, length_includes_head=True, color="#111111")
    # v0x component
    inset.arrow(0, 0, x_end, 0, width=0.006, head_width=0.04, length_includes_head=True, color="#1f77b4")
    # v0y component (from x-component tip up)
    inset.arrow(x_end, 0, 0, y_end, width=0.006, head_width=0.04, length_includes_head=True, color="#ff7f0e")

    # Angle arc
    arc = Arc((0, 0), 0.35, 0.35, angle=0, theta1=0, theta2=theta_demo_deg, color="gray", lw=1.2)
    inset.add_patch(arc)
    inset.text(0.22 * np.cos(theta_demo/2), 0.22 * np.sin(theta_demo/2), "θ", ha="center", va="center", color="gray")

    # Labels in inset
    inset.text(x_end * 0.5, -0.07, "v0 cos θ", ha="center", va="top", color="#1f77b4")
    inset.text(x_end + 0.03, y_end * 0.5, "v0 sin θ", ha="left", va="center", color="#ff7f0e")
    inset.text(x_end * 0.55, y_end * 0.55, "v0", ha="left", va="bottom", color="#111111")
    inset.text(0.02, 0.98, "Velocity components at launch", ha="left", va="top", fontsize=10)

    # Save figure
    out_name = "cricket_projectile_initial_velocity_angle.png"
    fig.savefig(out_name, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {out_name}")


if __name__ == "__main__":
    main()

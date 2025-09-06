import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Rectangle


def trajectory(v0, theta_deg, g=9.81, n=300):
    theta = np.radians(theta_deg)
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, n)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    return x, y


def range_of(v0, theta_deg, g=9.81):
    theta = np.radians(theta_deg)
    return (v0**2 * np.sin(2*theta)) / g


def max_height(v0, theta_deg, g=9.81):
    theta = np.radians(theta_deg)
    return (v0**2 * (np.sin(theta))**2) / (2*g)


def apex_x(v0, theta_deg, g=9.81):
    theta = np.radians(theta_deg)
    return (v0**2 * np.sin(2*theta)) / (2*g)


if __name__ == "__main__":
    # Parameters (representative of a cricket shot)
    v0 = 35.0  # m/s
    g = 9.81   # m/s^2
    angles = [25, 45, 65]  # low, balanced, high

    # Compute trajectories
    data = {}
    ranges = []
    for ang in angles:
        x, y = trajectory(v0, ang, g)
        data[ang] = (x, y)
        ranges.append(range_of(v0, ang, g))

    Rmax = max(ranges)

    # Figure setup
    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(10, 6))

    # Draw a simple grass field under y=0
    grass_left = -0.06 * Rmax
    grass_right = 1.06 * Rmax
    grass_depth = 0.35
    ax.add_patch(Rectangle((grass_left, -grass_depth), grass_right - grass_left, grass_depth,
                           facecolor="#a7d99b", edgecolor="none", zorder=0))
    ax.axhline(0, color="forestgreen", lw=3, zorder=1)

    # Colors for trajectories
    colors = {25: "#ff7f0e", 45: "#1f77b4", 65: "#2ca02c"}
    styles = {25: (0, (4, 3)), 45: "-", 65: (0, (4, 3))}
    alphas = {25: 0.9, 45: 1.0, 65: 0.9}

    # Plot trajectories
    for ang in angles:
        x, y = data[ang]
        ax.plot(x, y, color=colors[ang], lw=2.5 if ang == 45 else 2.0,
                linestyle=styles[ang], alpha=alphas[ang], label=f"{ang}°")

    # Primary (featured) angle for component illustration
    theta_deg = 45
    theta = np.radians(theta_deg)
    x45, y45 = data[theta_deg]

    # Mark discrete ball positions along the 45° path (cricket ball)
    t_flight = 2 * v0 * np.sin(theta) / g
    t_marks = np.linspace(0, t_flight, 7)
    x_marks = v0 * np.cos(theta) * t_marks
    y_marks = v0 * np.sin(theta) * t_marks - 0.5 * g * t_marks**2
    ax.scatter(x_marks, y_marks, s=28, color="#c71f1f", edgecolor="white", zorder=5, label="Ball positions")

    # Component arrows at origin for 45°
    L = 0.18 * Rmax  # visualization length scale for velocity vectors
    vx_ratio = np.cos(theta)
    vy_ratio = np.sin(theta)

    # Resultant v0 arrow
    ax.annotate("", xy=(L*vx_ratio, L*vy_ratio), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.8, color="#6a3d9a"), zorder=6)
    ax.text(L*vx_ratio*1.05, L*vy_ratio*1.05, "Initial speed v0\n(launch angle θ)", color="#6a3d9a",
            ha="left", va="bottom")

    # Horizontal component v0x
    ax.annotate("", xy=(L*vx_ratio, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#d45500"), zorder=6)
    ax.text(L*vx_ratio*0.5, -0.05*L, "v0x = v0 cos θ", color="#d45500", ha="center", va="top")

    # Vertical component v0y
    ax.annotate("", xy=(0, L*vy_ratio), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#2ca02c"), zorder=6)
    ax.text(-0.02*Rmax, L*vy_ratio*0.5, "v0y = v0 sin θ", color="#2ca02c", ha="right", va="center")

    # Projection guides (dashed)
    ax.plot([L*vx_ratio, L*vx_ratio], [0, L*vy_ratio], linestyle=(0, (3, 3)), color="#666666", lw=1.2)
    ax.plot([0, L*vx_ratio], [L*vy_ratio, L*vy_ratio], linestyle=(0, (3, 3)), color="#666666", lw=1.2)

    # Angle arc for θ
    arc_r = 0.65 * L
    arc = Arc((0, 0), width=2*arc_r, height=2*arc_r, angle=0, theta1=0, theta2=theta_deg,
              color="#6a3d9a", lw=2)
    ax.add_patch(arc)
    ax.text(arc_r*0.85*np.cos(np.radians(theta_deg/2)), arc_r*0.85*np.sin(np.radians(theta_deg/2)),
            "θ", color="#6a3d9a", ha="center", va="center")

    # Range annotation (for 45°)
    R = range_of(v0, theta_deg, g)
    y_range_annot = -0.22 * grass_depth
    ax.annotate("", xy=(R, y_range_annot), xytext=(0, y_range_annot),
                arrowprops=dict(arrowstyle="<->", lw=2, color="#333333"))
    ax.text(R/2, y_range_annot - 0.03, "Horizontal distance (Range)\nmainly influenced by v0x",
            ha="center", va="top", color="#333333")

    # Max height annotation (for 45°)
    H = max_height(v0, theta_deg, g)
    x_ap = apex_x(v0, theta_deg, g)
    ax.annotate("", xy=(x_ap, H), xytext=(x_ap, 0),
                arrowprops=dict(arrowstyle="<->", lw=2, color="#333333"))
    ax.text(x_ap + 0.02*Rmax, H/2, "Maximum height\nset by v0y", ha="left", va="center", color="#333333")

    # Narrative labels near low/high angle paths
    ax.text(data[25][0].max()*0.75, max(data[25][1])*0.6, "Lower angle:\nmore v0x,\nflatter path", color=colors[25],
            ha="center", va="center")
    ax.text(data[65][0].max()*0.55, max(data[65][1])*0.75, "Higher angle:\nmore v0y,\nhigher but shorter", color=colors[65],
            ha="center", va="center")

    # Legend for trajectories
    ax.legend(title="Launch angles", loc="upper right", frameon=True)

    # Axes formatting
    ax.set_xlim(grass_left, grass_right)
    y_top = max(max(y) for _, y in data.values())
    ax.set_ylim(-grass_depth, y_top * 1.15)
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Components of Initial Velocity: Cricket Projectile Motion")
    ax.grid(True, linestyle=(0, (3, 3)), alpha=0.4)

    # Tidy and save
    plt.tight_layout()
    out_name = "cricket_projectile_components.png"
    plt.savefig(out_name, dpi=200)
    # Also show if run interactively
    # plt.show()
    print(f"Saved figure to {out_name}")

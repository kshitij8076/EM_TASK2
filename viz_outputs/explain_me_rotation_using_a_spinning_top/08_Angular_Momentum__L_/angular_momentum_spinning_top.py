import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.hypot(v[0], v[1])
    return v / n if n != 0 else v


def draw_circular_arrow(ax, center, radius, start_deg, end_deg, color="tab:blue", lw=2, head_len=0.18):
    theta = np.radians(np.linspace(start_deg, end_deg, 100))
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, color=color, lw=lw)
    # Arrowhead at end
    theta_end = np.radians(end_deg)
    end_pt = np.array([center[0] + radius * np.cos(theta_end), center[1] + radius * np.sin(theta_end)])
    t_hat = np.array([-np.sin(theta_end), np.cos(theta_end)])  # tangent direction
    start_head = end_pt - t_hat * head_len
    ax.annotate("", xy=end_pt, xytext=start_head,
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color))


def draw_top(ax, origin, u, ring_radius, L_len, title_text):
    u = unit(u)
    p = np.array([-u[1], u[0]])  # perpendicular in-plane

    # Ground line and contact point
    ax.plot([-10, 10], [origin[1], origin[1]], color="#888", lw=1.2, alpha=0.6)
    ax.plot(origin[0], origin[1], marker='o', color="#555", ms=4, zorder=5)

    # Axis line
    axis_len = 3.6
    tip = np.array(origin)
    top_point = tip + u * axis_len
    ax.plot([tip[0], top_point[0]], [tip[1], top_point[1]], color="#333", lw=2)

    # Top body (simplified cone-like profile)
    top_h = 3.0
    r_max = ring_radius * 1.35
    t_vals = np.linspace(0, top_h, 50)
    r_vals = r_max * (t_vals / top_h)  # linear flare from tip
    left_edge = [tip + u * t + p * r for t, r in zip(t_vals, r_vals)]
    right_edge = [tip + u * t - p * r for t, r in zip(t_vals[::-1], r_vals[::-1])]
    body_pts = np.vstack([np.array([tip[0], tip[1]]), np.array(left_edge), np.array(right_edge)])
    body_poly = Polygon(body_pts, closed=True, facecolor="#f6c27a", edgecolor="#8a5a2b", lw=1.5, alpha=0.9)
    ax.add_patch(body_poly)

    # Mass distribution ring (dots) and rotational arrow
    ring_center = tip + u * (top_h * 0.55)
    ring_r = ring_radius
    angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
    ring_pts = np.column_stack([ring_center[0] + ring_r * np.cos(angles),
                                ring_center[1] + ring_r * np.sin(angles)])
    ax.scatter(ring_pts[:,0], ring_pts[:,1], s=14, color="#4c78a8", alpha=0.9, zorder=6)

    # Rotational (omega) curved arrow
    draw_circular_arrow(ax, ring_center, ring_r, start_deg=40, end_deg=320, color="tab:blue", lw=2)
    # omega label
    omega_pos = ring_center + np.array([ring_r * 0.9, ring_r * 0.1])
    ax.text(omega_pos[0], omega_pos[1], r"$\omega$", color="tab:blue", fontsize=12, ha='left', va='bottom')

    # Angular momentum vector L (along axis)
    L_start = tip + u * 0.35
    L_end = L_start + u * L_len
    ax.annotate("", xy=L_end, xytext=L_start,
                arrowprops=dict(arrowstyle='-|>', lw=2.5, color="crimson"))
    # L label slightly beyond the arrow head
    L_label_pos = L_end + unit(u) * 0.15
    ax.text(L_label_pos[0], L_label_pos[1], r"$\mathbf{L}$", color="crimson", fontsize=13, ha='left', va='center')

    # Title/descriptor
    ax.text(top_point[0], top_point[1] + 0.5, title_text, fontsize=12, ha='center', va='bottom', weight='bold')

    # Indicate I effect
    hint_vec = unit(p)  # sideways
    hint_start = ring_center + hint_vec * (ring_r + 0.15)
    ax.annotate("",
                xy=hint_start,
                xytext=hint_start + hint_vec * 0.001,  # tiny to force arrowhead placement
                arrowprops=dict(arrowstyle='-|>', color="#4c78a8"))
    ax.text(hint_start[0] + hint_vec[0] * 0.15,
            hint_start[1] + hint_vec[1] * 0.15,
            "mass around axis\n(moment of inertia, I)",
            fontsize=9, color="#1f4f7a", ha='left', va='center')


def main():
    plt.rcParams.update({
        'font.size': 11,
        'axes.facecolor': 'white'
    })

    fig, ax = plt.subplots(figsize=(12, 6))

    # Shared axis direction (slightly tilted from vertical)
    theta_deg = 78
    u = np.array([np.cos(np.radians(theta_deg)), np.sin(np.radians(theta_deg))])

    # Left: Small angular momentum (smaller I and/or smaller omega)
    origin_left = np.array([-4.0, -0.2])
    draw_top(
        ax,
        origin_left,
        u,
        ring_radius=0.55,
        L_len=1.1,
        title_text="Small angular momentum: small I and/or ω"
    )

    # Right: Large angular momentum (larger I and/or larger omega)
    origin_right = np.array([4.0, -0.2])
    draw_top(
        ax,
        origin_right,
        u,
        ring_radius=1.15,
        L_len=2.2,
        title_text="Large angular momentum: large I and/or ω"
    )

    # Global caption
    ax.text(0, 5.3,
            "Angular momentum L = I · ω. Larger L → more resistance to changes (e.g., toppling).\n"
            "Direction of L aligns with the rotation axis (right-hand rule).",
            fontsize=12, ha='center', va='center')

    ax.set_xlim(-9, 9)
    ax.set_ylim(-1.0, 5.8)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')

    out_name = "angular_momentum_spinning_top.png"
    plt.tight_layout()
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {out_name}")


if __name__ == "__main__":
    main()

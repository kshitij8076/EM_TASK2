import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc


def rotate_points(points, angle_deg, origin=(0.0, 0.0)):
    ang = np.deg2rad(angle_deg)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts = np.asarray(points)
    origin = np.asarray(origin)
    return (pts - origin) @ R.T + origin


def draw_circular_arrow(ax, center=(0, 0), radius=0.9, theta1=110, theta2=20, color="tab:red", lw=2, label=None):
    # Draw clockwise arc from theta1 to theta2 (degrees)
    arc = Arc(center, width=2*radius, height=2*radius, angle=0,
              theta1=theta1, theta2=theta2, color=color, lw=lw)
    ax.add_patch(arc)
    # Arrow head at theta2, pointing clockwise (tangent direction)
    t2 = np.deg2rad(theta2)
    end = np.array(center) + radius * np.array([np.cos(t2), np.sin(t2)])
    # Tangent direction for clockwise sense at theta2
    tangent = np.array([np.sin(t2), -np.cos(t2)])
    head_len = 0.18 * radius
    start_head = end - head_len * tangent
    ax.annotate("", xy=end, xytext=start_head,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=12))
    if label is not None:
        # Place label near the middle of the arc
        tm = np.deg2rad((theta1 + theta2) / 2)
        mid = np.array(center) + (radius + 0.18) * np.array([np.cos(tm), np.sin(tm)])
        ax.text(mid[0], mid[1], label, color=color, fontsize=11, weight='bold', ha='center', va='center')


def make_figure(savepath="newton_second_law_rotation_spinning_top.png"):
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.15], wspace=0.35)

    # Left panel: conceptual diagram of a tilted spinning top with torque
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlim(-2.0, 2.0)
    ax1.set_ylim(-0.6, 3.0)

    # Define a simple top silhouette in local coordinates with pivot (tip) at (0, 0)
    top_outline = np.array([
        [0.00, 0.00],   # tip/pivot
        [0.60, 0.70],
        [0.45, 1.20],
        [0.18, 1.80],
        [0.05, 2.30],
        [0.00, 2.50],
        [-0.05, 2.30],
        [-0.18, 1.80],
        [-0.45, 1.20],
        [-0.60, 0.70],
        [0.00, 0.00]    # close
    ])

    # Center of mass along the symmetry axis (approx)
    com_local = np.array([0.0, 1.25])

    # Tilt the top to the right by -20 degrees
    tilt_deg = -20
    top_rot = rotate_points(top_outline, tilt_deg)
    com = rotate_points(com_local, tilt_deg)

    # Draw ground/contact point
    ax1.plot([-1.6, 1.6], [0, 0], color="#888888", lw=1.5)
    ax1.plot(0, 0, 'o', color='k', ms=5)
    ax1.text(0.05, -0.1, "pivot", ha='left', va='top')

    # Draw the top body
    body = Polygon(top_rot, closed=True, facecolor="#cfe1ff", edgecolor="#2a4b8d", lw=2)
    ax1.add_patch(body)

    # Draw r vector from pivot to COM
    ax1.annotate("", xy=(com[0], com[1]), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', lw=2, color="tab:purple"))
    # Label r near the middle
    mid_r = 0.5 * com
    ax1.text(mid_r[0] + 0.05, mid_r[1] + 0.05, "r", color="tab:purple", fontsize=11)

    # Draw mg from COM downward
    g_len = 1.1
    mg_end = com + np.array([0.0, -g_len])
    ax1.annotate("", xy=(mg_end[0], mg_end[1]), xytext=(com[0], com[1]),
                 arrowprops=dict(arrowstyle='->', lw=2, color="tab:green"))
    # Label mg near mid of arrow
    mg_mid = (com + mg_end) / 2
    ax1.text(mg_mid[0] + 0.05, mg_mid[1] - 0.05, "mg", color="tab:green", fontsize=11)

    # Draw a circular arrow indicating torque (and alpha direction)
    draw_circular_arrow(ax1, center=(0, 0), radius=0.9, theta1=110, theta2=20, color="tab:red", lw=2, label="τ, α")

    # Equation annotation
    ax1.text(-1.85, 2.75, "Newton's Second Law for Rotation", fontsize=12, weight='bold')
    ax1.text(-1.85, 2.45, r"τ = I·α", fontsize=14)
    ax1.text(-1.85, 2.15, "For the top: torque from r × mg\ncauses angular acceleration.", fontsize=10)

    # Right panel: τ vs α for different I (slope = I)
    ax2 = fig.add_subplot(gs[0, 1])
    alpha = np.linspace(0, 12, 300)
    I_vals = [0.25, 0.6, 1.2]
    labels = ["small I (light/narrow top)", "medium I", "large I (heavy/wide top)"]
    colors = ["tab:blue", "tab:orange", "tab:red"]

    for I, lab, c in zip(I_vals, labels, colors):
        tau = I * alpha
        ax2.plot(alpha, tau, color=c, lw=2, label=f"{lab}\nI = {I:.2f}")

    ax2.set_xlabel("angular acceleration, α (rad/s²)")
    ax2.set_ylabel("torque, τ (N·m)")
    ax2.set_title("τ = I·α: Larger I ⇒ steeper line (harder to accelerate)")
    ax2.grid(True, ls='--', alpha=0.4)

    # Highlight same torque and show different α results
    tau_star = 6.0
    ax2.axhline(tau_star, color="#666666", lw=1.2, ls=":")
    ax2.text(0.3, tau_star + 0.25, r"same τ*", color="#444444")

    for I, c in zip(I_vals, colors):
        alpha_star = tau_star / I
        ax2.plot([alpha_star], [tau_star], marker='o', color=c)
        ax2.vlines(alpha_star, 0, tau_star, color=c, lw=1, ls=":", alpha=0.8)

    ax2.legend(frameon=False, loc="upper left")

    fig.suptitle("Newton's Second Law for Rotation — Spinning Top", fontsize=14, weight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(savepath, dpi=200, bbox_inches='tight')
    print(f"Saved figure to {savepath}")


if __name__ == "__main__":
    make_figure()

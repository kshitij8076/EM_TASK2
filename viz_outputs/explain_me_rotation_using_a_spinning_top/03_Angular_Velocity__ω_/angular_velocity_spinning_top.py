import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def draw_top(ax, tip, v_axis, L_total=3.0, w_max=1.0, facecolor="#e6a86b", edgecolor="#8a5a2b"):
    v = unit(v_axis)
    n = np.array([-v[1], v[0]])  # perpendicular in the drawing plane

    # Centerline parameter from tip (t=0) to top (t=L_total)
    t = np.linspace(0.0, L_total, 260)

    # Width profile: belly near middle; zero at ends
    # Smooth, single-bulge profile
    w = w_max * np.power(np.sin(np.pi * t / L_total), 0.7)

    # Slightly cinch near the neck (toward the top) for a realistic top silhouette
    neck = 0.25 * w_max * np.exp(-((t - 0.85 * L_total) / (0.12 * L_total)) ** 2)
    w = np.maximum(w - neck, 0.0)

    centerline = tip[None, :] + t[:, None] * v[None, :]
    right = centerline + w[:, None] * n[None, :]
    left = centerline - w[:, None] * n[None, :]

    poly = np.vstack([right, left[::-1]])
    ax.add_patch(Polygon(poly, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=2))

    # Small top knob
    top_pt = tip + L_total * v
    ax.add_patch(Circle(top_pt + 0.03 * v, radius=0.08, facecolor=edgecolor, edgecolor=edgecolor, zorder=3))

    # Tip accent
    ax.add_patch(Circle(tip - 0.02 * v, radius=0.05, facecolor=edgecolor, edgecolor=edgecolor, zorder=3))

    return {
        "tip": tip,
        "top": top_pt,
        "axis_dir": v,
        "normal_dir": n,
        "L": L_total
    }


def draw_ring_arrows(ax, center, radius=0.55, color="#1f77b4", n_arrows=3, ccw=True, lw=1.4, z=4):
    # Draw a thin circle to suggest rotation path
    circle = plt.Circle(center, radius, fill=False, ec=color, lw=1.0, ls=(0, (3, 3)), alpha=0.9, zorder=z)
    ax.add_patch(circle)

    # Tangential arrows around the ring
    sgn = 1.0 if ccw else -1.0
    angles = np.linspace(0, 2 * np.pi, n_arrows, endpoint=False) + (0.0 if ccw else np.pi / n_arrows)
    L = 0.38 * radius
    for phi in angles:
        p = np.array([center[0] + radius * np.cos(phi), center[1] + radius * np.sin(phi)])
        tvec = sgn * np.array([-np.sin(phi), np.cos(phi)])
        start = p - 0.5 * L * tvec
        end = p + 0.5 * L * tvec
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0),
            zorder=z + 1,
        )


def draw_omega(ax, base, v_axis, length=1.0, color="#1f77b4", label=True, z=5):
    v = unit(v_axis)
    end = base + length * v
    ax.annotate(
        "",
        xy=end,
        xytext=base,
        arrowprops=dict(arrowstyle="->", lw=2.2, color=color),
        zorder=z,
    )
    if label:
        text_pos = end + 0.12 * np.array([1, 1])
        ax.text(
            text_pos[0],
            text_pos[1],
            r"$\\omega$",
            color=color,
            fontsize=13,
            weight="bold",
            zorder=z + 1,
        )


def panel(ax, tilt_deg, omega_len, title, stability_note, body_color, rng_seed=0):
    np.random.seed(rng_seed)

    # Ground line
    ax.axhline(0, color="#777777", lw=1.2, alpha=0.6, zorder=1)

    theta = np.deg2rad(tilt_deg)
    v_axis = np.array([np.sin(theta), np.cos(theta)])  # 0 deg -> vertical up
    tip = np.array([0.0, 0.0])

    geom = draw_top(ax, tip, v_axis, L_total=3.0, w_max=1.0, facecolor=body_color)

    # Spin ring around mid-body
    ring_center = geom["tip"] + 1.5 * geom["axis_dir"]
    draw_ring_arrows(ax, ring_center, radius=0.55, color="#1f77b4", n_arrows=3, ccw=True)

    # Angular velocity vector along axis (right-hand rule direction)
    omega_base = geom["tip"] + 0.95 * geom["axis_dir"]
    draw_omega(ax, omega_base, geom["axis_dir"], length=omega_len, color="#1f77b4", label=True)

    # Light dashed axis guide
    ax.plot(
        [geom["tip"][0], geom["top"][0]],
        [geom["tip"][1], geom["top"][1]],
        color="#2b3a42",
        lw=1.0,
        ls=(0, (4, 4)),
        alpha=0.6,
        zorder=1,
    )

    # Title and notes
    ax.set_title(title, fontsize=13, pad=8)

    ax.text(
        ring_center[0] + 0.9,
        ring_center[1] - 0.1,
        "rotation",
        fontsize=10,
        color="#1f77b4",
    )

    ax.text(
        geom["top"][0] + 0.15,
        geom["top"][1] - 0.2,
        stability_note,
        fontsize=10,
        color="#2b3a42",
    )

    # A small curved arrow near the axis base to reinforce sense of rotation
    # Here we just reuse ring arrows but smaller
    draw_ring_arrows(ax, geom["tip"] + 0.45 * geom["axis_dir"], radius=0.25, color="#1f77b4", n_arrows=2, ccw=True, lw=1.2, z=4)

    # Bounds and aesthetics
    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-0.6, 3.6)
    ax.axis("off")


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.5))

    # Left: lower omega, more tilt (less stable)
    panel(
        axes[0],
        tilt_deg=25,
        omega_len=1.0,
        title="Lower c9 (slower spin)",
        stability_note="More tilt = less stable",
        body_color="#e6a86b",
    )

    # Right: higher omega, nearly upright (more stable)
    panel(
        axes[1],
        tilt_deg=0,
        omega_len=1.9,
        title="Higher c9 (faster spin)",
        stability_note="More upright = more stable",
        body_color="#e1b97c",
    )

    # Global annotations
    fig.suptitle(
        "Angular Velocity (\u03c9) for a Spinning Top",
        fontsize=16,
        y=0.98,
        weight="bold",
    )

    fig.text(
        0.5,
        0.07,
        "\u03c9 quantifies how fast the top rotates (radians/second). \nDirection follows the right-hand rule: curl your right-hand fingers with the spin; your thumb points along \u03c9 (the rotation axis).",
        ha="center",
        va="center",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    outname = "angular_velocity_spinning_top.png"
    plt.savefig(outname, dpi=300)
    print(f"Saved figure to {outname}")

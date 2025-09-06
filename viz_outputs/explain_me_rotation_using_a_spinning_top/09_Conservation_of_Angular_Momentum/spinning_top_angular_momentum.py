import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Rectangle
from matplotlib import transforms


def rotate_vec(v, theta_rad):
    R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)],
                  [np.sin(theta_rad),  np.cos(theta_rad)]])
    return R @ v


def draw_top(ax, apex, angle_deg=0, scale=1.6,
             body_color="#6aaed6", edge_color="#2b5b84", stem_color="#2b5b84"):
    """
    Draw a stylized spinning top with its tip (apex) at the given coordinate.
    The top is constructed in local coordinates with apex at (0,0) and then
    rotated by angle_deg and translated so the apex sits at 'apex'.

    Returns:
        dict with keys: 'apex', 'axis_dir', 'com' (approx center of mass location)
    """
    ax.set_aspect('equal')
    s = scale
    theta = np.deg2rad(angle_deg)

    # Local geometry (apex at (0,0), symmetry axis along +y)
    # Ellipse (body)
    ell_center_local = (0.0, 0.65*s)
    ell_width, ell_height = 1.2*s, 0.8*s

    # Tip (triangle)
    tip_base_y = 0.25*s
    tip_half_w = 0.12*s
    tip_pts = [(-tip_half_w, tip_base_y), (tip_half_w, tip_base_y), (0.0, 0.0)]

    # Stem (small rectangle on top)
    stem_w, stem_h = 0.12*s, 0.30*s
    stem_bl = (-stem_w/2.0, ell_center_local[1] + ell_height/2.0)  # bottom-left

    # Transformation: rotate in place then translate to apex
    tr = transforms.Affine2D().rotate_deg(angle_deg).translate(apex[0], apex[1]) + ax.transData

    # Patches
    body = Ellipse(ell_center_local, ell_width, ell_height, facecolor=body_color,
                   edgecolor=edge_color, lw=2, transform=tr, zorder=3)
    tip = Polygon(tip_pts, closed=True, facecolor=body_color, edgecolor=edge_color, lw=2,
                  transform=tr, zorder=3)
    stem = Rectangle(stem_bl, stem_w, stem_h, facecolor=stem_color, edgecolor=edge_color,
                     lw=2, transform=tr, zorder=4)

    ax.add_patch(body)
    ax.add_patch(tip)
    ax.add_patch(stem)

    # Compute global points for annotations
    ell_center_global = rotate_vec(np.array(ell_center_local), theta) + np.array(apex)
    com_approx = ell_center_global  # good enough for illustration
    axis_dir = rotate_vec(np.array([0.0, 1.0]), theta)  # symmetry axis direction

    return {"apex": np.array(apex), "axis_dir": axis_dir/np.linalg.norm(axis_dir), "com": com_approx}


def draw_arrow(ax, start, direction, length, color, lw=2.5, linestyle='solid', label=None, z=5):
    start = np.array(start)
    end = start + length * (direction / (np.linalg.norm(direction) + 1e-12))
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='-|>', lw=lw, color=color, linestyle=linestyle), zorder=z)
    if label is not None:
        # Place label slightly offset near the end
        offset = 0.06 * np.array([1, 1])
        ax.text(*(end + offset), label, color=color, fontsize=10, weight='bold', zorder=z+1)


def setup_panel(ax, title):
    ax.set_xlim(-2.3, 2.3)
    ax.set_ylim(-1.4, 2.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, weight='bold', pad=8)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.suptitle('Conservation of Angular Momentum — Spinning Top', fontsize=15, weight='bold')

    # Common ground line (smooth surface)
    ground_y = -1.0

    # Panel A: No external torque (tau = 0)
    setup_panel(ax1, 'A. No external torque (τ = 0)')
    ax1.plot([-3, 3], [ground_y, ground_y], color='#888888', lw=2)

    # Draw upright top
    apex_A = (0.0, ground_y)
    info_A = draw_top(ax1, apex=apex_A, angle_deg=0, scale=1.6)

    # Angular momentum L (constant) — draw one vector
    L_len = 1.2
    draw_arrow(ax1, start=info_A["com"], direction=info_A["axis_dir"], length=L_len,
               color='#2ca02c', lw=3, label='L (constant)')

    # Small note
    ax1.text(-2.2, 1.9, 'No external torque → angular momentum stays the same',
             fontsize=10, color='#333333')

    # Panel B: External torque (friction)
    setup_panel(ax2, 'B. External torque (friction)')
    ax2.plot([-3, 3], [ground_y, ground_y], color='#888888', lw=2)

    # Draw tilted top (same apex/contact), slight tilt
    tilt_deg = 18
    apex_B = (0.0, ground_y)
    info_B = draw_top(ax2, apex=apex_B, angle_deg=tilt_deg, scale=1.6)

    # Angular momentum vectors at two times: decreasing magnitude due to torque
    L0_len = 1.25
    L1_len = 0.7
    draw_arrow(ax2, start=info_B["com"], direction=info_B["axis_dir"], length=L0_len,
               color='#2ca02c', lw=2.5, linestyle='--', label='L(t0)')
    draw_arrow(ax2, start=info_B["com"], direction=info_B["axis_dir"], length=L1_len,
               color='#2ca02c', lw=3.0, linestyle='solid', label='L(t1)')

    # Torque arrow (friction) opposite to spin axis
    tau_dir = -info_B["axis_dir"]
    tau_start = info_B["apex"] + 0.05 * tau_dir + np.array([0.0, 0.04])  # slight lift above ground
    draw_arrow(ax2, start=tau_start, direction=tau_dir, length=0.5, color='#d62728', lw=3, label='τ (friction)')

    # Explanatory text
    ax2.text(-2.2, 1.9, 'External torque ≠ 0 → angular momentum changes', fontsize=10, color='#333333')
    ax2.text(-2.2, 1.7, 'Here: friction opposes spin → |L| decreases', fontsize=10, color='#333333')

    # Save figure
    out_name = 'conservation_of_angular_momentum_spinning_top.png'
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {out_name}')


if __name__ == '__main__':
    main()

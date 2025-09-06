import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Arc, Polygon


def draw_spinning_top(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-0.2, 2.8)
    ax.axis('off')

    # Draw a stylized spinning top body (side view)
    y = np.array([0.05, 0.25, 0.55, 0.9, 1.3, 1.7, 2.05, 2.25])
    halfw = np.array([0.02, 0.18, 0.7, 1.05, 1.2, 0.75, 0.35, 0.1])
    left = np.column_stack((-halfw, y))
    right = np.column_stack((halfw[::-1], y[::-1]))
    body_pts = np.vstack([left, right])
    top_body = Polygon(body_pts, closed=True, facecolor="#cfe8ff", edgecolor="#377eb8", lw=2, zorder=1)
    ax.add_patch(top_body)

    # Axis of rotation (dashed) and omega vector
    ax.plot([0, 0], [0, 2.6], ls='--', color='gray', lw=1.5, zorder=0)
    ax.text(0.08, 1.3, 'axis of\nrotation', color='gray', va='center')
    omega_arrow = FancyArrowPatch((0, 2.3), (0, 2.65), arrowstyle='-|>', mutation_scale=18,
                                  color='#1f77b4', lw=2, zorder=3)
    ax.add_patch(omega_arrow)
    ax.text(0.06, 2.62, '\u03C9', color='#1f77b4', fontsize=12, va='center')

    # Elliptical ring (perspective) indicating a circular path of a point on the top
    cy = 1.2
    a = 1.12  # horizontal semi-axis (visual radius)
    b = 0.30  # vertical semi-axis (perspective flattening)
    ring = Ellipse((0, cy), width=2*a, height=2*b, angle=0, facecolor='none', edgecolor='0.55', lw=1.5, zorder=2)
    ax.add_patch(ring)

    # Point on the ring at parameter t=0 (rightmost point)
    t = 0.0
    px = 0 + a * np.cos(t)
    py = cy + b * np.sin(t)
    ax.plot(px, py, 'o', color='black', ms=4, zorder=4)

    # Tangential velocity vector v (upward here for CCW rotation viewed from above)
    # Tangent direction on ellipse: (-a*sin t, b*cos t); at t=0 -> (0, b)
    v_len = 0.6
    vx, vy = 0.0, v_len
    v_arrow = FancyArrowPatch((px, py), (px + vx, py + vy), arrowstyle='-|>', mutation_scale=14,
                              color='#2ca02c', lw=2.0, zorder=4)
    ax.add_patch(v_arrow)
    ax.text(px + 0.1, py + v_len * 0.5, 'tangential\nvelocity v', color='#2ca02c', fontsize=9, va='center')

    # Centripetal acceleration a_c (toward axis)
    ac_len = 0.7
    ac_arrow = FancyArrowPatch((px, py), (px - ac_len, py), arrowstyle='-|>', mutation_scale=14,
                               color='#d62728', lw=2.0, zorder=4)
    ax.add_patch(ac_arrow)
    ax.text(px - ac_len * 0.9, py + 0.08, 'centripetal\nacceleration a_c', color='#d62728', fontsize=9, ha='right')

    # Curved arrow around axis to indicate rotation direction
    arc_start, arc_end = -30, 300
    rot_arc = Arc((0, cy), width=2*a, height=2*b, angle=0, theta1=arc_start, theta2=arc_end,
                  color='#1f77b4', lw=1.5, zorder=2)
    ax.add_patch(rot_arc)
    # Add a small arrowhead at arc end
    ang1 = np.deg2rad(arc_end - 6)
    ang2 = np.deg2rad(arc_end)
    x1, y1 = 0 + a * np.cos(ang1), cy + b * np.sin(ang1)
    x2, y2 = 0 + a * np.cos(ang2), cy + b * np.sin(ang2)
    rot_head = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                               color='#1f77b4', lw=1.5, zorder=3)
    ax.add_patch(rot_head)

    ax.set_title('Spinning top: side view', pad=8)


def draw_top_view(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis('off')

    # Concentric circular paths (top view)
    radii = [0.45, 0.9, 1.3]
    for r in radii:
        c = Circle((0, 0), r, facecolor='none', edgecolor='0.7', lw=1.5, zorder=1)
        ax.add_patch(c)

    # Angle reference and delta-theta arc
    theta = np.deg2rad(30)
    theta2 = np.deg2rad(55)
    ax.plot([0, 1.2*np.cos(theta)], [0, 1.2*np.sin(theta)], color='0.5', lw=1.0)
    dtheta_arc = Arc((0, 0), width=0.8, height=0.8, angle=0, theta1=np.rad2deg(theta), theta2=np.rad2deg(theta2),
                     color='0.4', lw=1.2)
    ax.add_patch(dtheta_arc)
    mid = (theta + theta2) / 2
    ax.text(0.5*np.cos(mid), 0.5*np.sin(mid) + 0.06, '\u0394\u03B8', fontsize=10, color='0.4', ha='center')

    # Points on circles and tangential velocity vectors (v = ω r)
    max_r = max(radii)
    base_len = 0.55
    for r in radii:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.plot(x, y, 'o', color='black', ms=4, zorder=3)
        # Tangent unit vector (perpendicular to radius, CCW): (-sinθ, cosθ)
        t_hat = np.array([-np.sin(theta), np.cos(theta)])
        v_len = base_len * (r / max_r)
        end = np.array([x, y]) + v_len * t_hat
        arr = FancyArrowPatch((x, y), end, arrowstyle='-|>', mutation_scale=14, color='#2ca02c', lw=2, zorder=3)
        ax.add_patch(arr)

    # Labels to emphasize concepts
    ax.text(0, 1.45, 'Top view: all points share same angular speed \u03C9', ha='center', fontsize=11)
    ax.text(0, -1.45, 'Linear speed v = \u03C9 r increases with radius', ha='center', fontsize=11, color='#2ca02c')

    # Small symbol for ω pointing out of the page (dot-in-circle convention)
    ax.add_patch(Circle((0, 0), 0.06, facecolor='#1f77b4', edgecolor='none', zorder=4))
    ax.add_patch(Circle((0, 0), 0.12, facecolor='none', edgecolor='#1f77b4', lw=2, zorder=4))
    ax.text(0.18, 0.0, '\u03C9 (same for all points)', va='center', color='#1f77b4')

    ax.set_title('Circular motion of points: top view', pad=8)


if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    draw_spinning_top(axes[0])
    draw_top_view(axes[1])

    fig.suptitle('Rotational Motion Illustrated with a Spinning Top', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = 'rotational_motion_spinning_top.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {out_name}')

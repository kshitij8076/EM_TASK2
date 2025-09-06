import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Arc, FancyArrowPatch


def rotate_points(pts, angle_rad):
    R = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                  [np.sin(angle_rad),  np.cos(angle_rad)]])
    return pts @ R.T


def curved_arc_with_arrow(ax, center, radius, theta1_deg, theta2_deg, color='k', lw=1.8, arrow_size=8, zorder=5):
    # Draw arc
    arc = Arc(center, width=2*radius, height=2*radius, angle=0,
              theta1=theta1_deg, theta2=theta2_deg, color=color, lw=lw, zorder=zorder)
    ax.add_patch(arc)
    # Arrowhead at end
    theta2 = np.deg2rad(theta2_deg)
    theta_mid = np.deg2rad(theta2_deg - 4)
    end = np.array(center) + radius * np.array([np.cos(theta2), np.sin(theta2)])
    start = np.array(center) + radius * np.array([np.cos(theta_mid), np.sin(theta_mid)])
    arr = FancyArrowPatch(posA=start, posB=end, arrowstyle='-|>', mutation_scale=arrow_size,
                          lw=lw, color=color, zorder=zorder)
    ax.add_patch(arr)


def arrow_along_vector(ax, start, direction, length, color='k', lw=2.2, label=None):
    d = np.array(direction)
    d = d / (np.linalg.norm(d) + 1e-9)
    end = np.array(start) + d * length
    arr = FancyArrowPatch(posA=start, posB=end, arrowstyle='-|>', mutation_scale=10, lw=lw, color=color)
    ax.add_patch(arr)
    if label:
        ax.text(*(end + 0.1*d), label, color=color, fontsize=10, ha='left', va='center')


def draw_spinning_top(ax):
    ax.set_aspect('equal')
    ax.axis('off')

    # Ground/reference line
    ax.plot([-3, 3], [0, 0], color='#888888', lw=2)
    ax.text(-2.9, 0.12, 'contact point', fontsize=9, color='#666666', va='bottom')

    # Axis tilt
    theta_deg = 28
    theta = np.deg2rad(theta_deg)
    u = np.array([np.sin(theta), np.cos(theta)])  # axis unit vector from tip
    tip = np.array([0.0, 0.0])

    # Draw the top body as a rotated silhouette
    # Define in local coords where axis is +y
    body_pts_local = np.array([
        [0.0, 0.0],
        [0.55, 1.1],
        [0.28, 2.1],
        [0.10, 2.7],
        [-0.10, 2.7],
        [-0.28, 2.1],
        [-0.55, 1.1],
        [0.0, 0.0]
    ])
    # Rotate local y (90 deg) to align with u (angle alpha)
    alpha = np.arctan2(u[1], u[0])
    rot = alpha - np.deg2rad(90)
    body_pts = rotate_points(body_pts_local, rot) + tip
    ax.add_patch(Polygon(body_pts, closed=True, facecolor='#c9d9ff', edgecolor='#425aab', lw=2, zorder=2))

    # Tip
    ax.add_patch(Circle(tip, 0.04, color='#425aab', zorder=3))

    # Axis line
    L_axis = 3.0
    ax.plot([tip[0], tip[0] + L_axis*u[0]], [tip[1], tip[1] + L_axis*u[1]], color='#1f3a93', lw=2.2)
    ax.text(*(tip + u*L_axis*0.62 + np.array([0.12, 0.08])), 'spin axis', color='#1f3a93', fontsize=10)

    # Center of mass (approx along axis)
    t_com = 1.5
    com = tip + u * t_com
    ax.add_patch(Circle(com, 0.05, facecolor='#2c3e50', edgecolor='white', lw=0.7, zorder=5))
    ax.text(*(com + np.array([0.12, 0.0])), 'center of mass', fontsize=9, color='#2c3e50', va='center')

    # Gravity at COM
    g_len = 0.9
    arrow_along_vector(ax, com, [0, -1], g_len, color='#c0392b', lw=2.2, label='mg')

    # Angular momentum vector L along the axis
    L_len = 1.0
    arrow_along_vector(ax, com + u*0.1, u, L_len, color='#27ae60', lw=2.4, label='L')

    # Spin arrow (local rotation around axis) - suggestive arc near COM
    curved_arc_with_arrow(ax, center=com, radius=0.38, theta1_deg=210, theta2_deg=520,
                          color='#34495e', lw=2, arrow_size=9, zorder=6)
    ax.text(*(com + np.array([0.6, 0.15])), 'spin', fontsize=9, color='#34495e')

    # Vertical reference axis and precession cue at the tip
    ax.plot([0, 0], [0, 3.0], ls='--', color='#777777', lw=1.5)
    ax.text(0.06, 2.8, 'vertical', color='#777777', fontsize=9, ha='left')

    # Precession path cue: a small circle around the tip (axis sweeps around vertical)
    precess_r = 0.8
    circ = Circle(tip, precess_r, fill=False, ls='--', lw=1.4, edgecolor='#7f8c8d')
    ax.add_patch(circ)
    curved_arc_with_arrow(ax, center=tip, radius=precess_r, theta1_deg=-30, theta2_deg=210,
                          color='#7f8c8d', lw=2, arrow_size=9, zorder=4)
    ax.text(precess_r*0.9, -0.18, 'precession (axis sweeps around vertical)', fontsize=9, color='#7f8c8d', ha='right')

    # Torque note
    ax.annotate('gravity + offset => torque\ncauses slow precession',
                xy=com + np.array([0, -g_len]), xytext=(-1.9, 1.6),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='#444444'),
                fontsize=9.5, color='#444444', ha='left', va='center')

    # Sub-title for main panel
    ax.text(-2.8, 3.05, 'Spinning top: angular momentum provides stability', fontsize=12, weight='bold')

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.7, 3.3)


def draw_gyroscope(ax):
    ax.set_aspect('equal')
    ax.axis('off')
    # Rings
    ax.add_patch(Circle((0, 0), 1.0, fill=False, lw=2, ec='#34495e'))
    ax.add_patch(Circle((0, 0), 0.7, fill=False, lw=2, ec='#34495e'))
    ax.add_patch(Circle((0, 0), 0.35, fill=True, fc='#95a5a6', ec='#2c3e50', lw=1.2))
    # Axle
    ax.plot([-1.2, 1.2], [0, 0], color='#2c3e50', lw=2)
    # Spin cue
    curved_arc_with_arrow(ax, center=(0, 0), radius=0.55, theta1_deg=200, theta2_deg=520,
                          color='#2c3e50', lw=2, arrow_size=8)
    # Angular momentum along axle
    arrow_along_vector(ax, [0.0, 0.0], [1, 0], 0.9, color='#27ae60', lw=2, label='L')
    ax.text(0, -1.25, 'Gyroscope\n(stability & navigation)', ha='center', fontsize=9)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)


def draw_bicycle_wheel(ax):
    ax.set_aspect('equal')
    ax.axis('off')
    # Rim and hub
    ax.add_patch(Circle((0, 0), 1.05, fill=False, lw=2, ec='#2c3e50'))
    ax.add_patch(Circle((0, 0), 0.12, fill=True, fc='#7f8c8d', ec='#2c3e50', lw=1))
    # Spokes
    for ang in np.linspace(0, 2*np.pi, 12, endpoint=False):
        ax.plot([0, 1.05*np.cos(ang)], [0, 1.05*np.sin(ang)], color='#7f8c8d', lw=1)
    # Axle and L
    ax.plot([-1.3, 1.3], [0, 0], color='#555555', lw=1.5)
    arrow_along_vector(ax, [0, 0], [1, 0], 0.95, color='#27ae60', lw=2, label='L')
    # Spin arrow on rim
    curved_arc_with_arrow(ax, center=(0, 0), radius=0.9, theta1_deg=120, theta2_deg=450,
                          color='#34495e', lw=2, arrow_size=8)
    ax.text(0, -1.35, 'Bicycle wheel\n(gyroscopic stability)', ha='center', fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)


def draw_turbine(ax):
    ax.set_aspect('equal')
    ax.axis('off')
    # Hub
    ax.add_patch(Circle((0, 0), 0.15, fc='#7f8c8d', ec='#2c3e50', lw=1))
    # Blades
    for k in range(4):
        ang = k * np.pi/2
        r1, r2 = 0.22, 1.05
        w = 0.35
        pts = np.array([
            [r1*np.cos(ang - w), r1*np.sin(ang - w)],
            [r2*np.cos(ang),     r2*np.sin(ang)],
            [r1*np.cos(ang + w), r1*np.sin(ang + w)]
        ])
        ax.add_patch(Polygon(pts, closed=True, fc='#c9d9ff', ec='#2c3e50', lw=1.2))
    # Housing ring
    ax.add_patch(Circle((0, 0), 1.15, fill=False, lw=2, ec='#34495e'))
    # Rotation cue
    curved_arc_with_arrow(ax, center=(0, 0), radius=0.95, theta1_deg=210, theta2_deg=520,
                          color='#34495e', lw=2, arrow_size=8)
    # L along axis (out of page suggested by small side arrow)
    arrow_along_vector(ax, [0, 0], [1, 0], 0.9, color='#27ae60', lw=2, label='L')
    ax.text(0, -1.45, 'Turbine/rotor\n(energy via rotation)', ha='center', fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)


def draw_planet(ax):
    ax.set_aspect('equal')
    ax.axis('off')
    # Planet sphere
    ax.add_patch(Circle((0, 0), 1.1, fc='#dff5ff', ec='#2c3e50', lw=2))
    # Equator band
    ax.plot([-1.1, 1.1], [0, 0], color='#3498db', lw=2)
    # Tilted axis
    tilt = np.deg2rad(23.5)
    a = np.array([np.sin(tilt), np.cos(tilt)])
    ax.plot([-1.3*a[0], 1.3*a[0]], [-1.3*a[1], 1.3*a[1]], color='#2c3e50', lw=2)
    # Rotation cue
    curved_arc_with_arrow(ax, center=(0, 0), radius=0.95, theta1_deg=120, theta2_deg=450,
                          color='#34495e', lw=2, arrow_size=8)
    # L along spin axis
    arrow_along_vector(ax, [0, 0], a, 0.95, color='#27ae60', lw=2, label='L')
    ax.text(0, -1.5, 'Planet\n(day/night from rotation)', ha='center', fontsize=9)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.8, 1.6)


def main():
    fig = plt.figure(figsize=(12, 7), dpi=150)
    gs = fig.add_gridspec(2, 3, width_ratios=[2.2, 1, 1], height_ratios=[1, 1], wspace=0.35, hspace=0.4)

    ax_main = fig.add_subplot(gs[:, 0])
    ax_r1 = fig.add_subplot(gs[0, 1])
    ax_r2 = fig.add_subplot(gs[0, 2])
    ax_r3 = fig.add_subplot(gs[1, 1])
    ax_r4 = fig.add_subplot(gs[1, 2])

    draw_spinning_top(ax_main)
    draw_gyroscope(ax_r1)
    draw_bicycle_wheel(ax_r2)
    draw_turbine(ax_r3)
    draw_planet(ax_r4)

    fig.suptitle('Real-World Applications of Rotation: From Spinning Top to Technology and Nature', fontsize=14, weight='bold', y=0.97)
    fig.text(0.36, 0.02, 'Mastering rotational physics (angular momentum, stability, energy) scales from toys to gyroscopes, vehicles, turbines, and planets.',
             ha='center', fontsize=10, color='#333333')

    out_png = 'rotation_real_world_spinning_top.png'
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    # Optionally also save a PDF for vector clarity
    fig.savefig('rotation_real_world_spinning_top.pdf', bbox_inches='tight')
    print(f'Saved figure to {out_png} and rotation_real_world_spinning_top.pdf')


if __name__ == '__main__':
    main()

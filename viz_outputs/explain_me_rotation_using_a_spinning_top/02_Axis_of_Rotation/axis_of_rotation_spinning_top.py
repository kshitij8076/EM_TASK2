import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse


def rotate_points(pts, angle_deg, origin=(0.0, 0.0)):
    pts = np.asarray(pts, dtype=float)
    ox, oy = origin
    angle = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    shifted = pts - np.array([ox, oy])
    rot = shifted @ R.T + np.array([ox, oy])
    return rot


def add_tangent_arrows_on_ellipse(ax, center, r, k, angle_deg, t_degs, color='tab:blue'):
    cx, cy = center
    angle = np.deg2rad(angle_deg)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    for t_deg in t_degs:
        t = np.deg2rad(t_deg)
        p_local = np.array([r*np.cos(t), k*r*np.sin(t)])
        v_local = np.array([-r*np.sin(t), k*r*np.cos(t)])
        p_rot = R @ p_local
        v_rot = R @ v_local
        p = np.array([cx, cy]) + p_rot
        v = v_rot
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        v_hat = v / norm
        L = max(0.25, 0.35*r)
        start = p - v_hat * L
        end = p
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.6, color=color))


def draw_top_scene(ax, axis_angle_deg, title, label_axis_text, show_ground=True):
    tip = np.array([0.0, 0.0])
    H = 6.0      # body height
    R = 2.4      # base radius
    hh = 1.8     # handle height
    rh = 0.5     # handle radius
    flatten_k = 0.33  # perspective flattening for circular paths

    body_color = '#d79f59'
    edge_color = '#8b5a2b'
    handle_color = '#b07a3b'

    # Rotation to align the top's axis from +y to the desired axis angle
    delta = axis_angle_deg - 90.0

    # Main body (cone-like) and handle (rect), defined upright then rotated
    tri = np.array([[0.0, 0.0], [-R, H], [R, H]])
    tri_rot = rotate_points(tri, delta, origin=(0.0, 0.0)) + tip

    rect = np.array([[-rh, H], [rh, H], [rh, H+hh], [-rh, H+hh]])
    rect_rot = rotate_points(rect, delta, origin=(0.0, 0.0)) + tip

    ax.add_patch(Polygon(tri_rot, closed=True, facecolor=body_color, edgecolor=edge_color, linewidth=2))
    ax.add_patch(Polygon(rect_rot, closed=True, facecolor=handle_color, edgecolor=edge_color, linewidth=1.8))

    # Axis line (dashed)
    u = np.array([np.cos(np.deg2rad(axis_angle_deg)), np.sin(np.deg2rad(axis_angle_deg))])
    start = tip - u * 0.4
    end = tip + u * (H + hh + 0.7)
    ax.plot([start[0], end[0]], [start[1], end[1]], linestyle=(0, (6, 6)), color='dimgray', linewidth=2.0)

    # Circular paths (rings) perpendicular to the axis
    levels = [0.28, 0.55, 0.85]
    ring_color = 'tab:blue'
    for frac in levels:
        y_along = frac * H
        center = tip + u * y_along
        r = R * (y_along / H)
        e = Ellipse(xy=center, width=2*r, height=2*r*flatten_k, angle=delta,
                    edgecolor=ring_color, facecolor='none', linewidth=1.8, alpha=0.95)
        ax.add_patch(e)

    # Tangent arrows indicating rotation direction on the middle ring
    y_mid = levels[1] * H
    center_mid = tip + u * y_mid
    r_mid = R * (y_mid / H)
    add_tangent_arrows_on_ellipse(ax, center_mid, r_mid, flatten_k, delta,
                                  t_degs=[35, 155, 275], color=ring_color)

    # Title and axis label
    ax.set_title(title, fontsize=14, pad=8)

    axis_label_point = tip + u * (H + hh * 0.6)
    offset_dir = np.array([-u[1], u[0]])  # perpendicular direction to axis
    text_pos = axis_label_point + offset_dir * 1.6
    ax.annotate(label_axis_text, xy=axis_label_point, xytext=text_pos,
                ha='center', va='center', fontsize=12, color='black',
                arrowprops=dict(arrowstyle='->', lw=1.4, color='black'))

    # Explanatory note
    ax.text(-4.2, -0.7, 'Points on the top move in circles around the axis.', fontsize=10)

    # Ground reference line
    if show_ground:
        ax.plot([-5, 5], [0, 0], color='lightgray', linewidth=2)

    # Layout
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-1.2, H + hh + 1.2)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')


def main():
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left: Upright axis (stable)
    draw_top_scene(axs[0], axis_angle_deg=90.0, title='Upright axis (stable)',
                   label_axis_text='Axis of rotation', show_ground=True)

    # Right: Tilted axis (wobble)
    draw_top_scene(axs[1], axis_angle_deg=70.0, title='Tilted axis (wobble)',
                   label_axis_text='Tilted axis', show_ground=True)

    fig.suptitle('Axis of Rotation: Spinning Top', fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    outname = 'axis_of_rotation_spinning_top.png'
    fig.savefig(outname, dpi=200, bbox_inches='tight')
    print(f'Saved figure to {outname}')


if __name__ == '__main__':
    main()

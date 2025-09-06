import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Arc


def draw_out_of_page(ax, center=(0, 0), R=0.12, color='purple', lw=2):
    # Circle with a central dot to indicate vector out of the page (\u2299)
    cx, cy = center
    outer = Circle((cx, cy), R, fill=False, lw=lw, color=color)
    inner = Circle((cx, cy), R * 0.35, fill=True, color=color, lw=0)
    ax.add_patch(outer)
    ax.add_patch(inner)


def draw_panel(ax, R=1.0, r_frac=0.35, phi_deg=45.0, title="", r_label="r"):
    ax.set_aspect('equal')
    ax.axis('off')
    lim = 1.55 * R
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Disk/top viewed from above
    disk = Circle((0, 0), R, fill=False, lw=2.0, color='0.35')
    ax.add_patch(disk)

    # Spin axis marker and label
    draw_out_of_page(ax, center=(0, 0), R=0.10 * R, color='purple', lw=2)
    ax.text(0.07 * R, -0.07 * R, 'Spin axis', color='purple', fontsize=10, ha='left', va='top')

    # Geometry
    phi = np.deg2rad(phi_deg)
    r_mag = r_frac * R
    r_hat = np.array([np.cos(phi), np.sin(phi)])
    t_hat = np.array([-np.sin(phi), np.cos(phi)])  # CCW tangential direction

    r_vec = r_mag * r_hat

    # Draw r (lever arm)
    ax.annotate('', xy=r_vec, xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='royalblue', lw=2.5))
    # Label r
    mid_r = 0.6 * r_vec
    ax.text(mid_r[0], mid_r[1], r_label, color='royalblue', fontsize=11,
            ha='left', va='bottom')

    # Draw force (same magnitude in both panels)
    F_len = 0.50 * R
    F_vec = F_len * t_hat
    F_start = r_vec
    F_end = r_vec + F_vec
    ax.annotate('', xy=F_end, xytext=F_start,
                arrowprops=dict(arrowstyle='->', color='crimson', lw=2.5))
    # Label F
    f_label_pos = F_start + 0.55 * F_vec
    ax.text(f_label_pos[0], f_label_pos[1], 'F', color='crimson', fontsize=11,
            ha='left', va='bottom')

    # Right-angle arc between r and F (theta = 90\u00b0 here)
    # Draw at the application point
    arc_radius = 0.28 * R
    arc = Arc((r_vec[0], r_vec[1]), width=2 * arc_radius, height=2 * arc_radius,
              angle=0, theta1=phi_deg, theta2=phi_deg + 90, color='0.25', lw=1.5)
    ax.add_patch(arc)
    mid_theta = np.deg2rad(phi_deg + 45)
    theta_text_pos = r_vec + (arc_radius + 0.08 * R) * np.array([np.cos(mid_theta), np.sin(mid_theta)])
    ax.text(theta_text_pos[0], theta_text_pos[1], '\u03b8 = 90\u00b0', fontsize=9, color='0.25', ha='center', va='center')

    # Title
    ax.set_title(title, fontsize=12, pad=8)

    # Torque magnitude cue (text)
    # \u03c4 = r F sin\u03b8
    # Also add a small note about direction (out of page)
    ax.text(0, -1.32 * R, '\u03c4 = r F sin\u03b8 (direction: out of page)',
            fontsize=9, color='purple', ha='center', va='center')


def main():
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))

    draw_panel(axs[0], R=1.0, r_frac=0.35, phi_deg=45,
               title='Force near axis (small r)', r_label='r (small)')
    axs[0].text(0, 1.32, 'Small torque', color='black', fontsize=11, ha='center')

    draw_panel(axs[1], R=1.0, r_frac=0.92, phi_deg=45,
               title='Force at rim (large r)', r_label='r (large)')
    axs[1].text(0, 1.32, 'Large torque', color='black', fontsize=11, ha='center')

    fig.suptitle('Torque (\u03c4) on a Spinning Top: \u03c4 = r F sin\u03b8', fontsize=16, y=0.98)

    # Bottom explanatory caption
    caption = (
        'In both panels, the same force F is applied tangentially (\u03b8 = 90\u00b0).\n'
        'Increasing the lever arm r increases the torque (\u03c4), making it easier to spin the top up to high speeds.\n'
        'Greater \u03c4 produces greater angular acceleration \u03b1 for a given moment of inertia I (\u03c4 = I\u03b1).'
    )
    fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=10)

    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    outfile = 'torque_spinning_top.png'
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {outfile}')


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def draw_arrow(ax, start, end, color='k', lw=2, label=None, label_offset=(0.1, 0.1), head=8):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, mutation_scale=head))
    if label is not None:
        mid = ((start[0] + end[0]) / 2 + label_offset[0], (start[1] + end[1]) / 2 + label_offset[1])
        ax.text(mid[0], mid[1], label, color=color, fontsize=10, ha='left', va='bottom')


def curved_arrow(ax, pstart, pend, color='k', lw=2, rad=-0.3, head=10):
    ax.annotate('', xy=pend, xytext=pstart,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                mutation_scale=head, connectionstyle=f'arc3,rad={rad}'))


def panel_A(ax):
    ax.set_aspect('equal')
    ax.set_xlim(-2.2, 3.2)
    ax.set_ylim(-0.6, 4.2)
    ax.axis('off')

    # Ground
    ax.plot([-2.2, 3.2], [0, 0], color='0.7', lw=2)
    ax.text(-2.1, 0.1, 'support point', fontsize=9, color='0.3', ha='left', va='bottom')

    # Vertical reference
    ax.plot([0, 0], [0, 3.8], linestyle='--', color='0.6', lw=1.5)
    ax.text(0.1, 3.75, 'vertical', fontsize=9, color='0.4', ha='left')

    # Top axis and angular momentum L
    phi = np.deg2rad(20)
    axis_len = 3.2
    tip = (axis_len * np.sin(phi), axis_len * np.cos(phi))
    ax.plot([0, tip[0]], [0, tip[1]], color='navy', lw=2.5)
    draw_arrow(ax, (0, 0), tip, color='navy', lw=2.5, label='L (angular momentum)', label_offset=(0.1, 0.1))

    # Center of mass along the axis
    d = 1.6
    com = (d * np.sin(phi), d * np.cos(phi))
    ax.scatter([com[0]], [com[1]], s=20, color='k')
    ax.text(com[0] + 0.08, com[1] + 0.05, 'COM', fontsize=9)

    # Weight mg at COM
    mg_len = 1.0
    draw_arrow(ax, com, (com[0], com[1] - mg_len), color='crimson', lw=2, label='mg', label_offset=(0.07, -0.05))

    # Label lever arm r (projection from pivot to COM)
    draw_arrow(ax, (0, 0), com, color='0.4', lw=1.6, label='r', label_offset=(0.0, 0.0), head=6)

    # Precession indication around vertical (caused by torque tau = r x mg)
    cy = 2.4
    r_arc = 1.2
    theta1 = np.deg2rad(65)
    theta2 = np.deg2rad(-20)
    pstart = (0 + r_arc * np.cos(theta1), cy + r_arc * np.sin(theta1))
    pend = (0 + r_arc * np.cos(theta2), cy + r_arc * np.sin(theta2))
    curved_arrow(ax, pstart, pend, color='darkgreen', lw=2.2, rad=-0.35)
    ax.text(0.1, cy + 1.1, 'precession (dL/dt = τ)', color='darkgreen', fontsize=10, ha='left')

    # Clarify torque direction (out of page)
    ax.text(2.0, 1.0, 'τ = r × mg\n(out of page)', fontsize=9, color='0.25', ha='center', va='center')

    # Small top body schematic (disk) near COM to suggest mass distribution
    circ = Circle((com[0], com[1]), 0.12, fill=False, ec='0.5', lw=1)
    ax.add_patch(circ)

    ax.set_title('A. Torque from gravity makes L precess around the vertical', fontsize=12)


def panel_cone(ax, tilt_deg, color, title, L_label, speed_label, thick=3):
    ax.set_aspect('equal')
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-0.6, 4.2)
    ax.axis('off')

    # Vertical reference
    ax.plot([0, 0], [0, 3.8], linestyle='--', color='0.6', lw=1.5)

    # Cone locus (circle traced by tip of the top's axis)
    h = 2.6
    r = h * np.tan(np.deg2rad(tilt_deg))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(r * np.cos(th), h + r * np.sin(th), color='0.75', lw=1.2, linestyle='--')
    ax.text(0.05, h + r + 0.1, 'precession path', fontsize=9, color='0.5', ha='left')

    # Draw one instantaneous L vector along the cone
    psi = np.deg2rad(35)
    tip = (r * np.cos(psi), h + r * np.sin(psi))
    draw_arrow(ax, (0, 0), tip, color=color, lw=thick, label=L_label, label_offset=(0.1, 0.1))

    # A few faint positions to suggest the sweep
    for ang in [psi + d for d in (-0.8, -0.4, 0.4, 0.8)]:
        pt = (r * np.cos(ang), h + r * np.sin(ang))
        ax.plot([0, pt[0]], [0, pt[1]], color=color, alpha=0.25, lw=thick)

    # Curved arrow to indicate precession speed
    theta_a = np.deg2rad(10)
    theta_b = np.deg2rad(-90)
    pstart = (r * np.cos(theta_a), h + r * np.sin(theta_a))
    pend = (r * np.cos(theta_b), h + r * np.sin(theta_b))
    rad = -0.25 if 'slow' in speed_label.lower() else -0.55
    curved_arrow(ax, pstart, pend, color='darkgreen', lw=2.2, rad=rad)
    ax.text(-1.9, h - r - 0.2, f'{speed_label} (Ω = τ/L)', fontsize=10, color='darkgreen', ha='left')

    # Title for this panel
    ax.set_title(title, fontsize=12)


def panel_time(ax):
    ax.set_xlim(0, 10)
    ax.set_xlabel('Time (arbitrary units)')

    # Angular momentum decay (normalized)
    t = np.linspace(0, 10, 400)
    L0 = 1.0
    tau_decay = 6.0
    L = L0 * np.exp(-t / tau_decay)

    # Tilt growth as L falls (illustrative)
    tilt_min = 8.0  # degrees
    tilt_max = 40.0 # degrees
    tilt_tau = 4.0
    tilt = tilt_min + (tilt_max - tilt_min) * (1 - np.exp(-t / tilt_tau))

    ax2 = ax.twinx()

    ax.plot(t, L, color='navy', lw=2.2, label='L (normalized)')
    ax.set_ylabel('Angular momentum (normalized L)', color='navy')
    ax.tick_params(axis='y', labelcolor='navy')

    ax2.plot(t, tilt, color='crimson', lw=2.2, label='Tilt angle')
    ax2.set_ylabel('Tilt angle (degrees)', color='crimson')
    ax2.tick_params(axis='y', labelcolor='crimson')

    # Annotate interpretation
    ax.text(0.4, 0.92, 'Friction reduces L', transform=ax.transAxes, color='navy', fontsize=10, ha='left')
    ax2.text(0.4, 0.2, 'Tilt grows → wobble → fall', transform=ax2.transAxes, color='crimson', fontsize=10, ha='left')

    ax.grid(alpha=0.3)
    ax.set_title('D. As L decreases with friction, stability fades and tilt increases', fontsize=12)


def main():
    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    plt.subplots_adjust(wspace=0.2, hspace=0.4)

    # Panel A: Torque and precession principle
    panel_A(axs[0, 0])

    # Panel B: High spin stability
    panel_cone(
        axs[0, 1],
        tilt_deg=12,
        color='tab:blue',
        title='B. High spin (large L): strong gyroscopic stability',
        L_label='L large (fast/heavy)',
        speed_label='slow precession'
    )

    # Panel C: Low spin instability
    panel_cone(
        axs[1, 0],
        tilt_deg=28,
        color='tab:red',
        title='C. Low spin (small L): weaker stability, faster wobble',
        L_label='L small (slow/light)',
        speed_label='fast precession'
    )

    # Panel D: Time evolution under friction
    panel_time(axs[1, 1])

    fig.suptitle('Gyroscopic Stability of a Spinning Top: Why fast-spinning tops stay upright', fontsize=16, y=0.98)

    outname = 'gyroscopic_stability_spinning_top.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {outname}')


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch


def draw_spinning_top(ax):
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.2, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # Body of the top (stylized polygon)
    body_pts = np.array([
        [-0.95, 1.5],
        [0.95, 1.5],
        [0.32, 0.55],
        [0.0, 0.0],
        [-0.32, 0.55]
    ])
    body = Polygon(body_pts, closed=True, facecolor='#D1E5F0', edgecolor='#2C7FB8', linewidth=1.8)
    ax.add_patch(body)

    # Stem and knob
    ax.plot([0, 0], [1.5, 2.1], color='#2C7FB8', lw=3)
    knob = Circle((0, 2.22), 0.08, facecolor='#2C7FB8', edgecolor='none')
    ax.add_patch(knob)

    # Center axis (dashed)
    ax.plot([0, 0], [-0.05, 2.35], color='gray', lw=1.2, ls='--', alpha=0.7)

    # Curved arrow indicating spin (ω)
    spin_arrow = FancyArrowPatch(
        posA=(0.75, 1.05), posB=(-0.75, 1.05),
        connectionstyle='arc3,rad=0.55',
        arrowstyle='-|>',
        mutation_scale=16,
        lw=2.0, color='#2C7FB8')
    ax.add_patch(spin_arrow)
    ax.text(0.0, 1.55, 'ω (angular velocity)', color='#2C7FB8', ha='center', va='bottom', fontsize=9)

    # Friction at the tip → opposing torque (slows spin)
    friction_arrow = FancyArrowPatch(
        posA=(-0.55, 0.18), posB=(0.55, 0.18),
        connectionstyle='arc3,rad=-0.55',
        arrowstyle='-|>',
        mutation_scale=16,
        lw=2.0, color='#D62728')
    ax.add_patch(friction_arrow)
    ax.text(0.0, 0.34, 'Friction at tip → opposing torque', color='#D62728', ha='center', va='bottom', fontsize=9)

    # Note about torque changing ω ⇒ α
    ax.text(0.0, 2.35, 'External torque changes ω → α', color='black', ha='center', va='bottom', fontsize=10)


def main():
    # Time and piecewise angular velocity ω(t)
    alpha1 = 8.0    # rad/s^2 (spin-up by hand)
    alpha2 = -2.5   # rad/s^2 (slowing due to friction)

    t1 = np.linspace(0, 2, 120)
    w1 = alpha1 * t1

    t2 = np.linspace(2, 5, 120)
    w2 = np.full_like(t2, w1[-1])

    t3 = np.linspace(5, 10, 240)
    w3 = w2[-1] + alpha2 * (t3 - 5)

    t = np.concatenate([t1, t2, t3])
    w = np.concatenate([w1, w2, w3])

    # Figure layout
    fig = plt.figure(figsize=(11.5, 6.5))
    ax = fig.add_axes([0.08, 0.12, 0.63, 0.80])  # main plot
    ax_top = fig.add_axes([0.75, 0.10, 0.22, 0.80])  # schematic of top

    # Plot ω(t)
    ax.plot(t, w, color='#1f77b4', lw=2.5)

    # Visual regions for α > 0 and α < 0
    ymax = max(w) * 1.1
    ax.set_ylim(-0.5, ymax)
    ax.set_xlim(0, 10)

    ax.axvspan(0, 2, color='#2ca02c', alpha=0.10)
    ax.axvspan(5, 10, color='#d62728', alpha=0.10)

    # Labels and annotations
    ax.set_title('Angular Acceleration (α) for a Spinning Top', fontsize=14)
    ax.set_xlabel('Time t (s)', fontsize=12)
    ax.set_ylabel('Angular velocity ω (rad/s)', fontsize=12)

    # α > 0 region annotation
    ax.annotate('α > 0 (speeding up)\n(hand torque)',
                xy=(1.2, alpha1 * 1.2), xytext=(0.6, ymax * 0.78),
                arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.8),
                ha='center', va='center', fontsize=11, color='#2ca02c')

    # α ≈ 0 plateau
    ax.text(3.5, w2[-1] + 0.5, 'α ≈ 0 (nearly constant spin)',
            ha='center', va='bottom', fontsize=10, color='gray')

    # α < 0 region annotation
    w_at_7 = w2[-1] + alpha2 * (7 - 5)
    ax.annotate('α < 0 (slowing down)\n(frictional torque)',
                xy=(7.0, w_at_7), xytext=(6.8, ymax * 0.35),
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.8),
                ha='right', va='center', fontsize=11, color='#d62728')

    # Small reminder: α = dω/dt
    ax.text(9.85, ymax * 0.95, 'α = dω/dt', ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))

    # Grid for readability
    ax.grid(True, ls='--', lw=0.6, alpha=0.5)

    # Draw the spinning top schematic
    draw_spinning_top(ax_top)
    ax_top.set_title('Spinning top: torques change ω → α', fontsize=10, pad=6)

    # Save figure
    out_name = 'angular_acceleration_spinning_top.png'
    plt.savefig(out_name, dpi=200, bbox_inches='tight')
    print(f'Saved figure to {out_name}')


if __name__ == '__main__':
    main()

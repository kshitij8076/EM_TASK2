import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle


def main():
    # Parameters
    v = 20.0  # m/s
    theta_deg = 30.0  # degrees
    theta = np.deg2rad(theta_deg)
    g = 9.81  # m/s^2

    # Components
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)

    # Trajectory (no air resistance)
    T = 2 * vy / g
    t = np.linspace(0, T, 300)
    x = vx * t
    y = vy * t - 0.5 * g * t**2

    R = vx * T
    H = vy**2 / (2 * g)

    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 6))

    # Ground line
    ax.axhline(0, color='0.7', lw=1)

    # Plot trajectory
    ax.plot(x, y, color='0.2', lw=2, ls='--')

    # Initial velocity vector and its resolution triangle
    # Arrow for resultant velocity v
    head_w = 0.6
    head_l = 1.2
    ax.arrow(0, 0, vx, vy, length_includes_head=True, head_width=head_w, head_length=head_l,
             fc='seagreen', ec='seagreen', lw=2)

    # Arrow for horizontal component vx
    ax.arrow(0, 0, vx, 0, length_includes_head=True, head_width=head_w*0.85, head_length=head_l*0.85,
             fc='royalblue', ec='royalblue', lw=2)

    # Arrow for vertical component vy (drawn from tip of vx)
    ax.arrow(vx, 0, 0, vy, length_includes_head=True, head_width=head_w*0.85, head_length=head_l*0.85,
             fc='darkorange', ec='darkorange', lw=2)

    # Angle arc at origin
    arc_r = 4.5
    arc = Arc((0, 0), 2*arc_r, 2*arc_r, angle=0, theta1=0, theta2=theta_deg, color='0.4', lw=1.5)
    ax.add_patch(arc)
    ax.text(arc_r*0.85*np.cos(theta/2), arc_r*0.85*np.sin(theta/2), r"$\theta=30^\circ$", color='0.3',
            ha='center', va='center')

    # Labels for vectors
    ax.text(vx*0.55, vy*0.55, r"$\vec{v}$ = 20 m/s", color='seagreen', fontsize=12,
            ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='seagreen', alpha=0.8))
    ax.text(vx*0.5, -0.9, r"$v_x = v\cos\theta = %.1f$ m/s" % vx, color='royalblue', fontsize=11,
            ha='center', va='top')
    ax.text(vx + 0.6, vy*0.5, r"$v_y = v\sin\theta = %.1f$ m/s" % vy, color='darkorange', fontsize=11,
            ha='left', va='center')

    # Helpful notes linking vectors to motion
    note = (
        "Resolve initial velocity into components:\n"
        "- Horizontal: constant $v_x$\n"
        "- Vertical: changes under gravity $g$"
    )
    ax.text(R*0.58, H*1.6 + 2, note, fontsize=11, color='0.2', ha='left', va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.5', alpha=0.9))

    # Minimal cricket context: ball at launch, simple stumps in the distance
    ball = Circle((0, 0), 0.35, color='crimson', ec='black', zorder=5)
    ax.add_patch(ball)
    # Stumps near the approximate range
    stump_x = R * 0.95
    stump_w = 0.25
    stump_h = 1.0
    for i in range(3):
        ax.add_patch(Rectangle((stump_x + i*(stump_w*1.1), 0), stump_w, stump_h, color='#C09050', ec='saddlebrown'))

    # Axes labels and title
    ax.set_xlabel('Horizontal distance x (m)')
    ax.set_ylabel('Vertical height y (m)')
    ax.set_title('Cricket Projectile: Vector Representation and Resolution')

    # Limits and aspect
    x_max = max(R*1.1, vx*1.15 + 2)
    y_max = max(H*2.2, vy*1.15 + 2)
    ax.set_xlim(-1.5, x_max)
    ax.set_ylim(-1.5, y_max)
    ax.set_aspect('equal', adjustable='box')

    # Clean look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.15)

    # Save
    out_name = 'cricket_vector_resolution.png'
    plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    print(f'Saved figure to {out_name}')


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def main():
    # Parameters for the football kick
    g = 9.81  # m/s^2
    v0 = 22.0  # initial speed (m/s)
    angle_deg = 40.0  # launch angle in degrees
    theta = np.deg2rad(angle_deg)

    v0x = v0 * np.cos(theta)
    v0y = v0 * np.sin(theta)
    T = 2 * v0y / g  # time of flight

    # Time, position arrays
    t = np.linspace(0, T, 400)
    x = v0x * t
    y = v0y * t - 0.5 * g * t**2

    xmax = float(x.max())
    ymax = float(y.max())

    # Figure layout: left panel (trajectory), right column (x vs t, y vs t)
    fig = plt.figure(figsize=(12, 5.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.3], wspace=0.25)

    ax_traj = fig.add_subplot(gs[0, 0])
    right = gs[0, 1].subgridspec(2, 1, hspace=0.35)
    ax_xt = fig.add_subplot(right[0, 0])
    ax_yt = fig.add_subplot(right[1, 0])

    # ---- Left: Trajectory with component arrows ----
    ax_traj.plot(x, y, color='tab:orange', lw=3, label='Trajectory')

    # Ground line
    ax_traj.axhline(0, color='saddlebrown', lw=2, zorder=0)

    # Place a football marker (ellipse) somewhere along the path
    t_ball = 0.35 * T
    x_b = v0x * t_ball
    y_b = v0y * t_ball - 0.5 * g * t_ball**2
    ball = Ellipse((x_b, y_b), width=1.2, height=0.7, angle=0, facecolor='#8B4513', edgecolor='black', lw=1.0, zorder=5)
    ax_traj.add_patch(ball)

    # Velocity component arrows at three points
    sample_times = [0.2 * T, 0.5 * T, 0.8 * T]
    Lh = 0.08 * xmax  # horizontal arrow length (constant)
    for ti in sample_times:
        xi = v0x * ti
        yi = v0y * ti - 0.5 * g * ti**2
        vy_i = v0y - g * ti
        # Horizontal velocity (constant)
        ax_traj.annotate('', xy=(xi + Lh, yi), xytext=(xi, yi),
                         arrowprops=dict(arrowstyle='->', color='tab:blue', lw=2.2))
        # Vertical velocity (changes)
        # Scale vertical arrow by instantaneous |vy| relative to v0
        Lv = 0.25 * ymax * (vy_i / v0)
        ax_traj.annotate('', xy=(xi, yi + Lv), xytext=(xi, yi),
                         arrowprops=dict(arrowstyle='->', color='tab:green', lw=2.2))

    # Acceleration due to gravity (downward)
    xg = 0.88 * xmax
    yg = 1.10 * ymax
    Lg = 0.22 * ymax
    ax_traj.annotate(r"$g$", xy=(xg, yg - Lg), xytext=(xg, yg),
                     arrowprops=dict(arrowstyle='->', lw=2, color='k'),
                     ha='center', va='bottom', fontsize=12)

    # Text: Kinematic equations and parameters
    ax_traj.text(0.02, 0.96, 'Kinematic equations:', transform=ax_traj.transAxes,
                 fontsize=12, weight='bold')
    ax_traj.text(0.02, 0.90, r"$x(t) = v_{0x}\, t$", transform=ax_traj.transAxes, fontsize=12)
    ax_traj.text(0.02, 0.84, r"$y(t) = v_{0y}\, t - \tfrac{1}{2} g t^2$", transform=ax_traj.transAxes, fontsize=12)
    ax_traj.text(0.02, 0.73,
                 f"v0 = {v0:.0f} m/s, angle = {angle_deg:.0f}°, g = {g:.2f} m/s²",
                 transform=ax_traj.transAxes, fontsize=10,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7', alpha=0.9))

    ax_traj.text(0.58, 0.32, r"Horizontal velocity $v_x$ is constant",
                 color='tab:blue', transform=ax_traj.transAxes, fontsize=10)
    ax_traj.text(0.53, 0.78, r"Vertical velocity $v_y$ changes with $g$",
                 color='tab:green', transform=ax_traj.transAxes, fontsize=10)

    ax_traj.set_title('Projectile motion of a football — components and equations', fontsize=13)
    ax_traj.set_xlabel('Horizontal distance x (m)')
    ax_traj.set_ylabel('Height y (m)')
    ax_traj.set_xlim(0, xmax * 1.05)
    ax_traj.set_ylim(-0.5, ymax * 1.25)
    ax_traj.grid(True, alpha=0.25)

    # ---- Right top: x(t) linear ----
    ax_xt.plot(t, x, color='tab:blue', lw=2.5)
    ax_xt.set_title('Horizontal motion: constant velocity', fontsize=12)
    ax_xt.set_ylabel('x (m)')
    ax_xt.set_xlim(0, T)
    ax_xt.set_ylim(0, xmax * 1.05)
    ax_xt.text(0.05, 0.85, r"$x = v_{0x} t$", transform=ax_xt.transAxes, fontsize=12, color='tab:blue')
    ax_xt.grid(True, alpha=0.3)

    # ---- Right bottom: y(t) quadratic ----
    y_t = v0y * t - 0.5 * g * t**2
    ax_yt.plot(t, y_t, color='tab:green', lw=2.5)
    ax_yt.set_title('Vertical motion: constant acceleration', fontsize=12)
    ax_yt.set_xlabel('time t (s)')
    ax_yt.set_ylabel('y (m)')
    ax_yt.set_xlim(0, T)
    ax_yt.set_ylim(min(-0.5, y_t.min()*1.05), ymax * 1.05)
    ax_yt.text(0.05, 0.85, r"$y = v_{0y} t - \tfrac{1}{2} g t^2$", transform=ax_yt.transAxes, fontsize=12, color='tab:green')

    # Mark the apex on y(t)
    t_apex = v0y / g
    y_apex = v0y * t_apex - 0.5 * g * t_apex**2
    ax_yt.scatter([t_apex], [y_apex], color='black', zorder=5)
    ax_yt.annotate('apex', xy=(t_apex, y_apex), xytext=(t_apex + 0.08 * T, y_apex + 0.12 * ymax),
                   arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=9)
    ax_yt.grid(True, alpha=0.3)

    # Save figure
    out_name = 'projectile_motion_football.png'
    fig.savefig(out_name, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure to {out_name}')


if __name__ == '__main__':
    main()

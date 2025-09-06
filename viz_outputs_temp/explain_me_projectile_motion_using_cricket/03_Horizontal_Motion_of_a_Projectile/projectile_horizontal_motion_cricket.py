import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def trajectory(v0, theta_deg, g=9.81, num=400):
    theta = np.deg2rad(theta_deg)
    T = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, T, num)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    return t, x, y, T


def add_wickets(ax, x0=0.0, ground_y=0.0):
    # Simple wickets at the origin (3 stumps)
    stump_width = 0.06
    stump_height = 0.9
    gap = 0.04
    total_width = 3 * stump_width + 2 * gap
    left = x0 - total_width / 2
    colors = {'edgecolor': 'dimgray', 'facecolor': 'lightgray'}
    for i in range(3):
        x_pos = left + i * (stump_width + gap)
        ax.add_patch(Rectangle((x_pos, ground_y), stump_width, stump_height, **colors, zorder=5))


def main():
    # Parameters
    v0 = 36.0  # m/s (a solid cricket shot)
    g = 9.81
    angles = [20, 35, 50]
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Compute ranges to set axis limits
    ranges = []
    peaks = []
    data = {}
    for ang, col in zip(angles, colors):
        t, x, y, T = trajectory(v0, ang, g)
        data[ang] = (t, x, y, T)
        R = x[-1]
        ranges.append(R)
        peaks.append(y.max())

    x_max = max(ranges) * 1.05
    y_max = max(peaks) * 1.2

    # Figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle('Horizontal Motion of a Projectile (Cricket Shot)', fontsize=14, y=0.98)

    # Left panel: 2D trajectories
    for ang, col in zip(angles, colors):
        t, x, y, T = data[ang]
        ax1.plot(x, y, color=col, lw=2, label=f"Launch angle: {ang}\N{DEGREE SIGN}")

    # Ground line
    ax1.axhline(0, color='k', lw=1)
    add_wickets(ax1, x0=0.0, ground_y=0.0)

    # Equal-time markers for the middle angle to show uniform horizontal spacing
    mid_angle = 35
    t_mid, x_mid, y_mid, T_mid = data[mid_angle]
    n_marks = 6
    t_marks = np.linspace(0, T_mid, n_marks + 2)[1:-1]  # skip t=0 and t=T
    x_marks = v0 * np.cos(np.deg2rad(mid_angle)) * t_marks
    y_marks = v0 * np.sin(np.deg2rad(mid_angle)) * t_marks - 0.5 * g * t_marks**2
    ax1.plot(x_marks, y_marks, 'o', color='tab:orange', ms=4, zorder=6, label='_nolegend_')
    for xm, ym in zip(x_marks, y_marks):
        ax1.vlines(xm, 0, ym, colors='tab:orange', linestyles='dotted', alpha=0.6, lw=1)

    ax1.text(
        x_marks[len(x_marks)//2],
        max(0.5, min(y_marks) + (max(y_marks) - min(y_marks)) * 0.15),
        'Equal time intervals → equal horizontal spacing',
        color='tab:orange', fontsize=9, ha='center', va='bottom',
    )

    ax1.set_xlim(0, x_max)
    ax1.set_ylim(0, y_max)
    ax1.set_xlabel('Horizontal distance x (m)')
    ax1.set_ylabel('Height y (m)')
    ax1.legend(frameon=False, loc='upper right')
    ax1.grid(True, ls='--', lw=0.5, alpha=0.3)
    for spine in ['top', 'right']:
        ax1.spines[spine].set_visible(False)

    # Right panel: x vs t is linear with slope = v_x = v0 cos(theta)
    for ang, col in zip(angles, colors):
        t, x, y, T = data[ang]
        ax2.plot(t, x, color=col, lw=2, label=f"{ang}\N{DEGREE SIGN}")

    # Annotations about slope
    sample_ang = 20
    t_s, x_s, _, T_s = data[sample_ang]
    # Pick two points to illustrate slope visually
    i1, i2 = int(0.2 * len(t_s)), int(0.5 * len(t_s))
    ax2.plot([t_s[i1], t_s[i2]], [x_s[i1], x_s[i2]], color='tab:blue', lw=3, alpha=0.3)
    vx_sample = v0 * np.cos(np.deg2rad(sample_ang))
    ax2.text(
        0.55 * T_s,
        0.2 * x_s[-1],
        r'slope = $v_x$ = $v_0 \cos\\theta$ (constant)',
        fontsize=10, color='k', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none')
    )

    ax2.set_xlabel('Time t (s)')
    ax2.set_ylabel('Horizontal distance x (m)')
    ax2.grid(True, ls='--', lw=0.5, alpha=0.3)
    ax2.legend(title='Angle', frameon=False, loc='upper left')
    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)

    # Tidy layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_name = 'horizontal_motion_projectile_cricket.png'
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {out_name}')


if __name__ == '__main__':
    main()

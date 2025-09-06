import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Parameters for projectile motion
theta_deg = 45  # MODIFIED: explicit 45-degree launch angle
theta = np.deg2rad(theta_deg)
g = 9.81  # m/s^2

# Choose v0 so the range is around 120 m at 45 degrees
R_target = 120.0
v0 = np.sqrt(R_target * g)

# Time of flight and trajectory
T = 2 * v0 * np.sin(theta) / g
t = np.linspace(0, T, 400)
x = v0 * np.cos(theta) * t
y = v0 * np.sin(theta) * t - 0.5 * g * t**2

# Apex (highest point)
t_apex = v0 * np.sin(theta) / g
x_apex = v0 * np.cos(theta) * t_apex
y_apex = v0 * np.sin(theta) * t_apex - 0.5 * g * t_apex**2

# Figure
plt.rcParams.update({
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False
})
fig, ax = plt.subplots(figsize=(14, 8))

# Ground line
ax.axhline(0, color='darkgreen', lw=2)

# Trajectory
ax.plot(x, y, color='crimson', lw=4, solid_capstyle='round')

# Apex marker and label
ax.scatter([x_apex], [y_apex], s=50, color='black', zorder=5)
ax.text(x_apex, y_apex + 1.0, 'Highest point (apex)', ha='center', va='bottom')

# Gravity arrow annotation (downward)
ax.annotate('Gravity g pulls downward',
            xy=(105, 14), xytext=(105, 24),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2),
            ha='center', va='center', color='gray', rotation=90)

# Initial velocity arrow at 45 degrees
arrow_len = 18
x_arr = arrow_len * np.cos(theta)
y_arr = arrow_len * np.sin(theta)
ax.annotate('', xy=(x_arr, y_arr), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', lw=3, color='#4C8CCE'))
ax.text(x_arr * 0.9, y_arr * 0.9 + 0.5,
        r'Initial velocity $v_0$ at angle $\theta = 45^\circ$',
        color='#4C8CCE')

# Angle arc showing 45 degrees
arc_radius = 10
arc = Arc((0, 0), 2*arc_radius, 2*arc_radius, angle=0, theta1=0, theta2=theta_deg,
          color='gray', lw=2)
ax.add_patch(arc)
ax.text(arc_radius * 0.75 * np.cos(np.deg2rad(22.5)),
        arc_radius * 0.75 * np.sin(np.deg2rad(22.5)),
        r'$\theta=45^\circ$', color='gray', ha='center', va='center')

# Explanatory textbox (definition)
explain = (
    'After the bat strike, the ball is in projectile motion:\n'
    'it moves only under the influence of gravity (no more push),\n'
    'so its path is a curved, parabolic arc.'
)
ax.text(8, 5.2, explain, ha='left', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

# Axes limits and labels
ax.set_xlim(0, 130)
ax.set_ylim(0, max(32, y_apex + 6))
ax.set_xlabel('Horizontal distance (m)')
ax.set_ylabel('Height (m)')
ax.set_title('Projectile Motion in Cricket: Ball path after the hit (gravity only)\n(Launch angle explicitly set to 45°)')

ax.grid(alpha=0.2)

plt.tight_layout()
fig.savefig('projectile_motion_cricket_slide1_modified.png', dpi=200)
print('Saved figure: projectile_motion_cricket_slide1_modified.png')

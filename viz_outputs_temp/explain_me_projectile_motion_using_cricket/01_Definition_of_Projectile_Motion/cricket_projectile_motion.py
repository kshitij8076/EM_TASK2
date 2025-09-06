import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch, Arc
from matplotlib import transforms

# Projectile motion parameters (gravity only, no air resistance)
g = 9.81  # m/s^2
v0 = 35.0  # initial speed in m/s
angle_deg = 35.0  # launch angle in degrees
x0, y0 = 0.0, 1.0  # launch point near bat (about waist height)

# Compute trajectory
theta = np.deg2rad(angle_deg)
# Flight time from y(t) = y0 + v0*sin(theta)*t - 0.5*g*t^2 = 0
# t_flight = (v0*sin(theta) + sqrt((v0*sin(theta))^2 + 2*g*y0)) / g
t_flight = (v0 * np.sin(theta) + np.sqrt((v0 * np.sin(theta)) ** 2 + 2 * g * y0)) / g

T = np.linspace(0, t_flight, 600)
X = x0 + v0 * np.cos(theta) * T
Y = y0 + v0 * np.sin(theta) * T - 0.5 * g * T ** 2

# Apex
t_apex = v0 * np.sin(theta) / g
x_apex = x0 + v0 * np.cos(theta) * t_apex
y_apex = y0 + v0 * np.sin(theta) * t_apex - 0.5 * g * t_apex ** 2

# Figure setup
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

fig, ax = plt.subplots()
ax.plot(X, Y, color='crimson', lw=3, label='Cricket ball trajectory (gravity only)')

# Ground line
x_end = X[-1]
y_max = max(Y)
ax.axhline(0, color='forestgreen', lw=4, zorder=0)

# Bat representation (simple rotated rectangle near the launch point)
bat_len = 0.7
bat_thick = 0.12
bat_color = '#c89b6d'
bat_edge = '#8b5a2b'
# Place the bat so its end is at (x0, y0), then rotate about (x0, y0)
bat_rect = Rectangle((x0 - bat_len, y0 - bat_thick / 2.0), bat_len, bat_thick,
                     facecolor=bat_color, edgecolor=bat_edge, linewidth=1.5)
bat_transform = transforms.Affine2D().rotate_deg_around(x0, y0, angle_deg - 15) + ax.transData
bat_rect.set_transform(bat_transform)
ax.add_patch(bat_rect)

# Initial ball at contact
ax.add_patch(Circle((x0, y0), 0.08, color='crimson', ec='black', zorder=5))

# Sample ball positions along the path to illustrate motion
for t_frac in [0.25, 0.5, 0.75]:
    ti = t_flight * t_frac
    xi = x0 + v0 * np.cos(theta) * ti
    yi = y0 + v0 * np.sin(theta) * ti - 0.5 * g * ti ** 2
    ax.add_patch(Circle((xi, yi), 0.06, color='crimson', alpha=0.8, zorder=4))

# Mark apex
ax.plot(x_apex, y_apex, marker='o', color='black')
ax.text(x_apex, y_apex + 0.5, 'Highest point (apex)', ha='center', va='bottom', fontsize=10)

# Initial velocity arrow
L = max(6.0, 0.18 * x_end)
vx_arrow_end = (x0 + L * np.cos(theta), y0 + L * np.sin(theta))
arrow_v0 = FancyArrowPatch((x0, y0), vx_arrow_end, arrowstyle='-|>', mutation_scale=15,
                           lw=2, color='steelblue')
ax.add_patch(arrow_v0)
ax.text(x0 + 0.55 * L * np.cos(theta), y0 + 0.55 * L * np.sin(theta) + 0.6,
        r'Initial velocity $v_0$ at angle $\theta$', color='steelblue', fontsize=10)

# Angle arc at launch
R = max(2.5, 0.1 * x_end)
arc = Arc((x0, y0), 2 * R, 2 * R, angle=0, theta1=0, theta2=angle_deg,
          color='gray', lw=1.5)
ax.add_patch(arc)
ax.text(x0 + 0.9 * R * np.cos(np.deg2rad(angle_deg / 2.0)),
        y0 + 0.9 * R * np.sin(np.deg2rad(angle_deg / 2.0)), r'$\theta$', color='gray', fontsize=12)

# Gravity arrow on the right
xg = 0.88 * x_end
yg_top = y_max * 0.9 if y_max > 3 else 3
gravity_arrow = FancyArrowPatch((xg, yg_top), (xg, yg_top - max(3, 0.25 * y_max)),
                                arrowstyle='-|>', mutation_scale=15, lw=2, color='dimgray')
ax.add_patch(gravity_arrow)
ax.text(xg + 0.4, yg_top - 1.2, r'Gravity $g$ pulls downward', rotation=270,
        va='center', ha='left', color='dimgray', fontsize=10)

# Explanatory text box tying to definition
exp_text = (
    'After the bat strike, the ball is in projectile motion:\n'
    'it moves only under the influence of gravity (no more push),\n'
    'so its path is a curved, parabolic arc.'
)
ax.text(0.02 * x_end, 0.1 * (y_max + 1) + 0.2, exp_text,
        fontsize=10, va='bottom', ha='left', color='black',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='lightgray'))

# Labels and styling
ax.set_title('Projectile Motion in Cricket: Ball path after the hit (gravity only)')
ax.set_xlabel('Horizontal distance (m)')
ax.set_ylabel('Height (m)')
ax.grid(True, linestyle='--', alpha=0.4)

# Limits with some margins
ax.set_xlim(-1.0, x_end * 1.05)
ax.set_ylim(0, max(y_max * 1.25, y0 + 2))

# Make layout tight and save figure
plt.tight_layout()
outfile = 'cricket_projectile_motion.png'
plt.savefig(outfile, dpi=200)
print(f'Saved figure to {outfile}')

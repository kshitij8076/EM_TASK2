import numpy as np
import matplotlib.pyplot as plt

# Parameters (SI units)
g = 9.81  # m/s^2 (gravity)
v0 = 25.0  # initial speed (m/s)
angle_deg = 40.0  # launch angle (degrees)
angle = np.deg2rad(angle_deg)

# Time of flight (neglecting air resistance)
t_flight = 2 * v0 * np.sin(angle) / g

# Trajectory samples for smooth curve
t = np.linspace(0, t_flight, 400)
x = v0 * np.cos(angle) * t
y = v0 * np.sin(angle) * t - 0.5 * g * t**2

# Equal-time markers (to show constant horizontal spacing)
num_marks = 9
t_marks = np.linspace(0, t_flight, num_marks, endpoint=False)
xm = v0 * np.cos(angle) * t_marks
ym = v0 * np.sin(angle) * t_marks - 0.5 * g * t_marks**2

# Useful values for annotations
x_range = x[-1]
y_max = y.max()

# Figure setup
plt.rcParams.update({
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False
})
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# Ground line
a x.axhline(0, color='forestgreen', lw=3, alpha=0.6)

# Parabolic trajectory
ax.plot(x, y, color='tab:blue', lw=2.5, label='Trajectory (parabola)')

# Equal-time positions of the football
ax.plot(xm, ym, 'o', ms=6, color='saddlebrown', mec='black', mew=0.5,
        label='Football positions (equal time intervals)')

# Vertical guide lines at equal time steps (equal horizontal spacing)
for xi, yi in zip(xm, ym):
    ax.vlines(xi, 0, yi, colors='0.75', linestyles=':', linewidth=1, zorder=0)

# Initial velocity vector and components (scaled for visibility)
scale = 0.2 * x_range / max(v0, 1e-6)
vx0 = v0 * np.cos(angle)
vy0 = v0 * np.sin(angle)
Dx = vx0 * scale
Dy = vy0 * scale

# Component arrows
ax.arrow(0, 0, Dx, 0, length_includes_head=True, head_width=0.6, head_length=1.0,
         fc='tab:orange', ec='tab:orange')
ax.text(Dx*0.5, -0.04*y_max, 'vx (constant)', color='tab:orange', ha='center', va='top')
ax.arrow(Dx, 0, 0, Dy, length_includes_head=True, head_width=0.6, head_length=1.0,
         fc='tab:cyan', ec='tab:cyan')
ax.text(Dx + 0.02*x_range, Dy*0.5, 'Initial vy', color='tab:cyan', va='center')

# Resultant initial velocity arrow
ax.arrow(0, 0, Dx, Dy, length_includes_head=True, head_width=0.7, head_length=1.2,
         fc='crimson', ec='crimson', alpha=0.8)
ax.text(Dx*0.6, Dy*0.6 + 0.03*y_max, 'Initial velocity v0', color='crimson', ha='center')

# Gravity arrow (constant downward acceleration)
ax.annotate('', xy=(0.87*x_range, 0.82*y_max), xytext=(0.87*x_range, 0.97*y_max),
            arrowprops=dict(arrowstyle='-|>', lw=2, color='black'))
ax.text(0.87*x_range + 0.01*x_range, 0.895*y_max, 'g downward', color='black', va='center')

# Apex annotation (vy = 0)
i_apex = np.argmax(y)
x_apex, y_apex = x[i_apex], y[i_apex]
ax.scatter([x_apex], [y_apex], s=30, color='purple', zorder=5)
ax.annotate('Apex (vy = 0)', xy=(x_apex, y_apex), xytext=(x_apex + 0.12*x_range, y_apex + 0.12*y_max),
            arrowprops=dict(arrowstyle='->', color='purple'), color='purple')

# Annotation for equal horizontal spacing (constant vx)
if len(xm) >= 4:
    target_x = xm[3]
else:
    target_x = xm[-1]
ax.annotate('Equal horizontal spacing\nfor equal time steps → vx constant',
            xy=(target_x, 0), xycoords='data',
            xytext=(target_x, 0.22*y_max), textcoords='data',
            arrowprops=dict(arrowstyle='->', color='0.3'),
            ha='center', va='bottom', color='0.2')

# Small cues for changing vertical velocity (early up, later down)
# Early point (vy > 0)
t1 = 0.2 * t_flight
x1 = v0 * np.cos(angle) * t1
y1 = v0 * np.sin(angle) * t1 - 0.5 * g * t1**2
vy1 = v0 * np.sin(angle) - g * t1
arrow_len1 = 0.1 * y_max * (abs(vy1) / max(v0, 1e-6))
ax.arrow(x1, y1 - 0.02*y_max, 0, arrow_len1, length_includes_head=True,
         head_width=0.5, head_length=0.8, fc='0.25', ec='0.25')
ax.text(x1 + 0.01*x_range, y1 + arrow_len1*0.5, 'vy upward', color='0.25', va='center')

# Late point (vy < 0)
t2 = 0.8 * t_flight
x2 = v0 * np.cos(angle) * t2
y2 = v0 * np.sin(angle) * t2 - 0.5 * g * t2**2
vy2 = v0 * np.sin(angle) - g * t2
arrow_len2 = 0.1 * y_max * (abs(vy2) / max(v0, 1e-6))
ax.arrow(x2, y2 + 0.02*y_max, 0, -arrow_len2, length_includes_head=True,
         head_width=0.5, head_length=0.8, fc='0.25', ec='0.25')
ax.text(x2 + 0.01*x_range, y2 - arrow_len2*0.5, 'vy downward', color='0.25', va='center')

# Labels, limits, legend
ax.set_title('Projectile Motion of a Kicked Football (Neglecting Air Resistance)')
ax.set_xlabel('Horizontal distance x (m)')
ax.set_ylabel('Vertical height y (m)')
ax.set_xlim(-0.02*x_range, 1.05*x_range)
ax.set_ylim(-0.06*y_max, 1.12*y_max)
ax.legend(loc='upper left', frameon=False)

# Subtle note
ax.text(0.01*x_range, 1.07*y_max, 'Key properties: parabolic path, constant vx, downward acceleration g',
        fontsize=9, color='0.25')

plt.tight_layout()

# Save figure
outfile = 'projectile_motion_football.png'
fig.savefig(outfile, dpi=300, bbox_inches='tight')
print(f'Saved figure to {outfile}')

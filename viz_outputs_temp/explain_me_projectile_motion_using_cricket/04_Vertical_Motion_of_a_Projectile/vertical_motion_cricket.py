import numpy as np
import matplotlib.pyplot as plt

# Vertical Motion of a Cricket Ball (Projectile)
# This script plots the vertical position y(t) and vertical velocity v_y(t)
# for a lofted cricket shot, highlighting key ideas: apex (v_y=0),
# constant downward acceleration (-g), and total hang time.

# Parameters (feel free to tweak these)
g = 9.81  # m/s^2
v0 = 35.0  # initial speed (m/s), a solid lofted hit
launch_angle_deg = 50.0  # launch angle (degrees)

# Derived vertical component
theta = np.deg2rad(launch_angle_deg)
v0y = v0 * np.sin(theta)

# Key times
t_apex = v0y / g
T = 2 * t_apex  # total hang time (assuming same launch/landing height)

# Time array
t = np.linspace(0, T, 400)

# Vertical motion equations
y = v0y * t - 0.5 * g * t**2
vy = v0y - g * t

# Derived values for annotations
y_max = v0y**2 / (2 * g)

# Sample ball positions over time (to visualize the ball rising/falling)
sample_times = np.linspace(0, T, 7)
y_samples = v0y * sample_times - 0.5 * g * sample_times**2

# Matplotlib styling for clarity
plt.rcParams.update({
    'figure.figsize': (9, 8),
    'axes.grid': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.titleweight': 'bold',
    'font.size': 11
})

fig, axes = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.15})

# --- Top: Vertical position vs time ---
ax = axes[0]
ax.plot(t, y, color='#0B6E4F', linewidth=2.5, label='Height y(t)')
ax.scatter(sample_times, y_samples, color='#B3001B', s=50, zorder=3, label='Ball samples')

# Ground line (y=0)
ax.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.8)

# Apex markers
ax.axvline(t_apex, color='#1F4E79', linestyle=':', linewidth=1.8)
ax.axhline(y_max, color='#1F4E79', linestyle=':', linewidth=1.2)
ax.plot([t_apex], [y_max], marker='o', color='#1F4E79', markersize=6)
ax.text(t_apex, y_max + 0.03 * y_max if y_max > 0 else 0.6, 'Apex (v_y = 0)', color='#1F4E79', ha='center', va='bottom')

# Hang time annotation
ax.annotate(
    'Hang time = 2·v0y/g',
    xy=(T, 0), xycoords='data', xytext=(T*0.62, y_max*0.35), textcoords='data',
    arrowprops=dict(arrowstyle='<->', color='black', shrinkA=0, shrinkB=0),
    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='black', alpha=0.85)
)

# Phase labels
ax.text(t_apex*0.35, y_max*0.55, 'Rising (v_y > 0)', color='#0B6E4F', bbox=dict(fc='white', ec='#0B6E4F', alpha=0.7))
ax.text(t_apex*1.25, y_max*0.55, 'Falling (v_y < 0)', color='#0B6E4F', bbox=dict(fc='white', ec='#0B6E4F', alpha=0.7))

# Axes labels and title
ax.set_ylabel('Height y (m)')
ax.set_title('Vertical Motion of a Cricket Ball')

# Add a concise context box (cricket framing)
context_text = (
    f"Lofted shot: v0 = {v0:.0f} m/s, angle = {launch_angle_deg:.0f}°\n"
    f"Vertical component: v0y = v0·sin(θ) = {v0y:.2f} m/s\n"
    f"Max height: y_max = v0y²/(2g) = {y_max:.2f} m"
)
ax.text(0.02, 0.98, context_text, transform=ax.transAxes, ha='left', va='top',
        bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.9))

# --- Bottom: Vertical velocity vs time ---
ax2 = axes[1]
ax2.plot(t, vy, color='#B85C00', linewidth=2.5, label='Vertical velocity v_y(t)')
ax2.axhline(0, color='gray', linewidth=1.2, linestyle='--', alpha=0.8)

# Mark apex where vy=0
ax2.axvline(t_apex, color='#1F4E79', linestyle=':', linewidth=1.8)
ax2.plot([t_apex], [0], marker='o', color='#1F4E79', markersize=6)
ax2.text(t_apex, max(vy)*0.08, 'v_y crosses 0 at apex', color='#1F4E79', ha='center', va='bottom')

# Acceleration annotation: slope = -g
# Draw a tangent-like guide and annotate slope
xg0 = T*0.15
yg0 = v0y - g * xg0
xg1 = xg0 + 0.6
yg1 = yg0 - g * 0.6
ax2.plot([xg0, xg1], [yg0, yg1], color='#444444', linewidth=2, alpha=0.8)
ax2.annotate('-g (constant acceleration)', xy=(xg0 + 0.3, (yg0 + yg1)/2), xytext=(xg0 + 0.9, (yg0 + yg1)/2 + 0.6),
             arrowprops=dict(arrowstyle='->', color='#444444'), color='#444444', bbox=dict(fc='white', ec='#444444', alpha=0.9))

# Labels
ax2.set_xlabel('Time t (s)')
ax2.set_ylabel('Vertical velocity v_y (m/s)')

# Global annotations
fig.suptitle('Slide 4: Vertical Motion of a Projectile — Cricket Example', y=0.995)

# Neat ticks
axes[0].set_xlim(0, T)
# Keep y-limits tidy with a small margin
axes[0].set_ylim(0, y_max * 1.15 if y_max > 0 else 1)
axes[1].set_xlim(0, T)
axes[1].set_ylim(v0y * -1.15, v0y * 1.15)

# Save the figure
outfile = 'vertical_motion_cricket.png'
plt.savefig(outfile, dpi=200, bbox_inches='tight')
print(f'Saved figure to {outfile}')

# Optional display (safe to leave; non-interactive environments will just skip drawing)
# plt.show()

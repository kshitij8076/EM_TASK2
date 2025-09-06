import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Parameters for the cricket shot (no air resistance)
g = 9.81  # m/s^2 (gravity)
v0 = 35.0  # m/s, typical lofted cricket shot speed
angle_deg = 35.0  # launch angle in degrees
angle = np.deg2rad(angle_deg)

# Key projectile quantities
T = 2 * v0 * np.sin(angle) / g  # time of flight (s)
R = (v0**2) * np.sin(2 * angle) / g  # range (m)
H = (v0**2) * (np.sin(angle)**2) / (2 * g)  # max height (m)

# Trajectory points
num = 400
t = np.linspace(0, T, num)
x = v0 * np.cos(angle) * t
y = v0 * np.sin(angle) * t - 0.5 * g * t**2
# Ensure ground is y >= 0 for plotting (avoid tiny negative due to float error)
y = np.maximum(y, 0)

# Apex point
t_apex = v0 * np.sin(angle) / g
x_apex = v0 * np.cos(angle) * t_apex
y_apex = H

# Prepare angle sweep for insights on optimization
theta_deg = np.linspace(5, 80, 400)
theta = np.deg2rad(theta_deg)
R_theta = (v0**2) * np.sin(2 * theta) / g
T_theta = 2 * v0 * np.sin(theta) / g
H_theta = (v0**2) * (np.sin(theta)**2) / (2 * g)

# Figure layout
plt.rcParams.update({
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.3
})
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
gs = GridSpec(3, 4, figure=fig, width_ratios=[2.2, 2.2, 1.2, 1.2], height_ratios=[1, 1, 1])
ax_main = fig.add_subplot(gs[:, :2])
ax_r = fig.add_subplot(gs[0, 2:])
ax_t = fig.add_subplot(gs[1, 2:])
ax_h = fig.add_subplot(gs[2, 2:])

# Main trajectory plot (left)
ax_main.plot(x, y, lw=3, color='#1f77b4', label='Ball trajectory')
ax_main.set_title('Projectile Motion of a Cricket Shot (no air resistance)')
ax_main.set_xlabel('Horizontal distance x (m)')
ax_main.set_ylabel('Height y (m)')

# Ground line
ax_main.axhline(0, color='k', lw=1)

# Mark start (bat impact) and apex
ax_main.plot([0], [0], 'o', color='tab:green', ms=6)
ax_main.text(0.5, 0.5, 'Bat impact', color='tab:green')
ax_main.plot([x_apex], [y_apex], 'o', color='tab:orange', ms=6)
ax_main.annotate(f'Max height H = {H:.1f} m',
                 xy=(x_apex, y_apex), xytext=(x_apex*0.6, y_apex*1.15),
                 arrowprops=dict(arrowstyle='->', color='tab:orange'),
                 bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='tab:orange', alpha=0.8),
                 color='tab:orange')

# Range marker
ax_main.annotate('', xy=(R, 0), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='<->', color='tab:purple', lw=2))
ax_main.text(R*0.5, -0.05*max(1, H), f'Range R = {R:.1f} m', color='tab:purple', ha='center', va='top')

# Initial velocity vector and its components (scaled for display)
vec_len = 0.22 * R
vx_dir = np.cos(angle)
vy_dir = np.sin(angle)
# Resultant v0 arrow
ax_main.annotate('', xy=(vec_len*vx_dir, vec_len*vy_dir), xytext=(0, 0),
                 arrowprops=dict(arrowstyle='->', lw=2, color='tab:red'))
ax_main.text(vec_len*vx_dir*0.6, vec_len*vy_dir*0.65, f'v0 = {v0:.1f} m/s\n(launch {angle_deg:.0f}°)', color='tab:red')
# Component guides (dashed to axes)
ax_main.plot([vec_len*vx_dir, vec_len*vx_dir], [0, vec_len*vy_dir], ls='--', color='gray', lw=1)
ax_main.plot([0, vec_len*vx_dir], [vec_len*vy_dir, vec_len*vy_dir], ls='--', color='gray', lw=1)
ax_main.text(vec_len*vx_dir*0.5, -0.03*max(1, H), 'v0x = v0 cos(θ)', ha='center')
ax_main.text(-0.02*R, vec_len*vy_dir*0.5, 'v0y = v0 sin(θ)', va='center', rotation=90)

# Impact speed vector near landing (magnitude ~ v0 for equal heights)
impact_anchor_x = R * 0.98
impact_anchor_y = 0
ax_main.annotate('',
                 xy=(impact_anchor_x, impact_anchor_y),
                 xytext=(impact_anchor_x - vec_len*vx_dir, impact_anchor_y - vec_len*vy_dir),
                 arrowprops=dict(arrowstyle='->', lw=2, color='tab:red'))
ax_main.text(impact_anchor_x - vec_len*vx_dir*0.9, impact_anchor_y - vec_len*vy_dir*0.95,
             f'Impact speed ≈ {v0:.1f} m/s', color='tab:red', ha='right', va='top')

# Info box with computed values
info = f"Time of flight T = {T:.2f} s\nMax height H = {H:.1f} m\nRange R = {R:.1f} m"
ax_main.text(0.02*R, 0.95*max(H, 1), info,
             bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='gray', alpha=0.9), va='top')
ax_main.text(0.02*R, 0.05*max(H, 1), 'Velocity arrows scaled for display', color='gray')

# Limits and legend
ax_main.set_xlim(-0.02*R, R*1.08)
ax_main.set_ylim(0, max(H*1.3, 5))
ax_main.legend(loc='upper right')

# Right column: How angle affects outcomes
ax_r.plot(theta_deg, R_theta, color='tab:purple', lw=2)
ax_r.axvline(angle_deg, color='gray', ls='--', lw=1)
ax_r.plot([angle_deg], [R], 'o', color='tab:purple')
ax_r.set_title('Range vs Launch Angle')
ax_r.set_ylabel('Range (m)')
ax_r.set_xticklabels([])

ax_t.plot(theta_deg, T_theta, color='tab:blue', lw=2)
ax_t.axvline(angle_deg, color='gray', ls='--', lw=1)
ax_t.plot([angle_deg], [T], 'o', color='tab:blue')
ax_t.set_title('Flight Time vs Launch Angle')
ax_t.set_ylabel('Time (s)')
ax_t.set_xticklabels([])

ax_h.plot(theta_deg, H_theta, color='tab:orange', lw=2)
ax_h.axvline(angle_deg, color='gray', ls='--', lw=1)
ax_h.plot([angle_deg], [H], 'o', color='tab:orange')
ax_h.set_title('Max Height vs Launch Angle')
ax_h.set_ylabel('Height (m)')
ax_h.set_xlabel('Launch angle θ (degrees)')

for ax in [ax_r, ax_t, ax_h]:
    ax.set_xlim(5, 80)

# Super title tying to cricket analytics context
fig.suptitle('Application of Mathematics in Cricket Projectile Motion: Predicting range, flight time, and height', y=0.99, fontsize=12)

# Save figure
out_name = 'cricket_projectile_motion.png'
plt.savefig(out_name, dpi=200, bbox_inches='tight')
print(f'Saved figure to {out_name}')

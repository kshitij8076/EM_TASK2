import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # gravity (m/s^2)
v0 = 25.0  # initial speed (m/s)
angles_deg = [25, 40, 65]  # kick angles in degrees
colors = ['#1f77b4', '#2ca02c', '#d62728']

# Compute trajectories
trajectories = []
for ang, c in zip(angles_deg, colors):
    theta = np.radians(ang)
    T = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, T, 300)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    trajectories.append({
        'angle_deg': ang,
        'theta': theta,
        'T': T,
        'x': x,
        'y': y,
        'color': c
    })

# Determine plotting limits
max_range = max(tr['x'][-1] for tr in trajectories)
max_height = max(np.max(tr['y']) for tr in trajectories)

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(9, 6))

# Plot trajectories
for tr in trajectories:
    ax.plot(tr['x'], tr['y'], lw=2.2, color=tr['color'], label=f"{tr['angle_deg']}°")

# Choose reference angle (40°) to annotate key parameters
ref_angle = 40
ref = next(tr for tr in trajectories if tr['angle_deg'] == ref_angle)
ref_theta = ref['theta']
ref_color = ref['color']
ref_x, ref_y = ref['x'], ref['y']
R = v0**2 * np.sin(2 * ref_theta) / g
H = (v0 * np.sin(ref_theta))**2 / (2 * g)
T = 2 * v0 * np.sin(ref_theta) / g

# Apex point for reference trajectory
apex_idx = np.argmax(ref_y)
x_apex, y_apex = ref_x[apex_idx], ref_y[apex_idx]

# Draw ground line
ax.axhline(0, color='0.25', lw=1.5)

# Mark apex with a football-like marker
ax.plot(x_apex, y_apex, 'o', ms=9, mfc='#8B4513', mec='k', mew=1.0, zorder=5)

# Dashed guides: max height and range
ax.hlines(H, 0, x_apex, colors='0.6', linestyles='dashed', lw=1.2)
ax.vlines(R, 0, 0.02 * max_height + 0.001, colors='0.6', linestyles='dashed', lw=1.2)

# Annotate max height
ax.annotate('Max height H', xy=(x_apex, y_apex), xytext=(x_apex * 0.6, y_apex + 0.18 * max_height),
            textcoords='data', ha='center', va='bottom', color=ref_color,
            arrowprops=dict(arrowstyle='->', color=ref_color, lw=1.5))

# Range double-headed arrow near ground
y_offset = -0.08 * max_height
ax.annotate('', xy=(R, y_offset), xytext=(0, y_offset),
            arrowprops=dict(arrowstyle='<->', color='0.3', lw=1.5))
ax.text(R / 2, y_offset - 0.02 * max_height, 'Range R', ha='center', va='top', color='0.25')

# Initial velocity arrow and angle indication
arrow_len = 0.18 * max_range
ax.annotate('', xy=(arrow_len * np.cos(ref_theta), arrow_len * np.sin(ref_theta)), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=ref_color, lw=2))
ax.text(arrow_len * np.cos(ref_theta) * 0.9, arrow_len * np.sin(ref_theta) * 0.9, 'v0',
        color=ref_color, ha='left', va='bottom')

# Key formulas and computed values box
box_text = (rf"Given v$_0$ = {v0:.1f} m/s,  $\theta$ = {ref_angle}°"\
            + "\n" +
            rf"$R = \frac{{v_0^2\sin(2\theta)}}{{g}} = {R:.1f}\,\mathrm{{m}}$" + "\n" +
            rf"$H = \frac{{v_0^2\sin^2\theta}}{{2g}} = {H:.1f}\,\mathrm{{m}}$" + "\n" +
            rf"$T = \frac{{2v_0\sin\theta}}{{g}} = {T:.2f}\,\mathrm{{s}}$")
ax.text(0.98, 0.98, box_text, transform=ax.transAxes, ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.95))

# Labels, legend, grid
ax.set_title('Football Projectile Motion: Trajectory and Key Parameters')
ax.set_xlabel('Horizontal distance (m)')
ax.set_ylabel('Height (m)')
ax.legend(title='Kick angle', loc='upper left')
ax.grid(True, linestyle=':', color='0.85')

# Limits and layout
ax.set_xlim(0, 1.05 * max_range)
ax.set_ylim(min(-0.12 * max_height, -0.5), 1.18 * max_height)

plt.tight_layout()
plt.savefig('football_projectile_trajectory.png', dpi=220, bbox_inches='tight')
plt.close()

import numpy as np
import matplotlib.pyplot as plt

# Parameters (cricket context)
g = 9.81                  # gravitational acceleration (m/s^2)
m = 0.156                # mass of a cricket ball (kg)
v0 = 30.0                # initial speed (m/s) ~ fast throw/bowl
angle_deg = 35.0         # launch angle (degrees)
y0 = 1.5                 # release height (m)

# Derived quantities
angle = np.deg2rad(angle_deg)
vx0 = v0 * np.cos(angle)
vy0 = v0 * np.sin(angle)

# Time of flight until the ball hits the ground (y=0)
# Solve y(t) = y0 + vy0*t - 0.5*g*t^2 = 0 for t > 0
# t = (vy0 + sqrt(vy0^2 + 2*g*y0)) / g
t_flight = (vy0 + np.sqrt(vy0**2 + 2*g*y0)) / g

# Time array
N = 400
t = np.linspace(0, t_flight, N)

# Trajectory
x = vx0 * t
y = y0 + vy0 * t - 0.5 * g * t**2

# Velocities and momentum components
vx = np.full_like(t, vx0)
vy = vy0 - g * t
px = m * vx
py = m * vy

# Energies (relative potential energy with ground at y=0)
KE = 0.5 * m * (vx**2 + vy**2)
PE = m * g * np.maximum(y, 0)  # clamp below ground to 0 for neatness near landing
E_total = KE + PE

# Key points: apex (vy = 0)
t_apex = vy0 / g
x_apex = vx0 * t_apex
y_apex = y0 + vy0 * t_apex - 0.5 * g * t_apex**2

# Figure layout
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10
})

fig = plt.figure(figsize=(12, 7))
from matplotlib.gridspec import GridSpec

gs = GridSpec(2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.32)
ax_traj = fig.add_subplot(gs[:, 0])
ax_E = fig.add_subplot(gs[0, 1])
ax_p = fig.add_subplot(gs[1, 1])

# 1) Trajectory with momentum vectors
ax_traj.plot(x, y, color='tab:blue', lw=2, label='Ball trajectory')
ax_traj.axhline(0, color='k', lw=1, alpha=0.5)

# Mark release, apex, and landing
ax_traj.plot([x[0]], [y[0]], marker='o', color='tab:green', label='Release')
ax_traj.plot([x_apex], [y_apex], marker='^', color='tab:purple', label='Apex')
ax_traj.plot([x[-1]], [0], marker='s', color='tab:red', label='Impact')

# Momentum vectors along the path (aligned with velocity)
# Scale vectors so they are visible in data units
n_arrows = 8
idxs = np.linspace(10, N-10, n_arrows, dtype=int)
Lscale = 8.0  # meters for visual scaling of vectors
u = (vx[idxs] / v0) * Lscale
v = (vy[idxs] / v0) * Lscale
ax_traj.quiver(x[idxs], y[idxs], u, v, angles='xy', scale_units='xy', scale=1,
               color='tab:orange', width=0.004, alpha=0.9, label='Momentum direction')

# Labels and annotations
ax_traj.set_title('(A) Projectile path with momentum vectors')
ax_traj.set_xlabel('Horizontal distance x (m)')
ax_traj.set_ylabel('Height y (m)')
ax_traj.grid(True, ls='--', alpha=0.3)

# Text annotations on trajectory
ax_traj.annotate('Apex (vy = 0)\nKE minimized, PE maximized',
                 xy=(x_apex, y_apex), xytext=(x_apex + 10, y_apex + 10),
                 arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
ax_traj.text(0.02, 0.95,
             'Ignoring air resistance:\n- Horizontal momentum (px) constant\n- Momentum always along velocity',
             transform=ax_traj.transAxes, va='top', ha='left', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, ec='gray'))

# Set reasonable limits
x_margin = max(10, 0.05 * x.max())
y_margin = max(2, 0.1 * (max(y.max(), y0)))
ax_traj.set_xlim(-2, x.max() + x_margin)
ax_traj.set_ylim(-2, max(y.max(), y0) + y_margin)
ax_traj.legend(loc='lower right')

# 2) Energies vs time
ax_E.plot(t, KE, color='tab:red', lw=2, label='Kinetic energy (KE)')
ax_E.plot(t, PE, color='tab:green', lw=2, label='Potential energy (PE)')
ax_E.plot(t, E_total, color='tab:blue', lw=2, ls='--', label='Total energy (KE + PE)')
ax_E.set_title('(B) Energy exchange during flight')
ax_E.set_xlabel('Time t (s)')
ax_E.set_ylabel('Energy (J)')
ax_E.grid(True, ls='--', alpha=0.3)
ax_E.legend(loc='best')
ax_E.annotate('Energy trades between KE and PE\nTotal remains constant',
              xy=(t_apex, E_total[np.searchsorted(t, t_apex)]),
              xytext=(t_apex + 0.5, np.max(E_total)*0.9),
              arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)

# 3) Momentum components vs time
ax_p.plot(t, px, color='tab:orange', lw=2, ls='-', label='px (horizontal) — constant')
ax_p.plot(t, py, color='tab:purple', lw=2, label='py (vertical) — changes linearly')
ax_p.axhline(0, color='k', lw=1, alpha=0.4)
ax_p.set_title('(C) Momentum components')
ax_p.set_xlabel('Time t (s)')
ax_p.set_ylabel('Momentum (kg·m/s)')
ax_p.grid(True, ls='--', alpha=0.3)
ax_p.legend(loc='best')

# Global title with context
fig.suptitle('Projectile Motion of a Cricket Ball: Momentum and Energy\n'
             f'm = {m:.3f} kg, v0 = {v0:.0f} m/s, angle = {angle_deg:.0f}°, release height = {y0:.1f} m, g = {g:.2f} m/s²',
             y=0.98, fontsize=14)

# Save figure
outfile = 'cricket_projectile_energy_momentum.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f'Figure saved as {outfile}')

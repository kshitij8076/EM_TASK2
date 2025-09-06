import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81          # gravity (m/s^2)
y0 = 2.0          # initial height (m), approximate ball release height
v0x = 12.0        # horizontal kick speed (m/s)

# Time to impact (both balls): depends only on vertical motion here (v0y = 0)
t_f = np.sqrt(2 * y0 / g)

# Time arrays
T = np.linspace(0, t_f, 300)

# Kinematics (no air resistance)
# Kicked forward: v0y = 0, v0x > 0
x_kick = v0x * T
y_common = y0 - 0.5 * g * T**2  # vertical motion shared by both

# Dropped: v0x = 0
x_drop = np.zeros_like(T)

# Mark a few simultaneous snapshots
t_marks = np.array([0.2, 0.4, 0.6]) * t_f
xk_marks = v0x * t_marks
xd_marks = np.zeros_like(t_marks)
y_marks = y0 - 0.5 * g * t_marks**2

# Landing positions
xk_land = v0x * t_f
yd_land = 0.0  # y at landing

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150, gridspec_kw={"width_ratios": [6, 5]})
ax_xy, ax_yt = axes

# Left: Trajectories in x-y plane
ax_xy.plot(x_kick, y_common, color='tab:green', lw=2.5, label='Kicked forward')
ax_xy.plot(x_drop, y_common, color='tab:blue', lw=2.5, label='Dropped')

# Ground line
ax_xy.axhline(0, color='k', lw=1, alpha=0.6)

# Simultaneous snapshots (same times -> same heights)
for ym in y_marks:
    ax_xy.axhline(ym, color='gray', lw=1, ls=':', alpha=0.4)

ax_xy.scatter(xk_marks, y_marks, s=50, facecolor='none', edgecolor='tab:green', lw=2, zorder=5)
ax_xy.scatter(xd_marks, y_marks, s=50, facecolor='none', edgecolor='tab:blue', lw=2, zorder=5)

# Landing markers and annotations
ax_xy.plot([xk_land], [0], marker='o', color='tab:green', ms=6, zorder=6)
ax_xy.plot([0], [0], marker='o', color='tab:blue', ms=6, zorder=6)
ax_xy.annotate(f"both land at t = {t_f:.2f} s", xy=(xk_land, 0), xytext=(xk_land*0.45, y0*0.35),
               arrowprops=dict(arrowstyle='->', color='0.2'), color='0.2', fontsize=10)

# Labels and styling
ax_xy.set_title('Same vertical motion at equal times', fontsize=12)
ax_xy.set_xlabel('Horizontal distance x (m)')
ax_xy.set_ylabel('Height y (m)')
ax_xy.set_xlim(-1, xk_land + 1)
ax_xy.set_ylim(-0.2, y0 + 0.5)
ax_xy.legend(frameon=False, loc='upper right')
ax_xy.grid(True, alpha=0.2)

# Right: Vertical position vs time (both overlap)
ax_yt.plot(T, y_common, color='tab:green', lw=2.5, label='Kicked forward (y(t))')
ax_yt.plot(T, y_common, color='tab:blue', lw=1.8, ls='--', label='Dropped (y(t))')
ax_yt.axvline(t_f, color='k', ls='--', lw=1)
ax_yt.annotate('impact time (same for both)', xy=(t_f, 0), xytext=(t_f*0.55, y0*0.55),
               arrowprops=dict(arrowstyle='->', color='0.2'), color='0.2', fontsize=10)

# Time markers
for tm in t_marks:
    ym = y0 - 0.5 * g * tm**2
    ax_yt.plot([tm], [ym], marker='o', color='0.25', ms=4)

ax_yt.set_title('Vertical position vs time (overlapping curves)', fontsize=12)
ax_yt.set_xlabel('Time t (s)')
ax_yt.set_ylabel('Height y (m)')
ax_yt.set_xlim(0, t_f * 1.05)
ax_yt.set_ylim(-0.2, y0 + 0.5)
ax_yt.legend(frameon=False, loc='upper right')
ax_yt.grid(True, alpha=0.2)

# Overall message
fig.suptitle('Independence of Horizontal and Vertical Motions (Football)', fontsize=14, y=0.98)
fig.text(0.5, 0.01, 'Horizontal speed changes range only; time in air depends on vertical motion (initial vertical velocity and gravity).',
         ha='center', va='bottom', fontsize=10, color='0.25')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])

# Save figure
out_name = 'projectile_independence_football.png'
plt.savefig(out_name, dpi=300)
print(f'Saved figure to {out_name}')

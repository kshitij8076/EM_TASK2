import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # m/s^2
y0 = 1.5  # release height (m), approximate shoulder height of a fielder
vx = 18.0  # horizontal speed (m/s), same for left-panel throws

# Different vertical launch speeds for throws with the same horizontal speed
vy_list = [3.0, 8.0, 13.0]  # low, flat, lofted
colors = ['tab:green', 'tab:orange', 'tab:blue']
labels = [
    'Low throw (vy0 = 3 m/s)',
    'Flat throw (vy0 = 8 m/s)',
    'Lofted throw (vy0 = 13 m/s)'
]

# Utility to compute flight time until the ball hits the ground (y=0)
def flight_time(vy0, y0=y0, g=g):
    # Solve y0 + vy0 t - 0.5 g t^2 = 0 for positive root
    return (vy0 + np.sqrt(vy0**2 + 2*g*y0)) / g

# Prepare figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# LEFT PANEL: Same vx, different vy0
ax = axes[0]

# Full trajectories for context
t_flights = [flight_time(vy) for vy in vy_list]

ymax_marks = 0.0
for vy, c, lab in zip(vy_list, colors, labels):
    tf = flight_time(vy)
    t = np.linspace(0, tf, 300)
    x = vx * t
    y = y0 + vy * t - 0.5 * g * t**2
    y = np.maximum(y, 0)
    ax.plot(x, y, color=c, lw=2, label=lab)

# Time-synchronized markers (up to earliest landing)
t_sync = min(t_flights)
t_marks = np.linspace(0, t_sync, 6)

# Vertical guide lines at equal times (same x for all because vx is the same)
for t in t_marks:
    x_mark = vx * t
    ax.axvline(x_mark, color='0.85', lw=1, linestyle='--', zorder=0)

# Place synchronized dots on each trajectory
for vy, c in zip(vy_list, colors):
    x_marks = vx * t_marks
    y_marks = y0 + vy * t_marks - 0.5 * g * t_marks**2
    y_marks = np.maximum(y_marks, 0)
    ymax_marks = max(ymax_marks, np.max(y_marks))
    ax.plot(x_marks, y_marks, 'o', color=c, ms=4)

# Annotations and styling
ax.axhline(0, color='saddlebrown', lw=2)
ax.set_xlim(0, vx * (t_sync + 0.25))
ax.set_ylim(0, max(ymax_marks + 1.0, y0 + 1.0))
ax.set_xlabel('Horizontal distance x (m)')
ax.set_ylabel('Height y (m)')
ax.set_title('Same vx, different vy: motions are independent')
ax.legend(loc='upper right', fontsize=8, frameon=False)
ax.text(0.03, 0.95, 'x(t) = vx · t (vx constant)', transform=ax.transAxes,
        va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, lw=0))
ax.text(0.03, 0.88, 'Vertical dashed lines = same time for all balls', transform=ax.transAxes,
        va='top', ha='left', fontsize=8, color='0.3')
ax.annotate('Gravity acts downward only (ay = -g)',
            xy=(0.5 * vx * t_sync, y0 + 0.5),
            xytext=(0.5 * vx * t_sync + 3, y0 + 3.5),
            arrowprops=dict(arrowstyle='->', color='k'), fontsize=9)
ax.grid(True, color='0.9', linestyle='-')

# RIGHT PANEL: Dropped ball vs horizontally thrown ball from same height
ax2 = axes[1]

vx2 = 18.0  # horizontal speed for the throw
vy0_drop = 0.0
vy0_throw = 0.0

# Both share the same vertical motion when vy0 is equal and gravity acts downward
t_flight_drop = np.sqrt(2 * y0 / g)
t = np.linspace(0, t_flight_drop, 250)
y_shared = y0 - 0.5 * g * t**2
x_throw = vx2 * t
x_drop = np.zeros_like(t)

ax2.plot(x_throw, y_shared, color='tab:blue', lw=2, label='Horizontal throw (vx > 0, vy0 = 0)')
ax2.plot(x_drop, y_shared, color='tab:red', lw=2, label='Dropped ball (vx = 0, vy0 = 0)')

# Time-synchronized markers and horizontal guides (same y at each time)
t_marks2 = np.linspace(0, t_flight_drop, 6)
y_marks2 = y0 - 0.5 * g * t_marks2**2
x_marks_throw = vx2 * t_marks2
x_marks_drop = np.zeros_like(t_marks2)

for yline in y_marks2:
    ax2.axhline(yline, color='0.85', lw=1, linestyle='--', zorder=0)
ax2.plot(x_marks_throw, y_marks2, 'o', color='tab:blue', ms=4)
ax2.plot(x_marks_drop, y_marks2, 'o', color='tab:red', ms=4)

ax2.axhline(0, color='saddlebrown', lw=2)
ax2.set_xlim(-1, vx2 * t_flight_drop + 7)
ax2.set_ylim(0, y0 + 0.8)
ax2.set_xlabel('Horizontal distance x (m)')
ax2.set_title('Drop vs horizontal throw: same y(t), different x(t)')
ax2.legend(loc='upper right', fontsize=8, frameon=False)
ax2.text(0.03, 0.95, 'y(t) = y0 - 0.5 g t^2 (same for both)', transform=ax2.transAxes,
         va='top', ha='left', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, lw=0))
ax2.annotate('Same vertical fall at equal times',
             xy=(x_marks_throw[3], y_marks2[3]),
             xytext=(x_marks_throw[3] + 2, y_marks2[3] + 0.35),
             arrowprops=dict(arrowstyle='->', color='k'), fontsize=9)
ax2.grid(True, color='0.9', linestyle='-')

# Titles and footnote
fig.suptitle('Independence of Horizontal and Vertical Motion — Cricket Ball', fontsize=14, y=0.98)
fig.text(0.5, 0.02, 'Neglecting air resistance; g = 9.81 m/s^2; release height = 1.5 m', ha='center', fontsize=9)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])

# Save figure
outname = 'cricket_projectile_independence.png'
plt.savefig(outname, dpi=200, bbox_inches='tight')

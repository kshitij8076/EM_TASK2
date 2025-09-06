import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# Parameters for the cricket shot (idealized, no air resistance)
g = 9.81  # m/s^2
v0 = 35.0  # initial speed of the ball in m/s (typical strong lofted shot)
angles_deg = [15, 30, 45, 60]  # launch angles to illustrate

# Cricket pitch reference
pitch_length = 20.12  # meters

# Helper functions
def trajectory_points(v0, theta_deg, g=9.81, n=400):
    theta = np.deg2rad(theta_deg)
    t_f = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_f, n)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    # Ensure numerical floor at ground
    y = np.maximum(y, 0)
    return x, y, t_f

def range_of(v0, theta_deg, g=9.81):
    theta = np.deg2rad(theta_deg)
    return (v0**2) * np.sin(2 * theta) / g

def apex_of(v0, theta_deg, g=9.81):
    theta = np.deg2rad(theta_deg)
    t_peak = (v0 * np.sin(theta)) / g
    x_peak = v0 * np.cos(theta) * t_peak
    y_peak = (v0**2) * (np.sin(theta)**2) / (2 * g)
    return x_peak, y_peak

# Compute trajectories
trajectories = {}
max_range = 0
max_height = 0
for ang in angles_deg:
    x, y, tf = trajectory_points(v0, ang, g)
    R = range_of(v0, ang, g)
    x_ap, y_ap = apex_of(v0, ang, g)
    trajectories[ang] = {
        'x': x,
        'y': y,
        'R': R,
        'apex': (x_ap, y_ap),
        'tf': tf
    }
    max_range = max(max_range, R)
    max_height = max(max_height, y.max())

# Figure setup
plt.rcParams.update({
    'figure.figsize': (11, 6.5),
    'axes.grid': True,
    'grid.alpha': 0.25,
    'font.size': 12
})

fig, ax = plt.subplots()

# Plot trajectories
colors = {
    15: '#1f77b4',
    30: '#2ca02c',
    45: '#ff7f0e',
    60: '#d62728'
}
for ang in angles_deg:
    dat = trajectories[ang]
    ax.plot(dat['x'], dat['y'], label=f"{ang}\N{DEGREE SIGN}", lw=2.5, color=colors[ang])
    # Mark landing point
    ax.scatter([dat['R']], [0], color=colors[ang], s=22, zorder=3)

# Highlight apex on the 45° shot
apx45 = trajectories[45]['apex']
ax.scatter([apx45[0]], [apx45[1]], color=colors[45], s=60, zorder=4)
ax.annotate(
    'Apex (max height)',
    xy=(apx45[0], apx45[1]),
    xytext=(apx45[0] + 8, apx45[1] + 6),
    arrowprops=dict(arrowstyle='->', lw=1.5, color=colors[45]),
    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=colors[45], alpha=0.9)
)

# Draw the ground line
xmax = max_range * 1.06
ax.hlines(0, 0, xmax, colors='k', lw=1.5)

# Annotate RANGE for the 45° shot (double-headed arrow along ground)
R45 = trajectories[45]['R']
range_y = -1.9  # place below ground for clarity
ax.annotate('', xy=(0, range_y), xytext=(R45, range_y),
            arrowprops=dict(arrowstyle='<->', color=colors[45], lw=2))
ax.text(R45/2, range_y + 0.15, f"Range (45\N{DEGREE SIGN}) = {R45:0.1f} m",
        color=colors[45], ha='center', va='bottom')

# Show equal range for complementary angles 30° and 60°
R30 = trajectories[30]['R']
ax.vlines(R30, 0, max_height*0.98, colors='gray', linestyles='dashed', lw=1.5)
ax.text(R30 + 1.2, max_height*0.8, 'Same range\n(30\N{DEGREE SIGN} & 60\N{DEGREE SIGN})',
        color='gray', va='center')

# Draw a simple cricket pitch reference (not to scale visually on this axis)
# Bracket for 20.12 m pitch
pitch_y = -0.8
ax.annotate('', xy=(0, pitch_y), xytext=(pitch_length, pitch_y),
            arrowprops=dict(arrowstyle='<->', color='#8B4513', lw=1.8))
ax.text(pitch_length/2, pitch_y + 0.12, 'Cricket pitch: 20.12 m', color='#8B4513',
        ha='center', va='bottom')

# Stylized wickets at both ends of the pitch
def draw_wickets(ax, x, height=0.9, spacing=0.35, color='#8B4513', lw=3):
    for dx in (-spacing, 0, spacing):
        ax.vlines(x + dx, 0, height, colors=color, linewidth=lw)

draw_wickets(ax, 0.0)
draw_wickets(ax, pitch_length)

# Labels and aesthetics
ax.set_title('Projectile Motion in Cricket: Trajectory and Path Characteristics', pad=12)
ax.set_xlabel('Horizontal distance (m)')
ax.set_ylabel('Height (m)')
ax.set_xlim(0, xmax)
ax.set_ylim(-2.4, max(max_height * 1.12, 10))
ax.legend(title='Launch angle', loc='upper left', framealpha=0.95)

# Helpful notes
note_text = (
    "Idealized vacuum trajectories (no air resistance).\n"
    "Complementary angles (θ and 90°−θ) have the same range at fixed speed.\n"
    "45° gives maximum range for a given speed (in vacuum)."
)
ax.text(xmax*0.53, max_height*0.55, note_text, fontsize=10, bbox=dict(fc='white', ec='0.7', alpha=0.9))

plt.tight_layout()
plt.savefig('cricket_projectile_trajectory.png', dpi=200, bbox_inches='tight')
plt.close()

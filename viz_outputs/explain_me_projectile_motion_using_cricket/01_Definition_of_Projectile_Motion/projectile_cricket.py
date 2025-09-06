import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Parameters
g = 9.81  # gravity (m/s^2)
v0 = 32.0  # initial speed of the cricket ball (m/s)
angle_deg = 35.0  # launch angle in degrees
angle = np.deg2rad(angle_deg)

# Air properties and ball properties for drag model
rho = 1.225  # air density (kg/m^3)
Cd = 0.5     # drag coefficient for a roughly spherical ball
r = 0.036    # radius of a cricket ball (m)
A = np.pi * r**2  # cross-sectional area (m^2)
m = 0.156    # mass of a cricket ball (kg)

k = 0.5 * rho * Cd * A / m  # drag acceleration constant

# Ideal (no air resistance) trajectory
T_ideal = 2 * v0 * np.sin(angle) / g
T_plot = T_ideal

t_ideal = np.linspace(0, T_ideal, 400)
x_ideal = v0 * np.cos(angle) * t_ideal
y_ideal = v0 * np.sin(angle) * t_ideal - 0.5 * g * t_ideal**2

# Drag trajectory (numerical integration - simple Euler method)
dt = 0.002
vx = v0 * np.cos(angle)
vy = v0 * np.sin(angle)
x, y = 0.0, 0.0
x_drag_list = [x]
y_drag_list = [y]

x_land_drag = None

for i in range(int(20 / dt)):  # cap at 20 s as a safe upper bound
    v = np.hypot(vx, vy)
    ax = -k * v * vx
    ay = -g - k * v * vy

    # Update velocities
    vx += ax * dt
    vy += ay * dt
    # Update positions
    x_new = x + vx * dt
    y_new = y + vy * dt

    x_drag_list.append(x_new)
    y_drag_list.append(y_new)

    if y > 0 and y_new <= 0:
        # Linear interpolation to find a better landing x at y=0
        frac = y / (y - y_new)
        x_land_drag = x + frac * (x_new - x)
        break

    x, y = x_new, y_new

x_drag = np.array(x_drag_list)
y_drag = np.array(y_drag_list)

# Landing points
x_land_ideal = x_ideal[-1]
if x_land_drag is None:
    x_land_drag = x_drag[-1]

# Apex (ideal)
apex_idx = np.argmax(y_ideal)
apex_x = x_ideal[apex_idx]
apex_y = y_ideal[apex_idx]

# Figure
plt.figure(figsize=(10, 6))

# Ground line
xmin_plot = -1.0
xmax_plot = max(x_land_ideal, x_land_drag) * 1.05
plt.plot([xmin_plot, xmax_plot], [0, 0], color="#5a8f3d", lw=3, solid_capstyle='butt', alpha=0.8, label="Ground")

# Wickets (simple drawing to anchor the cricket context)
stump_width = 0.03
stump_height = 0.71
stump_gap = 0.09
wicket_x0 = -0.6
wicket_color = '#8b5a2b'
for i in range(3):
    rect = Rectangle((wicket_x0 + i * stump_gap, 0), stump_width, stump_height, facecolor=wicket_color, edgecolor='black', zorder=5)
    plt.gca().add_patch(rect)
# Bails
bail_height = stump_height + 0.03
bail_len = stump_gap - 0.02
bail_thick = 0.015
bail1 = Rectangle((wicket_x0 + 0.01, bail_height), bail_len, bail_thick, facecolor=wicket_color, edgecolor='black', zorder=6)
bail2 = Rectangle((wicket_x0 + stump_gap + 0.01, bail_height), bail_len, bail_thick, facecolor=wicket_color, edgecolor='black', zorder=6)
plt.gca().add_patch(bail1)
plt.gca().add_patch(bail2)

# Plot trajectories
plt.plot(x_ideal, y_ideal, color="#1f77b4", lw=2.5, label="No air resistance (parabola)")
plt.plot(x_drag, y_drag, color="#ff7f0e", lw=2.5, ls='--', label="With air resistance")

# Annotate initial velocity components
origin = np.array([0.0, 0.0])
len_scale = 0.6  # scale for arrow lengths for clarity
v0x = v0 * np.cos(angle)
v0y = v0 * np.sin(angle)

plt.arrow(0, 0, len_scale * np.cos(angle), len_scale * np.sin(angle),
          head_width=0.05, head_length=0.08, fc='#2ca02c', ec='#2ca02c', length_includes_head=True)
plt.text(0.05, 0.08, f"Initial speed v0 = {v0:.0f} m/s\n(angle {angle_deg:.0f}°)", color='#2ca02c')

# Component arrows
plt.arrow(0, 0, len_scale, 0, head_width=0.04, head_length=0.06, fc='#2ca02c', ec='#2ca02c', alpha=0.8, length_includes_head=True)
plt.text(len_scale + 0.05, 0.02, "Horizontal component v0 cos(θ)\n(approx. constant without drag)", color='#2ca02c', va='bottom')

plt.arrow(0, 0, 0, len_scale * np.sin(angle), head_width=0.04, head_length=0.06, fc='#2ca02c', ec='#2ca02c', alpha=0.8, length_includes_head=True)
plt.text(0.02, len_scale * np.sin(angle) + 0.05, "Vertical component v0 sin(θ)\n(changes due to gravity)", color='#2ca02c', va='bottom')

# Gravity annotation
plt.arrow(xmin_plot + 0.1, apex_y * 0.9, 0, -0.4, head_width=0.05, head_length=0.08, fc='k', ec='k', length_includes_head=True)
plt.text(xmin_plot + 0.15, apex_y * 0.9 - 0.2, "Gravity g\nacts downward", va='center')

# Apex annotation (ideal)
plt.plot(apex_x, apex_y, 'o', color='#1f77b4')
plt.annotate("Highest point\n(vertical speed = 0)",
             xy=(apex_x, apex_y), xytext=(apex_x + 5, apex_y + 5),
             arrowprops=dict(arrowstyle='->', color='#1f77b4'), color='#1f77b4')

# Landing points
plt.plot([x_land_ideal], [0], marker='o', color='#1f77b4')
plt.text(x_land_ideal, -0.4, "Landing (no air)", color='#1f77b4', ha='center', va='top')

plt.plot([x_land_drag], [0], marker='o', color='#ff7f0e')
plt.text(x_land_drag, -0.8, "Landing (with air)\n(shorter range)", color='#ff7f0e', ha='center', va='top')

# Context label
title = "Projectile motion in cricket: a lofted shot"
plt.title(title)
plt.xlabel("Horizontal distance (m)")
plt.ylabel("Height (m)")
plt.legend(loc='upper right')

# Set limits and aesthetics
ymax_plot = max(np.max(y_ideal), np.max(y_drag)) * 1.15
plt.xlim(xmin_plot, xmax_plot)
plt.ylim(-1.2, ymax_plot)
plt.grid(True, alpha=0.2)

plt.tight_layout()

# Save figure
outfile = "projectile_cricket.png"
plt.savefig(outfile, dpi=200)
print(f"Saved figure to {outfile}")

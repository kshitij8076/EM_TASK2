import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Physics constants
g = 9.81  # m/s^2

# Helper functions

def trajectory(v0, theta_deg, g=9.81, n=300):
    theta = np.deg2rad(theta_deg)
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, n)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    y = np.maximum(y, 0)
    return x, y, t_flight


def apex_point(v0, theta_deg, g=9.81):
    theta = np.deg2rad(theta_deg)
    t_apex = v0 * np.sin(theta) / g
    x_apex = v0 * np.cos(theta) * t_apex
    y_apex = (v0 * np.sin(theta))**2 / (2 * g)
    return x_apex, y_apex


def range_ideal(v0, theta_deg, g=9.81):
    theta = np.deg2rad(theta_deg)
    return (v0**2 * np.sin(2 * theta)) / g


def draw_wickets(ax, x0=0.7, y0=0.0, stump_height=0.71, stump_d=0.035, gap=0.09, color="#6b4f1d"):
    # Three stumps
    for i in range(3):
        ax.add_patch(Rectangle((x0 + i * gap, y0), stump_d, stump_height, color=color, zorder=3))
    # Simple bails
    bail_y = y0 + stump_height
    ax.add_patch(Rectangle((x0, bail_y), gap*2 + stump_d, 0.02, color=color, zorder=3))

# Figure setup
plt.figure(figsize=(11, 9))

# Top panel: effect of launch angle at fixed speed
ax1 = plt.subplot(2, 1, 1)
angles = [15, 30, 45, 60, 75]
v0_fixed = 30  # m/s
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(angles)))

xmax1 = 0
ymax1 = 0
lines1 = []
for ang, col in zip(angles, colors):
    x, y, _ = trajectory(v0_fixed, ang, g)
    ax1.plot(x, y, color=col, lw=2.5, label=f"{ang}\N{DEGREE SIGN}")
    xmax1 = max(xmax1, x[-1])
    ymax1 = max(ymax1, y.max())

# Add wickets near origin for cricket context
draw_wickets(ax1, x0=0.7, y0=0.0)

# Annotate optimal angle (idealized)
xR_45 = range_ideal(v0_fixed, 45, g)
ax1.plot([xR_45], [0], marker='o', color=colors[2], ms=6, zorder=4)
ax1.annotate("Longest range\n(idealized ~45\N{DEGREE SIGN})",
             xy=(xR_45, 0), xytext=(xR_45*0.6, ymax1*0.55),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6'))

# Annotate steep and low angles
x_apex_hi, y_apex_hi = apex_point(v0_fixed, 75, g)
ax1.annotate("Steep angle: higher apex\nshorter distance",
             xy=(x_apex_hi, y_apex_hi), xytext=(x_apex_hi*0.5, y_apex_hi*1.1),
             arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=10, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6'))

x_apex_lo, y_apex_lo = apex_point(v0_fixed, 15, g)
ax1.annotate("Low angle: low height\nshort hop",
             xy=(x_apex_lo, y_apex_lo), xytext=(x_apex_lo*2.2, y_apex_lo*2.8 + 0.2),
             arrowprops=dict(arrowstyle='->', lw=1.2), fontsize=10, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.6'))

ax1.set_xlim(0, xmax1 * 1.08)
ax1.set_ylim(0, ymax1 * 1.2)
ax1.set_xlabel("Horizontal distance (m)")
ax1.set_ylabel("Height (m)")
ax1.set_title("Effect of Launch Angle at Fixed Speed (v0 = 30 m/s)")
ax1.grid(True, ls='--', alpha=0.4)
leg1 = ax1.legend(title="Angle", frameon=True)
leg1.get_frame().set_alpha(0.9)

# Bottom panel: effect of initial speed at fixed angle
ax2 = plt.subplot(2, 1, 2)
angle_fixed = 45  # degrees
speeds = [20, 28, 36]  # m/s
colors2 = plt.cm.plasma(np.linspace(0.15, 0.85, len(speeds)))

xmax2 = 0
ymax2 = 0
for v0, col in zip(speeds, colors2):
    x, y, _ = trajectory(v0, angle_fixed, g)
    ax2.plot(x, y, color=col, lw=2.5, label=f"{v0} m/s")
    # Mark landing point
    ax2.plot([x[-1]], [0], marker='o', color=col, ms=6)
    xmax2 = max(xmax2, x[-1])
    ymax2 = max(ymax2, y.max())

# Wickets again for context
draw_wickets(ax2, x0=0.7, y0=0.0)

# Annotate speed influence near the farthest curve
ax2.annotate("Higher initial speed\n→ farther AND higher",
             xy=(xmax2, 0), xytext=(xmax2*0.55, ymax2*0.6),
             arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6'))

ax2.set_xlim(0, xmax2 * 1.08)
ax2.set_ylim(0, ymax2 * 1.2)
ax2.set_xlabel("Horizontal distance (m)")
ax2.set_ylabel("Height (m)")
ax2.set_title(f"Effect of Initial Speed at Fixed Angle (\N{GREEK SMALL LETTER THETA} = {angle_fixed}\N{DEGREE SIGN})")
ax2.grid(True, ls='--', alpha=0.4)
leg2 = ax2.legend(title="Speed", frameon=True)
leg2.get_frame().set_alpha(0.9)

# Overall figure title and note
plt.suptitle("Projectile Motion in Cricket: Influence of Launch Angle and Initial Velocity", fontsize=14, y=0.98)

# Assumptions note
note = ("Idealized model: no air resistance, flat ground, same launch and landing height.\n"
        "In real cricket, air drag, spin, and wind shift the optimal angle below 45\N{DEGREE SIGN}.")
plt.figtext(0.5, 0.01, note, ha='center', va='bottom', fontsize=9, bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='0.7'))

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

outfile = "cricket_projectile_influence_angle_velocity.png"
plt.savefig(outfile, dpi=200)
print(f"Saved figure to {outfile}")

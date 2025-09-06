import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81  # m/s^2
boundary_distance = 70  # meters (typical boundary distance for a six)


def trajectory(v0, angle_deg, g=9.81, num=300):
    theta = np.radians(angle_deg)
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, num)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    return x, y


# Figure setup
plt.figure(figsize=(12, 5.5))

# Left subplot: Vary launch angle at fixed speed
ax1 = plt.subplot(1, 2, 1)
v0_fixed = 35  # m/s
angles = [25, 35, 45, 55]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
max_x1 = 0
max_y1 = 0
for ang, c in zip(angles, colors):
    x, y = trajectory(v0_fixed, ang, g)
    ax1.plot(x, y, label=f"{ang}°", color=c, lw=2)
    ax1.plot(x[-1], y[-1], marker='o', color=c, ms=5)
    max_x1 = max(max_x1, x[-1])
    max_y1 = max(max_y1, y.max())

# Boundary marker
ax1.axvline(boundary_distance, color='gray', ls='--', lw=1.5)
ax1.text(boundary_distance, max_y1 * 0.05 + 0.5, "Boundary ~70 m (Six)", rotation=90,
         va='bottom', ha='center', color='gray')

# Ground line
ax1.axhline(0, color='k', lw=1)

# Annotation for 45° maximum range (on level ground, no air resistance)
x45, y45 = trajectory(v0_fixed, 45, g)
i_apex = np.argmax(y45)
ax1.annotate("45° gives maximum range\n(on level ground, no air resistance)",
             xy=(x45[i_apex], y45[i_apex]), xytext=(x45[i_apex] * 0.55, y45[i_apex] * 1.25),
             arrowprops=dict(arrowstyle='->', color='#2ca02c'),
             fontsize=9, color='#2ca02c', ha='center')

ax1.set_title(f"Effect of launch angle (speed = {v0_fixed} m/s)")
ax1.set_xlabel("Horizontal distance (m)")
ax1.set_ylabel("Height (m)")
ax1.grid(True, ls=':', alpha=0.6)
ax1.legend(title="Angle", frameon=False)
ax1.set_xlim(0, max_x1 * 1.1)
ax1.set_ylim(0, max_y1 * 1.2)

# Right subplot: Vary initial speed at fixed angle
ax2 = plt.subplot(1, 2, 2)
angle_fixed = 35  # degrees
speeds = [25, 30, 35, 40]  # m/s
colors2 = ["#9467bd", "#8c564b", "#17becf", "#e377c2"]
max_x2 = 0
max_y2 = 0
for v, c in zip(speeds, colors2):
    x, y = trajectory(v, angle_fixed, g)
    ax2.plot(x, y, label=f"{v} m/s", color=c, lw=2)
    ax2.plot(x[-1], y[-1], marker='o', color=c, ms=5)
    max_x2 = max(max_x2, x[-1])
    max_y2 = max(max_y2, y.max())

# Boundary marker
ax2.axvline(boundary_distance, color='gray', ls='--', lw=1.5)
ax2.text(boundary_distance, max_y2 * 0.05 + 0.5, "Boundary ~70 m (Six)", rotation=90,
         va='bottom', ha='center', color='gray')

# Ground line
ax2.axhline(0, color='k', lw=1)

ax2.set_title(f"Effect of initial speed (angle = {angle_fixed}°)")
ax2.set_xlabel("Horizontal distance (m)")
ax2.set_ylabel("Height (m)")
ax2.grid(True, ls=':', alpha=0.6)
ax2.legend(title="Speed", frameon=False)
ax2.set_xlim(0, max_x2 * 1.1)
ax2.set_ylim(0, max_y2 * 1.2)

# Suptitle and note
plt.suptitle("Cricket ball projectile motion: launch angle and initial speed", y=1.02, fontsize=13)
plt.figtext(0.5, -0.02,
            "Assumptions: level ground, no air resistance. Higher speed and an optimal angle increase both height and range.",
            ha='center', fontsize=9, color='dimgray')

plt.tight_layout()
plt.savefig("cricket_projectile_angle_velocity.png", dpi=200, bbox_inches='tight')
print("Saved figure: cricket_projectile_angle_velocity.png")

import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 9.81            # gravitational acceleration (m/s^2)
v0 = 30.0           # initial speed (m/s), typical strong throw
y0 = 1.5            # release height ~ shoulder height (m)
angles_deg = [20, 35, 50]  # different launch angles (degrees)
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# Wicket placement (for context)
wicket_x = 60.0     # distance to stumps (m)
wicket_height = 0.71
stump_spacing = 0.1  # spacing between stump centers (m)


def trajectory(v0, theta_deg, y0, g=9.81, n=400):
    theta = np.deg2rad(theta_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    t_f = (vy0 + np.sqrt(vy0**2 + 2 * g * y0)) / g
    t = np.linspace(0.0, t_f, n)
    x = vx0 * t
    y = y0 + vy0 * t - 0.5 * g * t**2
    return x, y, t_f

# Compute trajectories
trajectories = []
max_range = 0.0
for ang in angles_deg:
    x, y, tf = trajectory(v0, ang, y0, g)
    trajectories.append((ang, x, y))
    max_range = max(max_range, x[-1])

# Figure setup
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11
})

fig, ax = plt.subplots(figsize=(11, 6))

# Ground
ax.axhline(0, color="#4d7c0f", lw=2, alpha=0.8)
ax.axhspan(-0.8, 0, color="#c7e9c0", alpha=0.6)

# Plot trajectories
for (ang, x, y), c in zip(trajectories, colors):
    ax.plot(x, y, color=c, lw=2.5, label=f"{ang}°")

# Highlight sample ball positions on the 35° trajectory
mid_idx = [i for i, (ang, _, _) in enumerate(trajectories) if ang == 35][0]
_, x_mid, y_mid = trajectories[mid_idx]
sel = np.linspace(0, len(x_mid) - 1, 8, dtype=int)
ax.scatter(x_mid[sel], y_mid[sel], s=30, facecolor="white", edgecolor="#ff7f0e", zorder=5, label="ball positions (35°)")

# Wickets (stumps and bails)
for dx in [-stump_spacing, 0.0, stump_spacing]:
    ax.plot([wicket_x + dx, wicket_x + dx], [0, wicket_height], color="#8b5a2b", lw=5, solid_capstyle='butt', zorder=3)
# Bails across top of stumps
ax.plot([wicket_x - stump_spacing, wicket_x + stump_spacing], [wicket_height, wicket_height], color="#8b5a2b", lw=3, zorder=3)
ax.text(wicket_x, wicket_height + 0.4, "Wickets", ha="center", va="bottom", color="#8b5a2b")

# Release point
ax.scatter([0], [y0], s=60, color="black", zorder=6)
ax.annotate("Release: projectile motion begins", xy=(0, y0), xytext=(15, y0 + 8),
            arrowprops=dict(arrowstyle="->", lw=1.8, color="black"), ha="left", va="center")

# Gravity arrow and annotation
ax.annotate("Gravity g acts downward", xy=(0.92, 0.9), xycoords='axes fraction', xytext=(0.92, 0.9),
            textcoords='axes fraction', ha='right', va='center')
ax.annotate("", xy=(0.92, 0.78), xytext=(0.92, 0.9), xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'))

# Conceptual note
ax.text(0.55, 0.2, "After release: motion shaped only by gravity\n(Ignoring air resistance)", transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9))

# Labels, legend, limits
ax.set_title("Projectile Motion in Cricket: Curved flight of a thrown ball")
ax.set_xlabel("Horizontal distance x (m)")
ax.set_ylabel("Height y (m)")

xmax = max(max_range * 1.05, wicket_x + 8)
ax.set_xlim(0, xmax)
ax.set_ylim(0, 30)

# Legend (combine angle lines and ball positions)
handles, labels = ax.get_legend_handles_labels()
# Ensure angle legend groups nicely
ax.legend(handles, [f"Launch angle: {lab}" if lab.endswith("°") else lab for lab in labels],
          title="Same speed, different launch angles", loc="upper left", framealpha=0.95)

ax.grid(True, ls='--', alpha=0.3)

plt.tight_layout()
fig.savefig("projectile_motion_cricket.png", dpi=300, bbox_inches='tight')
plt.close(fig)

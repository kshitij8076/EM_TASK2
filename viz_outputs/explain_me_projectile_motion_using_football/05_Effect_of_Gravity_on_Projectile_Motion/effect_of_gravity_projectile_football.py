import numpy as np
import matplotlib.pyplot as plt

# Parameters (idealized: no air resistance)
g = 9.8  # m/s^2
v0 = 20.0  # initial speed (m/s)
angle_deg = 35.0  # kick angle (degrees)
angle = np.deg2rad(angle_deg)

# Initial velocity components
vx = v0 * np.cos(angle)
vy0 = v0 * np.sin(angle)

# Key times
t_peak = vy0 / g
T = 2 * vy0 / g  # time of flight (back to ground)

# Trajectory (with gravity)
t = np.linspace(0, T, 400)
x = vx * t
y = vy0 * t - 0.5 * g * t**2

# Useful extents
x_end = x[-1]
y_max = vy0**2 / (2 * g)

# Figure
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 15
})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.5))
fig.suptitle("Effect of Gravity on Projectile Motion (Football)")

# --------------------
# Panel A: Path on the field
# --------------------
# Ground line
ax1.axhline(0, color="0.7", lw=1)

# With-gravity trajectory
ax1.plot(x, y, color="tab:green", lw=2.5, label="With gravity: parabolic path")

# Mark the peak
x_peak = vx * t_peak
ax1.plot(x_peak, y_max, "o", color="tab:green")
ax1.annotate("Peak (vy = 0)", xy=(x_peak, y_max), xytext=(x_peak + 0.03 * x_end, y_max + 0.1 * (y_max + 1)),
             arrowprops=dict(arrowstyle="->", color="tab:green"), color="tab:green")

# "No gravity" straight-line motion: show as a dashed ray from the kick
L_dashed = 0.45 * x_end  # show a modest-length segment to keep it within frame
ux, uy = np.cos(angle), np.sin(angle)
x_ng = np.linspace(0, L_dashed, 100)
y_ng = (uy / ux) * x_ng  # y = x * tan(theta)
ax1.plot(x_ng, y_ng, linestyle="--", color="0.4", lw=2, label="Without gravity: straight-line motion")

# Velocity component arrows along the parabolic path
# Red: horizontal component (constant); Blue: vertical component (changes)
# Choose sample times avoiding endpoints
sample_ts = np.linspace(0.12 * T, 0.88 * T, 5)
scale = 0.08  # scale m/s to plot units (meters on axes)
for ts in sample_ts:
    xs = vx * ts
    ys = vy0 * ts - 0.5 * g * ts**2
    vy_t = vy0 - g * ts
    # Horizontal component (constant length)
    ax1.annotate("", xy=(xs + scale * vx, ys), xytext=(xs, ys),
                 arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    # Vertical component (varies; up then down)
    ax1.annotate("", xy=(xs, ys + scale * vy_t), xytext=(xs, ys),
                 arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))

# Gravity arrow (downward acceleration)
ax1.annotate("g = 9.8 m/s^2", xy=(0.85 * x_end, 0.75 * max(y_max, 1.0)),
             xytext=(0.85 * x_end, 0.75 * max(y_max, 1.0) + 2.5),
             arrowprops=dict(arrowstyle="->", color="0.2", lw=2), color="0.2", ha="center")

ax1.set_title("Trajectory on the field (idealized)")
ax1.set_xlabel("Horizontal distance x (m)")
ax1.set_ylabel("Height y (m)")
ax1.set_xlim(0, x_end * 1.05)
ax1.set_ylim(-0.5, max(y_max * 1.2, 6))
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

# --------------------
# Panel B: Velocity components vs time
# --------------------
vy_t = vy0 - g * t
ax2.plot(t, vy_t, color="tab:blue", lw=2.5, label="Vertical velocity vy(t)")
ax2.plot(t, np.full_like(t, vx), linestyle="--", color="tab:red", lw=2, label="Horizontal velocity vx (constant)")

# Zero line and peak marker
ax2.axhline(0, color="0.7", lw=1)
ax2.plot(t_peak, 0, "o", color="tab:blue")
ax2.annotate("vy = 0 at peak", xy=(t_peak, 0), xytext=(t_peak + 0.12 * T, 0 + 0.15 * v0),
             arrowprops=dict(arrowstyle="->", color="tab:blue"), color="tab:blue")

# Indicate slope = -g on vy(t)
# Draw a small reference segment near t=0.2T
ref_t = 0.2 * T
ref_v = vy0 - g * ref_t
dt = 0.12 * T
ax2.plot([ref_t, ref_t + dt], [ref_v, ref_v - g * dt], color="0.3", lw=2)
ax2.text(ref_t + 0.5 * dt, ref_v - 0.5 * g * dt, "slope = -g", color="0.2", ha="left", va="center")

ax2.set_title("Velocity components vs time")
ax2.set_xlabel("Time t (s)")
ax2.set_ylabel("Velocity (m/s)")
ax2.set_xlim(0, T)
ax2.set_ylim(min(vy_t.min(), -vy0) - 2, max(vx, vy0) + 4)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
outfile = "effect_of_gravity_projectile_football.png"
plt.savefig(outfile, dpi=300)
# plt.show()  # Uncomment to display interactively

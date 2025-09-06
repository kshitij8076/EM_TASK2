import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Parameters for vertical launch (theta = 90 degrees)
g = 9.81  # m/s^2
v0 = 20.8  # m/s, chosen to give an apex around ~22 m like the original figure
theta_deg = 90.0
theta = np.deg2rad(theta_deg)

# Time of flight and trajectory
t_flight = 2 * v0 * np.sin(theta) / g
t = np.linspace(0, t_flight, 400)
x = np.zeros_like(t)  # cos(90°) = 0, so no horizontal motion
y = v0 * np.sin(theta) * t - 0.5 * g * t**2
y = np.maximum(y, 0)  # keep above ground

# Apex
t_apex = v0 * np.sin(theta) / g
x_apex = 0.0
y_apex = v0 * np.sin(theta) * t_apex - 0.5 * g * t_apex**2

# Figure
fig, ax = plt.subplots(figsize=(12, 7))

# Projectile path (vertical line)
ax.plot(x, y, color="#C2185B", lw=4, solid_capstyle="round", label="Ball trajectory (θ=90°)")

# Apex marker and label
ax.plot(x_apex, y_apex, "ko", ms=6)
ax.text(x_apex + 0.12, y_apex + 0.7, "Highest point (apex)", ha="left", va="bottom", fontsize=11)

# Initial velocity arrow straight up
ax.annotate("", xy=(0, min(8, y_apex*0.35)), xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', lw=3, color="#3E7CB1"))
ax.text(0.15, min(8.5, y_apex*0.35 + 0.6), r"Initial velocity $v_0$ at angle $\theta = 90^\circ$",
        color="#3E7CB1", fontsize=11, va="bottom")

# Gravity arrow on the right side
x_right = 2.4
ax.annotate("Gravity g pulls downward", xy=(x_right, y_apex*0.65), xytext=(x_right, y_apex*0.95),
            arrowprops=dict(arrowstyle='-|>', color='gray', lw=2), ha='center', va='center', color='gray', rotation=270)

# Ground line
x_min, x_max = -3.0, 3.0
ax.hlines(0, x_min, x_max, colors='darkgreen', lw=3)

# Axes and labels
ax.set_xlim(x_min, x_max)
ax.set_ylim(0, max(26.0, float(y_apex) + 5.0))
ax.set_xlabel("Horizontal distance (m)", fontsize=13)
ax.set_ylabel("Height (m)", fontsize=13)
ax.set_title("Projectile Motion in Cricket: Vertical shot (θ=90°)", fontsize=16, pad=12)

# Explanatory text box
explain_text = (
    "After the bat strike and with θ = 90°, the ball is launched straight up.\n"
    "There is no horizontal motion (range ≈ 0). Gravity slows it to the apex\n"
    "and then brings it back down along the same vertical line."
)
ax.text(x_min + 0.2, y_apex * 0.30, explain_text, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='lightgray'))

# Tidy spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

fig.tight_layout()
fig.savefig("projectile_cricket_slide1_modified.png", dpi=200, bbox_inches="tight")

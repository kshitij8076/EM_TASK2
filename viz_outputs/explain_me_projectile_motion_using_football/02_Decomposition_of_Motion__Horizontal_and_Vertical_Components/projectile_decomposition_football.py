import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc

# Parameters for the football kick
v0 = 25.0           # initial speed (m/s)
angle_deg = 40.0    # launch angle (degrees)
g = 9.81            # gravity (m/s^2)

# Derived quantities
theta = np.deg2rad(angle_deg)
v0x = v0 * np.cos(theta)
v0y = v0 * np.sin(theta)

# Time of flight and trajectory samples
t_flight = 2 * v0y / g
T = np.linspace(0, t_flight, 400)
X = v0x * T
Y = v0y * T - 0.5 * g * T**2

# Useful values
x_max = X[-1]
y_max = Y.max()

# Figure layout
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11
})
fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(2, 3, width_ratios=[2.1, 0.05, 1.0], height_ratios=[1, 1], wspace=0.35, hspace=0.35)
ax_traj = fig.add_subplot(gs[:, 0])
ax_x = fig.add_subplot(gs[0, 2])
ax_y = fig.add_subplot(gs[1, 2])

# Left panel: Trajectory and component decomposition
ax_traj.plot(X, Y, color="black", lw=2, label="Trajectory")

# Ground line (football field ground)
ax_traj.plot([0, x_max * 1.05], [0, 0], color="#2ca02c", lw=3, alpha=0.25)

# Equal time markers to illustrate constant horizontal spacing
n_marks = 9
T_marks = np.linspace(0, t_flight, n_marks)
X_marks = v0x * T_marks
Y_marks = v0y * T_marks - 0.5 * g * T_marks**2
for i, (xm, ym) in enumerate(zip(X_marks, Y_marks)):
    if 0 < i < n_marks - 1:  # skip ends to reduce clutter
        ax_traj.plot([xm, xm], [0, ym], ls=":", lw=0.9, color="gray", alpha=0.8)
    ax_traj.plot(xm, ym, marker="o", ms=3.2, color="black")

# Initial velocity and its components at launch
arrow_len = min(y_max * 0.8, x_max * 0.25)
# Resultant v0 arrow
ax_traj.annotate(
    "",
    xy=(arrow_len * np.cos(theta), arrow_len * np.sin(theta)),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2.2, color="0.2")
)
ax_traj.text(arrow_len * np.cos(theta) * 1.03, arrow_len * np.sin(theta) * 1.03, "v₀", color="0.2", va="bottom")
# Horizontal component v0x
ax_traj.annotate(
    "",
    xy=(arrow_len * np.cos(theta), 0),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2, color="#1f77b4")
)
ax_traj.text(arrow_len * np.cos(theta) * 0.55, arrow_len * 0.045, "v₀x = v₀ cos θ", color="#1f77b4")
# Vertical component v0y
ax_traj.annotate(
    "",
    xy=(0, arrow_len * np.sin(theta)),
    xytext=(0, 0),
    arrowprops=dict(arrowstyle="->", lw=2, color="#d62728")
)
ax_traj.text(arrow_len * 0.05, arrow_len * np.sin(theta) * 0.55, "v₀y = v₀ sin θ", color="#d62728", rotation=90, va="center")

# Angle arc at launch
arc_r = arrow_len * 0.4
arc = Arc((0, 0), width=2 * arc_r, height=2 * arc_r, theta1=0, theta2=angle_deg, color="0.3", lw=1.5)
ax_traj.add_patch(arc)
mid_ang = np.deg2rad(angle_deg / 2)
ax_traj.text(arc_r * np.cos(mid_ang) * 1.05, arc_r * np.sin(mid_ang) * 1.05, "θ", color="0.25")

# Labels and limits for trajectory panel
ax_traj.set_title("Decomposition of Motion: Football as a Projectile")
ax_traj.set_xlabel("Horizontal distance x (m)")
ax_traj.set_ylabel("Vertical height y (m)")
ax_traj.set_xlim(0, x_max * 1.05)
ax_traj.set_ylim(0, y_max * 1.25)
ax_traj.grid(True, which="both", ls=":", alpha=0.4)

# Right-top panel: Horizontal motion x(t) — uniform
ax_x.plot(T, X, color="#1f77b4", lw=2)
ax_x.set_title("Horizontal motion (independent)")
ax_x.set_ylabel("x(t) (m)")
ax_x.set_xlabel("time t (s)")
ax_x.grid(True, ls=":", alpha=0.5)
ax_x.text(0.03 * t_flight, 0.85 * x_max, f"x(t) = v₀x·t\n(vₓ constant = {v0x:.1f} m/s)", color="#1f77b4",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#1f77b4", alpha=0.9))

# Right-bottom panel: Vertical motion y(t) — accelerated by gravity
ax_y.plot(T, Y, color="#d62728", lw=2)
# Peak height annotation
t_peak = v0y / g
y_peak = v0y * t_peak - 0.5 * g * t_peak**2
ax_y.axvline(t_peak, ls=":", color="gray", lw=1)
ax_y.axhline(y_peak, ls=":", color="gray", lw=1)
ax_y.plot(t_peak, y_peak, 'o', color="#d62728")
ax_y.text(t_peak * 1.02, y_peak * 1.02, "apex", color="#d62728")
ax_y.set_title("Vertical motion (independent)")
ax_y.set_ylabel("y(t) (m)")
ax_y.set_xlabel("time t (s)")
ax_y.grid(True, ls=":", alpha=0.5)
ax_y.text(0.03 * t_flight, 0.85 * y_max, "y(t) = v₀y·t − ½ g t²\n(vᵧ decreases linearly)", color="#d62728",
          bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", alpha=0.9))

# Overall caption-like annotation on the left panel to link concepts
ax_traj.text(x_max * 0.03, y_max * 1.15,
             "Break the motion into two independent parts:\n"
             "• Horizontal: constant speed (spacing between dots is uniform).\n"
             "• Vertical: accelerated by gravity (up, stop at apex, then down).",
             fontsize=10)

# Save figure
out_file = "projectile_decomposition_football.png"
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved figure to {out_file}")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# Create figure with two panels
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# -------------------------
# Left panel: Postulate 1
# -------------------------
ax = axes[0]
ax.set_title('Postulate 1: Same laws in all inertial frames', fontsize=11)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Draw two identical "labs" with a light clock in each
lab_width, lab_height = 4.8, 3.0
labs = [
    {"xy": (0.8, 6.2), "label": "Frame S (lab at rest)"},
    {"xy": (0.8, 2.2), "label": "Frame S' (lab moving at constant v)"}
]

for i, lab in enumerate(labs):
    x0, y0 = lab["xy"]
    # Lab body
    rect = Rectangle((x0, y0), lab_width, lab_height, fill=False, lw=1.8, ec='black')
    ax.add_patch(rect)

    # Light clock mirrors (two horizontal plates)
    margin_x = 0.8
    y_bottom = y0 + 0.6
    y_top = y0 + lab_height - 0.6
    ax.plot([x0 + margin_x, x0 + lab_width - margin_x], [y_bottom, y_bottom], color='#444', lw=3)
    ax.plot([x0 + margin_x, x0 + lab_width - margin_x], [y_top, y_top], color='#444', lw=3)

    # Light path vertical (bouncing)
    x_center = x0 + lab_width / 2
    ax.plot([x_center, x_center], [y_bottom, y_top], color='#1f77b4', lw=2)
    # Bounce markers
    ax.plot(x_center, y_bottom, 'o', color='#1f77b4', ms=5)
    ax.plot(x_center, y_top, 'o', color='#1f77b4', ms=5)

    # Labels above each lab
    ax.text(x0 + lab_width/2, y0 + lab_height + 0.5, lab["label"], ha='center', va='bottom', fontsize=10)

# Velocity arrow for moving lab
x0, y0 = labs[1]["xy"]
ax.annotate('v', xy=(x0 + lab_width + 0.4, y0 + lab_height/2), xytext=(x0 + lab_width + 1.2, y0 + lab_height/2),
            arrowprops=dict(arrowstyle='->', lw=1.8, color='crimson'), color='crimson', fontsize=11, va='center')

# Caption text
ax.text(5.0, 0.9, 'Identical experiment (a light clock) behaves the same in any inertial frame\n— there is no preferred frame.',
        ha='center', va='bottom', fontsize=9)

# -------------------------
# Right panel: Postulate 2
# -------------------------
ax2 = axes[1]
ax2.set_title('Postulate 2: Speed of light is constant (c)', fontsize=11)

# Spacetime diagram settings (use units with c = 1)
xmax = 5.0
tmax = 5.0
ax2.set_xlim(-xmax, xmax)
ax2.set_ylim(-0.2, tmax)
ax2.set_aspect('equal', adjustable='box')

# Axes
ax2.axhline(0, color='black', lw=1)
ax2.axvline(0, color='black', lw=1)
ax2.text(xmax - 0.3, -0.1, 'x', ha='right', va='top', fontsize=10)
ax2.text(0.15, tmax - 0.1, 'ct', ha='left', va='top', fontsize=10)

# Light lines at 45 degrees (ct = ±x)
x = np.linspace(-xmax, xmax, 200)
ax2.plot(x, np.abs(x), color='#f39c12', lw=2, label='light (c)')  # plot both branches via |x|
# Annotate light lines
ax2.text(2.2, 2.2, 'light', color='#f39c12', fontsize=9, rotation=45, ha='left', va='bottom')
ax2.text(-2.2, 2.2, 'light', color='#f39c12', fontsize=9, rotation=-45, ha='right', va='bottom')

# Primed axes for a moving frame S' with velocity v = 0.6 c
v = 0.6
# t' axis: x = v t -> t = x / v
x_tp = np.linspace(0, v * tmax, 100)
t_tp = x_tp / v  # but since x_tp = v*t, t_tp recovers 0..tmax
# A cleaner param: param by t
t_vals = np.linspace(0, tmax, 200)
x_tprime = v * t_vals
ax2.plot(x_tprime, t_vals, color='#6a3d9a', lw=2)
ax2.text(x_tprime[-1] + 0.1, t_vals[-1], "t'", color='#6a3d9a', fontsize=10, ha='left', va='center')

# x' axis: t = v x
x_vals = np.linspace(-xmax, xmax, 200)
t_xprime = v * x_vals
ax2.plot(x_vals, t_xprime, color='#6a3d9a', lw=2)
# Place label near positive x side of x' axis
ax2.text(xmax*0.7, v * xmax*0.7 + 0.05, "x'", color='#6a3d9a', fontsize=10, ha='left', va='bottom')

# Origin event
ax2.plot(0, 0, 'ko', ms=4)
ax2.text(0.08, 0.08, 'emission', fontsize=8, ha='left', va='bottom')

# Decorative: draw small light arrows from origin along light lines
for sgn in (+1, -1):
    xs = np.linspace(0, 2.2, 20)
    ts = sgn * xs
    ts = np.abs(ts)
# Ticks and labels
ax2.set_xlabel('space (x)')
ax2.set_ylabel('time (ct)')
ax2.set_xticks([-4, -2, 0, 2, 4])
ax2.set_yticks([0, 1, 2, 3, 4, 5])
ax2.grid(alpha=0.15, linestyle='--')

# Explanatory annotation
ax2.text(0.02, -0.16, 'Light rays always follow 45° lines in spacetime (|x| = ct)\nfor every inertial frame — the measured speed is c.',
         transform=ax2.transAxes, fontsize=9, ha='left', va='top')

fig.tight_layout()
fig.savefig('special_relativity_postulates.png', dpi=300, bbox_inches='tight')
print('Saved figure: special_relativity_postulates.png')

import numpy as np
import matplotlib.pyplot as plt

# Configuration
c = 1.0  # Use units where c = 1 so ct and x share units
beta = 0.6  # v/c
gamma = 1.0 / np.sqrt(1.0 - beta**2)

# Plot limits
x_min, x_max = -6.0, 6.0
ct_min, ct_max = -6.0, 6.0

fig, ax = plt.subplots(figsize=(8, 8))

# Axes: x (horizontal), ct (vertical)
ax.plot([x_min, x_max], [0, 0], color='black', linewidth=1.8)  # x-axis
ax.plot([0, 0], [ct_min, ct_max], color='black', linewidth=1.8)  # ct-axis

# Light cone (ct = ±x)
x_line = np.linspace(x_min, x_max, 500)
ax.plot(x_line, x_line, color='goldenrod', linestyle='--', linewidth=1.5, label='Light cone ct=±x')
ax.plot(x_line, -x_line, color='goldenrod', linestyle='--', linewidth=1.5)

# Primed axes for a boosted frame with v = beta * c
# ct' axis: x = beta * ct -> param by ct
ct_vals = np.linspace(ct_min, ct_max, 500)
x_ctprime = beta * ct_vals
ct_ctprime = ct_vals
ctp_line, = ax.plot(x_ctprime, ct_ctprime, color='#1f77b4', linewidth=2.2, label="ct' axis (worldline x=vt)")

# x' axis: ct = beta * x -> param by x
x_vals = np.linspace(x_min, x_max, 500)
ct_xprime = beta * x_vals
xp_line, = ax.plot(x_vals, ct_xprime, color='#d62728', linewidth=2.2, label="x' axis (t'=0)")

# Draw a few lines of constant ct' and x' to show Lorentz grid
# ct' = const -> ct = beta*x + const/gamma
ctprime_consts = [-3.0, -1.5, 1.5, 3.0]
first_ctp = True
for k in ctprime_consts:
    y = beta * x_vals + k / gamma
    lbl = "lines of simultaneity (t' const)" if first_ctp else None
    ax.plot(x_vals, y, color='#1f77b4', alpha=0.35, linestyle='-', linewidth=1.2, label=lbl)
    first_ctp = False

# x' = const -> x = beta*ct + const/gamma
xprime_consts = [-3.0, -1.5, 1.5, 3.0]
first_xp = True
for s in xprime_consts:
    x_line2 = beta * ct_vals + s / gamma
    lbl = "x' = const lines" if first_xp else None
    ax.plot(x_line2, ct_vals, color='#d62728', alpha=0.35, linestyle='-', linewidth=1.2, label=lbl)
    first_xp = False

# Example event and its transformed coordinates
x_e, ct_e = 3.0, 4.0
x_e_prime = gamma * (x_e - beta * ct_e)
ct_e_prime = gamma * (ct_e - beta * x_e)
ax.scatter([x_e], [ct_e], color='purple', s=60, zorder=5, label='Event E')
ax.annotate(
    f"E\n(x, ct) = ({x_e:.0f}, {ct_e:.0f})\n(x', ct') = ({x_e_prime:.2f}, {ct_e_prime:.2f})",
    xy=(x_e, ct_e), xytext=(x_e + 0.4, ct_e + 0.6),
    fontsize=11, color='purple', bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='purple', alpha=0.85))

# Labels for axes
ax.text(x_max - 0.4, -0.6, 'x', fontsize=13)
ax.text(0.3, ct_max - 0.6, 'ct', fontsize=13)
ax.text(beta * (ct_max - 0.4), ct_max - 0.3, "ct'", color='#1f77b4', fontsize=12)
ax.text(x_max - 1.0, beta * (x_max - 1.0) + 0.2, "x'", color='#d62728', fontsize=12)

# Informational textbox with equations and parameters
eq_text = (
    "Lorentz transformations (c=1):\n"
    "x' = γ (x − β ct)\n"
    "ct' = γ (ct − β x)\n"
    f"β = v/c = {beta:.1f},  γ = {gamma:.2f}"
)
ax.text(
    x_min + 0.3, ct_max - 1.9, eq_text, fontsize=11,
    bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='gray', alpha=0.9))

# Aesthetic tweaks
ax.set_xlim(x_min, x_max)
ax.set_ylim(ct_min, ct_max)
ax.set_aspect('equal', adjustable='box')
ax.set_xlabel('x (units where c=1)')
ax.set_ylabel('ct (units where c=1)')
ax.grid(alpha=0.15)

# Clean up spines
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Legend
handles, labels = ax.get_legend_handles_labels()
# Remove potential duplicate labels while preserving order
seen = set()
uniq_handles, uniq_labels = [], []
for h, l in zip(handles, labels):
    if l not in seen and l:
        uniq_handles.append(h)
        uniq_labels.append(l)
        seen.add(l)
ax.legend(uniq_handles, uniq_labels, loc='lower right', framealpha=0.95)

plt.tight_layout()

# Save figure
out_name = 'lorentz_transformations_minkowski.png'
plt.savefig(out_name, dpi=300)
print(f'Saved figure to {out_name}')

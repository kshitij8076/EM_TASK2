import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Reproducibility
np.random.seed(42)

# Parameters for two tops (top-down view)
R = 1.0  # nominal radius
N = 60   # number of point-mass markers to visualize distribution
M_total = 1.0  # total mass (kg)
m_i = M_total / N

# Generate mass distributions
# Compact: masses concentrated near center (0 to 0.3 R)
rad_compact = np.random.uniform(0.0, 0.3 * R, size=N)
ang_compact = np.random.uniform(0, 2*np.pi, size=N)
x_compact = -1.6 + rad_compact * np.cos(ang_compact)  # shift left
y_compact = 0.0 + rad_compact * np.sin(ang_compact)

# Rim-heavy: masses concentrated near rim (~0.9 R with small jitter)
rad_rim = np.clip(np.random.normal(0.9 * R, 0.03 * R, size=N), 0.7 * R, 1.0 * R)
ang_rim = np.random.uniform(0, 2*np.pi, size=N)
x_rim = 1.6 + rad_rim * np.cos(ang_rim)  # shift right
y_rim = 0.0 + rad_rim * np.sin(ang_rim)

# Compute moment of inertia approximations: I = sum m r^2, with r about spin axis (z-axis)
I_compact = np.sum(m_i * (rad_compact**2))
I_rim = np.sum(m_i * (rad_rim**2))

# Dynamics under same applied torque (spin-up)
tau = 0.2  # N·m
T_spinup = 5.0
T = np.linspace(0, T_spinup, 300)
omega_compact_up = (tau / I_compact) * T
omega_rim_up = (tau / I_rim) * T

# Dynamics under same friction torque (spin-down)
omega0 = 50.0  # rad/s initial spin
tau_f = 0.1    # N·m opposing torque
T_decay = np.linspace(0, 12.0, 300)
omega_compact_down = np.maximum(omega0 - (tau_f / I_compact) * T_decay, 0)
omega_rim_down = np.maximum(omega0 - (tau_f / I_rim) * T_decay, 0)

# Colors
c_compact = 'tab:blue'
c_rim = 'tab:orange'

# Create figure
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Panel A: Mass distributions
ax = axs[0, 0]
# Draw the two top footprints (circles)
for cx in [-1.6, 1.6]:
    circ = Circle((cx, 0), R, edgecolor='0.3', facecolor='none', linewidth=2)
    ax.add_patch(circ)
    # Mark the spin axis (center)
    ax.plot(cx, 0, marker='o', color='k', markersize=5)

# Scatter mass markers
ax.scatter(x_compact, y_compact, s=18, color=c_compact, alpha=0.8, label='Compact mass (low r)')
ax.scatter(x_rim, y_rim, s=18, color=c_rim, alpha=0.8, label='Rim-heavy mass (high r)')

# Labels and annotations
ax.set_aspect('equal', 'box')
ax.set_xlim(-3.2, 3.2)
ax.set_ylim(-1.4, 1.4)
ax.axis('off')
ax.text(-1.6, -1.25, 'Compact Top (low I)', ha='center', va='top', fontsize=11, color=c_compact)
ax.text(1.6, -1.25, 'Rim-Heavy Top (high I)', ha='center', va='top', fontsize=11, color=c_rim)
ax.set_title('Mass distribution relative to the spin axis (top-down view)')

# Panel B: Bar chart of I
ax = axs[0, 1]
labels = ['Compact', 'Rim-Heavy']
Is = [I_compact, I_rim]
colors = [c_compact, c_rim]
bars = ax.bar(labels, Is, color=colors, edgecolor='0.3')
ax.set_ylabel('Moment of Inertia I (kg·m²)')
ax.set_title('Moment of inertia: I = Σ m r² (about the vertical spin axis)')
# Annotate values
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2, h + 0.02*max(Is), f"{h:.3f}", ha='center', va='bottom', fontsize=10)
ax.set_ylim(0, max(Is) * 1.25)

# Panel C: Spin-up under same torque
ax = axs[1, 0]
ax.plot(T, omega_compact_up, color=c_compact, lw=2, label=f'Compact (I={I_compact:.3f})')
ax.plot(T, omega_rim_up, color=c_rim, lw=2, label=f'Rim-Heavy (I={I_rim:.3f})')
ax.set_xlabel('Time t (s)')
ax.set_ylabel('Angular speed ω (rad/s)')
ax.set_title('Same torque τ ⇒ different angular acceleration α = τ/I')
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

# Panel D: Spin-down under same friction torque
ax = axs[1, 1]
ax.plot(T_decay, omega_compact_down, color=c_compact, lw=2, label='Compact')
ax.plot(T_decay, omega_rim_down, color=c_rim, lw=2, label='Rim-Heavy')
ax.set_xlabel('Time t (s)')
ax.set_ylabel('Angular speed ω (rad/s)')
ax.set_title('Same friction torque ⇒ higher I slows down more slowly')
ax.legend(frameon=False)
ax.grid(True, alpha=0.3)

# Overall title and explanatory note
fig.suptitle('Moment of Inertia (I) and Mass Distribution in a Spinning Top', fontsize=16, y=0.98)
fig.text(0.5, 0.02, 'Mass farther from the spin axis increases I, making the top harder to speed up or slow down (but often more stable).',
         ha='center', va='center', fontsize=11)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])

# Save figure
out_name = 'moment_of_inertia_spinning_top.png'
fig.savefig(out_name, dpi=300, bbox_inches='tight')
print(f'Saved figure to {out_name}')

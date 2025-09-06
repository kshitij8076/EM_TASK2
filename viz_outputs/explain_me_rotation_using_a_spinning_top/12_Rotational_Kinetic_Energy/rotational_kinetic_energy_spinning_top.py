import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Circle


def draw_top(ax):
    # Define half-profile of a stylized spinning top (right side), then mirror for left side
    right_profile = [
        (0.00, -1.20),  # bottom tip
        (0.30, -0.95),
        (0.55, -0.60),
        (0.75, -0.20),
        (0.85,  0.10),  # widest belly
        (0.60,  0.50),
        (0.40,  0.80),
        (0.22,  1.00),
        (0.12,  1.10),  # handle start
        (0.10,  1.20)   # handle tip
    ]
    left_profile = [(-x, y) for (x, y) in right_profile[::-1]]
    outline = left_profile + right_profile

    body = Polygon(outline, closed=True, facecolor="#f3b24d", edgecolor="#7a5a2e", linewidth=2, joinstyle='round')
    ax.add_patch(body)

    # Central symmetry axis (dashed)
    ax.plot([0, 0], [-1.35, 1.30], linestyle='--', color='gray', linewidth=1)

    # Contact point at the tip
    tip_circle = Circle((0, -1.22), 0.03, color='dimgray')
    ax.add_patch(tip_circle)

    # Angular velocity vector ω along the axis
    omega_arrow = FancyArrowPatch(
        posA=(0.18, -0.20), posB=(0.18, 1.05),
        arrowstyle='-|>', mutation_scale=14, linewidth=2.2, color='tab:red')
    ax.add_patch(omega_arrow)
    ax.text(0.26, 0.95, r"$\omega$", color='tab:red', fontsize=12, va='bottom')

    # Curved arrow indicating rotation about the axis
    rot_arrow = FancyArrowPatch(
        (0.70, 0.10), (-0.70, 0.10),
        connectionstyle="arc3,rad=0.8", arrowstyle='-|>', mutation_scale=14,
        linewidth=2, color='tab:blue', alpha=0.85)
    ax.add_patch(rot_arrow)
    ax.text(0, 0.55, "rotation", color='tab:blue', fontsize=10, ha='center')

    # Energy equation
    ax.text(-1.30, 1.28, r"$E_{rot}=\tfrac{1}{2}\,I\,\omega^2$", fontsize=13, bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='gray', alpha=0.9))

    # Note on moment of inertia
    ax.text(-1.32, -1.28, "I increases with mass\n& spread-out mass", fontsize=9, color='black', ha='left', va='bottom', bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')


def energy_decay_plot(ax):
    # Parameters
    t = np.linspace(0, 10, 600)
    k = 0.30  # viscous-like damping for angular speed (s^-1)

    I_light = 2e-3   # kg m^2
    I_heavy = 6e-3   # kg m^2 (heavier -> larger I)
    w0_base = 100.0  # rad/s
    w0_fast = 200.0  # rad/s (faster spin)

    # Angular speed decay (simplified model)
    w_light = w0_base * np.exp(-k * t)
    w_heavy = w0_base * np.exp(-k * t)
    w_fast  = w0_fast * np.exp(-k * t)

    # Rotational kinetic energy E = 1/2 I w^2
    E_light = 0.5 * I_light * w_light**2
    E_heavy = 0.5 * I_heavy * w_heavy**2
    E_fast  = 0.5 * I_light * w_fast**2

    # Plot
    ax.plot(t, E_light, label=r"Light top: $I=2\times10^{-3}$, $\omega_0=100$", color='tab:blue', linewidth=2.5)
    ax.plot(t, E_heavy, label=r"Heavy top: $I=6\times10^{-3}$, $\omega_0=100$", color='tab:orange', linewidth=2.5)
    ax.plot(t, E_fast,  label=r"Faster spin: $I=2\times10^{-3}$, $\omega_0=200$", color='tab:green', linewidth=2.5)

    ax.set_xlabel('Time t (s)')
    ax.set_ylabel(r'Rotational kinetic energy $E_{rot}$ (J)')
    ax.set_title('Energy decreases as the top slows due to friction')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Place legend with compact font
    leg = ax.legend(title='Scenarios', frameon=True, fontsize=9)
    if leg is not None:
        leg.get_title().set_fontsize(10)

    # Highlight quadratic dependence on speed
    ax.annotate(r"$E_{rot} \propto \omega^2$", xy=(1.5, E_fast[np.searchsorted(t, 1.5)]),
                xytext=(4.0, E_fast[np.searchsorted(t, 1.5)]*1.4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=11, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))

    # Annotation about friction
    ax.annotate("friction slows spin\n→ energy decays", xy=(7.0, E_light[np.searchsorted(t, 7.0)]),
                xytext=(5.8, max(E_light)*0.55),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                fontsize=10, bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9))


def make_figure():
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'figure.dpi': 150
    })

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={'width_ratios': [1, 1.2]})

    draw_top(axes[0])
    axes[0].set_title('Spinning top and rotation axis')

    energy_decay_plot(axes[1])

    fig.suptitle('Rotational Kinetic Energy of a Spinning Top', y=0.98, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    outname = 'rotational_kinetic_energy_spinning_top.png'
    plt.savefig(outname, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved figure to {outname}")


if __name__ == '__main__':
    make_figure()

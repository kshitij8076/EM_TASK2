import numpy as np
import matplotlib.pyplot as plt

def format_joules(E):
    units = [("J", 1), ("kJ", 1e3), ("MJ", 1e6), ("GJ", 1e9), ("TJ", 1e12), ("PJ", 1e15), ("EJ", 1e18)]
    for i in range(len(units)-1, -1, -1):
        unit, scale = units[i]
        if E >= scale:
            return f"{E/scale:.3g} {unit}"
    return f"{E:.3g} J"

def format_tnt(E):
    ton_tnt_j = 4.184e9  # joules per ton of TNT
    tons = E / ton_tnt_j
    if tons < 1e3:
        return f"{tons:.3g} tons TNT"
    elif tons < 1e6:
        return f"{tons/1e3:.3g} kilotons TNT"
    else:
        return f"{tons/1e6:.3g} megatons TNT"

# Constants
c = 299_792_458.0  # m/s
c2 = c**2

# Data for the line E = m c^2 over a broad mass range (1 mg to 1 kg)
masses = np.logspace(-6, 0, 400)  # kg
energies = c2 * masses            # J

# Example points
points = [("1 mg", 1e-6), ("1 g", 1e-3), ("1 kg", 1.0)]

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(masses, energies, color='C0', lw=2, label=r"$E = m c^2$")
ax.scatter([m for _, m in points], [c2 * m for _, m in points], color='C3', s=60, zorder=5, label='Example masses')

# Annotate example points with energies and TNT equivalents
for name, m in points:
    E = c2 * m
    label = f"{name}\n{format_joules(E)}\n\u2248 {format_tnt(E)}"
    # Choose annotation offsets for clarity
    if np.isclose(m, 1e-6):
        xytext = (12, 14)
        ha, va = 'left', 'bottom'
    elif np.isclose(m, 1e-3):
        xytext = (12, -22)
        ha, va = 'left', 'top'
    else:  # 1 kg
        xytext = (-90, -10)
        ha, va = 'right', 'top'
    ax.annotate(label,
                xy=(m, E), xycoords='data',
                xytext=xytext, textcoords='offset points',
                ha=ha, va=va,
                fontsize=10,
                arrowprops=dict(arrowstyle='->', lw=1, color='0.3'))

# Axes formatting
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Mass m (kg)', fontsize=12)
ax.set_ylabel('Energy E (J)', fontsize=12)
ax.set_title('Mass–Energy Equivalence: $E = m c^2$', fontsize=14)
ax.grid(True, which='both', ls='--', alpha=0.3)
ax.legend(loc='lower right', frameon=False)

# Explanatory textbox
textbox = (r"Proportionality on log–log axes: slope = 1 (E \u221d m)\n"
           r"Speed of light: $c = 2.998\times10^8\,\mathrm{m/s}$")
ax.text(0.02, 0.04, textbox, transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.8'))

plt.tight_layout()
plt.savefig('mass_energy_equivalence.png', dpi=300, bbox_inches='tight')

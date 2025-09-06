import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

# Constants
c = 299792458.0  # m/s
G = 6.67430e-11  # m^3 kg^-1 s^-2
M_earth = 5.972e24  # kg
mu_earth = G * M_earth
R_earth = 6371e3  # m

# Figure setup
plt.rcParams.update({
    'figure.figsize': (12, 7.5),
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})
fig, axes = plt.subplots(2, 3)

# Panel A: GPS clock corrections (SR vs GR)
ax = axes[0, 0]
alt_gps = 20200e3  # m
r_gps = R_earth + alt_gps
v_gps = np.sqrt(mu_earth / r_gps)
# Special relativity rate (moving clock runs slower)
gamma_gps = 1.0 / np.sqrt(1.0 - (v_gps / c) ** 2)
sr_rate_diff = (1.0 / gamma_gps - 1.0)  # fractional per second
# General relativity rate (higher altitude runs faster)
phi_ground = np.sqrt(1.0 - 2.0 * mu_earth / (R_earth * c ** 2))
phi_sat = np.sqrt(1.0 - 2.0 * mu_earth / (r_gps * c ** 2))
gr_rate_diff = (phi_sat / phi_ground - 1.0)
seconds_per_day = 86400.0
sr_us_per_day = sr_rate_diff * seconds_per_day * 1e6
gr_us_per_day = gr_rate_diff * seconds_per_day * 1e6
net_us_per_day = sr_us_per_day + gr_us_per_day
vals = [sr_us_per_day, gr_us_per_day, net_us_per_day]
labels = ['SR (motion)', 'GR (gravity)', 'Net']
colors = ['#1f77b4', '#2ca02c', '#9467bd']
ax.bar(labels, vals, color=colors)
ax.axhline(0, color='k', linewidth=0.8)
for i, v in enumerate(vals):
    ax.text(i, v + (0.02 if v >= 0 else -0.02) * (abs(max(vals)) + 5), f"{v:.1f} µs/day",
            ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)
ax.set_ylabel('Clock offset (microseconds per day)')
ax.set_title('A) GPS atomic clocks require SR+GR corrections')
ax.grid(True, axis='y', alpha=0.3)

# Panel B: Cosmic-ray muons survival vs altitude (with/without time dilation)
ax = axes[0, 1]
h_km = np.linspace(0, 15, 400)
h_m = h_km * 1000.0
tau_mu = 2.1969811e-6  # s (proper lifetime)
# Assume v ≈ c
gamma_mu = 15.0  # ~1.6 GeV muons
def survival(hm, gamma):
    return np.exp(-hm / (gamma * c * tau_mu))
F_rel = survival(h_m, gamma_mu)
F_no = survival(h_m, 1.0)
ax.plot(h_km, F_no, label='No relativity', color='#d62728', lw=2, ls='--')
ax.plot(h_km, F_rel, label=f'With relativity (γ≈{gamma_mu:.0f})', color='#1f77b4', lw=2)
ax.set_xlabel('Altitude traveled (km)')
ax.set_ylabel('Survival fraction to ground')
ax.set_ylim(0, 1.05)
ax.set_title('B) Cosmic-ray muons reach Earth due to time dilation')
ax.grid(True, alpha=0.3)
ax.legend()

# Panel C: Particle accelerators — energy vs speed (electron)
ax = axes[0, 2]
beta = np.linspace(1e-6, 0.9999999, 1000)
gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
mc2_e_MeV = 0.511
KE_MeV = (gamma - 1.0) * mc2_e_MeV
ax.plot(beta, KE_MeV, color='#ff7f0e', lw=2)
ax.set_xlabel('Speed as a fraction of c (β)')
ax.set_ylabel('Kinetic energy (MeV)')
ax.set_title('C) Accelerators: energy skyrockets as v → c (E=γmc²)')
ax.grid(True, alpha=0.3)
# Annotate high-energy regime
ax.annotate('Ultra-relativistic regime\n(β ≈ 1)', xy=(0.995, KE_MeV[np.searchsorted(beta, 0.995)]),
            xytext=(0.8, KE_MeV.max()*0.7),
            arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

# Panel D: Atomic clocks in flight — 10-hour run vs speed
ax = axes[1, 0]
speeds = np.linspace(0, 1000, 400)  # m/s
T = 10.0 * 3600.0  # 10 hours
# Time difference for moving clock relative to rest: Δt = (1/γ - 1) * T
Gamma = 1.0 / np.sqrt(1.0 - (speeds / c) ** 2)
Delta_ns = (1.0 / Gamma - 1.0) * T * 1e9  # nanoseconds (negative)
ax.plot(speeds, Delta_ns, color='#1f77b4', lw=2)
# Mark a typical jet (~250 m/s)
jet_v = 250.0
jet_gamma = 1.0 / np.sqrt(1.0 - (jet_v / c) ** 2)
jet_ns = (1.0 / jet_gamma - 1.0) * T * 1e9
ax.scatter([jet_v], [jet_ns], color='#d62728', zorder=5)
ax.annotate(f'Jet ~250 m/s\n≈ {jet_ns:.1f} ns (SR)', xy=(jet_v, jet_ns), xytext=(450, jet_ns*2.5 if jet_ns != 0 else -40),
            arrowprops=dict(arrowstyle='->'), fontsize=9)
ax.set_xlabel('Speed (m/s)')
ax.set_ylabel('Clock offset over 10 h (ns)')
ax.set_title('D) Atomic clocks on jets directly measure SR time dilation')
ax.grid(True, alpha=0.3)

# Panel E: Gravitational lensing schematic
ax = axes[1, 1]
ax.set_aspect('equal')
ax.axis('off')
# Positions
obs = np.array([3.5, 0.0])
lens = np.array([0.0, 0.0])
src1 = np.array([-3.0, 1.5])
src2 = np.array([-3.0, -1.5])
# Draw lens mass
lens_circle = Circle(lens, 0.35, color='#9467bd', alpha=0.4)
ax.add_patch(lens_circle)
ax.text(lens[0], lens[1]-0.65, 'Lens (mass)', ha='center', va='top', fontsize=9)
# Observer
ax.plot(obs[0], obs[1], marker='o', color='black')
ax.text(obs[0]+0.08, obs[1], 'Observer', va='center', fontsize=9)
# Sources
ax.plot(src1[0], src1[1], marker='*', color='#ff7f0e', ms=10)
ax.text(src1[0]-0.05, src1[1]+0.35, 'Source', ha='right', fontsize=9)
ax.plot(src2[0], src2[1], marker='*', color='#ff7f0e', ms=10)
ax.text(src2[0]-0.05, src2[1]-0.35, 'Source', ha='right', va='top', fontsize=9)
# Straight (unlensed) paths as dashed lines
ax.plot([src1[0], obs[0]], [src1[1], obs[1]], ls='--', color='gray', lw=1)
ax.plot([src2[0], obs[0]], [src2[1], obs[1]], ls='--', color='gray', lw=1)
# Bent light paths using curved arrows
arrow_kwargs = dict(arrowstyle='->', lw=2, color='#1f77b4', shrinkA=0, shrinkB=4)
arc1 = FancyArrowPatch((src1[0], src1[1]), (obs[0], obs[1]),
                       connectionstyle='arc3,rad=-0.25', **arrow_kwargs)
arc2 = FancyArrowPatch((src2[0], src2[1]), (obs[0], obs[1]),
                       connectionstyle='arc3,rad=0.25', **arrow_kwargs)
ax.add_patch(arc1)
ax.add_patch(arc2)
# Approximate Einstein ring (dashed)
ring = Circle(lens, 1.2, edgecolor='k', facecolor='none', ls='--', lw=1)
ax.add_patch(ring)
ax.text(lens[0]+1.25, lens[1]+0.05, 'Einstein ring', fontsize=8, va='bottom')
ax.set_xlim(-4, 4.2)
ax.set_ylim(-2.5, 2.5)
ax.set_title('E) Gravitational lensing: mass bends light')

# Panel F: Summary text
ax = axes[1, 2]
ax.axis('off')
summary = (
    "F) Evidence and Applications:\n"
    "• GPS: SR (−) and GR (+) shifts must be corrected for accurate positioning.\n"
    "• Atomic clocks: flights and satellites directly confirm time dilation.\n"
    "• Particle accelerators: E=γmc²; speed saturates near c while energy soars.\n"
    "• Cosmic rays: muons survive to sea level thanks to relativistic lifetimes.\n"
    "• Astronomy: gravitational lensing maps mass via light bending."
)
ax.text(0.0, 0.96, summary, va='top', ha='left', fontsize=10, family='DejaVu Sans Mono')

plt.tight_layout()
outfile = 'relativity_evidence_and_applications.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f'Saved figure to {outfile}')

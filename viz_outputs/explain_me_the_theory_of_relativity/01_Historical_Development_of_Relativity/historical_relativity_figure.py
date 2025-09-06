import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Configure style for readability
plt.rcParams.update({
    'figure.figsize': (12, 7),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

fig = plt.figure()
gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.35, wspace=0.25)

# Top: Timeline spanning both columns
ax_t = fig.add_subplot(gs[0, :])
ax_t.set_xlim(1600, 2005)
ax_t.set_ylim(0, 1)
ax_t.axis('off')

# Draw base timeline
ax_t.hlines(0.5, 1600, 2000, colors='black', linewidth=2)
for yr in range(1600, 2001, 50):
    ax_t.vlines(yr, 0.48, 0.52, colors='gray', linewidth=0.8, alpha=0.4)
ax_t.text(1600, 0.92, 'Historical Development of Relativity', fontsize=14, fontweight='bold', va='top')
ax_t.text(1600, 0.86, 'From classical mechanics to modern relativity and technology', fontsize=10, va='top', color='dimgray')

# Events on the timeline
events = [
    {
        'year': 1632,
        'title': 'Galileo (1632)',
        'desc': 'Relativity principle in mechanics: laws are the same in uniform motion.',
        'pos': 'up'
    },
    {
        'year': 1687,
        'title': 'Newton (1687)',
        'desc': 'Principia: absolute space & time; highly successful at low speeds.',
        'pos': 'down'
    },
    {
        'year': 1887,
        'title': 'Michelson–Morley (1887)',
        'desc': 'Null result for ether drift: c shows no directional change.',
        'pos': 'up'
    },
    {
        'year': 1905,
        'title': 'Einstein (1905)',
        'desc': 'Special Relativity: c is constant; simultaneity is relative.',
        'pos': 'down'
    },
    {
        'year': 1915,
        'title': 'Einstein (1915)',
        'desc': 'General Relativity: gravity = spacetime curvature.',
        'pos': 'up'
    },
    {
        'year': 1995,
        'title': 'GPS era (~1995)',
        'desc': 'Relativistic time corrections enable precise navigation.',
        'pos': 'down'
    }
]

for i, ev in enumerate(events):
    y0 = 0.5
    ytext = 0.82 if ev['pos'] == 'up' else 0.18
    va = 'bottom' if ev['pos'] == 'up' else 'top'
    arrow_y = 0.67 if ev['pos'] == 'up' else 0.33
    # Node
    ax_t.plot(ev['year'], y0, 'o', color='tab:blue', markersize=6)
    # Stem
    ax_t.vlines(ev['year'], y0, arrow_y, colors='tab:blue', linewidth=1.5)
    # Text box
    txt = f"{ev['title']}\n{ev['desc']}"
    ax_t.annotate(
        txt,
        xy=(ev['year'], arrow_y),
        xytext=(ev['year'], ytext),
        ha='center', va=va,
        textcoords='data',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='lightgray', alpha=0.95),
        arrowprops=dict(arrowstyle='-|>', color='gray', shrinkA=0, shrinkB=4, lw=0.8)
    )

# Annotative hints to subplots
ax_t.text(1887, 0.95, 'See lower-left: null result', color='tab:red', fontsize=9, ha='center')
ax_t.text(1905, 0.05, 'See lower-right: time dilation', color='tab:green', fontsize=9, ha='center')

# Bottom-left: Michelson–Morley expected vs observed fringe shift (cartoon)
ax_mm = fig.add_subplot(gs[1, 0])
angles = np.linspace(0, 360, 361)
pred = np.cos(np.deg2rad(2 * angles))  # normalized expected ether signal
observed = np.zeros_like(angles)  # null result

ax_mm.fill_between(angles, -0.15, 0.15, color='lightgray', alpha=0.5, label='Noise band (~null)')
ax_mm.plot(angles, pred, color='tab:red', lw=2, label='Predicted (ether hypothesis)')
ax_mm.plot(angles, observed, color='black', lw=2, label='Observed (near zero)')
ax_mm.set_title('Michelson–Morley (1887): No ether drift detected')
ax_mm.set_xlabel('Apparatus orientation (degrees)')
ax_mm.set_ylabel('Normalized fringe shift')
ax_mm.set_xlim(0, 360)
ax_mm.set_ylim(-1.2, 1.2)
ax_mm.set_xticks([0, 60, 120, 180, 240, 300, 360])
ax_mm.legend(loc='upper right', frameon=True)
ax_mm.text(5, 1.02, 'Classical expectation: orientation-dependent signal', color='tab:red', fontsize=9)
ax_mm.text(5, -1.1, 'Result: consistent with constant speed of light', color='black', fontsize=9)
ax_mm.grid(True, alpha=0.2)

# Bottom-right: Time dilation factor gamma vs speed (Special Relativity)
ax_sr = fig.add_subplot(gs[1, 1])
beta = np.linspace(0, 0.99, 500)  # v/c
gamma = 1.0 / np.sqrt(1 - beta**2)
ax_sr.plot(beta, gamma, color='tab:green', lw=2)
ax_sr.set_title('Special Relativity (1905): Time dilation grows with speed')
ax_sr.set_xlabel('Speed as a fraction of c (v/c)')
ax_sr.set_ylabel('Time dilation factor b3')
ax_sr.set_xlim(0, 0.99)
ax_sr.set_ylim(1, 7.5)
# Reference lines
for b in [0.5, 0.9, 0.99]:
    g = 1.0 / np.sqrt(1 - b**2)
    ax_sr.axvline(b, color='gray', ls='--', lw=0.8)
    ax_sr.annotate(f"v/c={b}\nb39{g:.2f}", xy=(b, g), xytext=(b+0.02, min(g+0.7, 7.2)),
                   arrowprops=dict(arrowstyle='->', lw=0.8, color='gray'), fontsize=9)

ax_sr.text(0.08, 1.2, 'Newtonian limit\n(low speeds: b3 9 1)', fontsize=9, color='dimgray')
ax_sr.text(0.55, 5.8, 'High-speed regime:\nRelativistic effects dominate', fontsize=9, color='dimgray')
ax_sr.text(0.2, 3.8, 'GPS requires precise\nrelativistic time corrections', fontsize=9,
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9))
ax_sr.grid(True, alpha=0.2)

fig.tight_layout(rect=[0, 0, 1, 0.95])

outfile = 'historical_development_of_relativity.png'
fig.savefig(outfile, dpi=300)
print(f'Saved figure to {outfile}')

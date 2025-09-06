import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
from matplotlib.patches import ArrowStyle

# Figure and axes
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 20)
ax.set_ylim(0, 10)
ax.axis('off')

# Background regions: lumen (top) and stroma (bottom)
ax.add_patch(Rectangle((0, 5.8), 20, 4.2, facecolor='#e6f2ff', edgecolor='none'))  # Lumen
ax.add_patch(Rectangle((0, 0), 20, 4.2, facecolor='#eaf7ea', edgecolor='none'))   # Stroma

# Thylakoid membrane band
mem_y = 4.2
mem_h = 1.6
ax.add_patch(Rectangle((0.2, mem_y), 19.6, mem_h, facecolor='#78c679', edgecolor='#3a7f3a', lw=1.5, zorder=1))
ax.text(10, mem_y + mem_h/2, 'Thylakoid membrane', ha='center', va='center', fontsize=11, color='black')

# Region labels
ax.text(19.5, 8.8, 'Thylakoid lumen (inside)', ha='right', va='center', fontsize=11, color='#1f4e79')
ax.text(19.5, 1.2, 'Stroma (outside)', ha='right', va='center', fontsize=11, color='#1b5e20')

# Complex positions and sizes
psii = (2.2, mem_y, 2.6, mem_h)
cytb6f = (8.0, mem_y, 2.6, mem_h)
psi = (13.6, mem_y, 2.6, mem_h)
atp_mem = (17.4, mem_y, 1.2, mem_h)

# Draw complexes
def draw_complex(x, y, w, h, label, color='#cfe8cf'):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='black', lw=1.5, zorder=2))
    ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=10)

draw_complex(*psii, 'Photosystem II\n(PSII)', color='#cfe8cf')
draw_complex(*cytb6f, 'Cytochrome b6f', color='#d9e4f5')
draw_complex(*psi, 'Photosystem I\n(PSI)', color='#cfe8cf')

# ATP synthase (membrane part + catalytic head in stroma)
ax.add_patch(Rectangle((atp_mem[0], atp_mem[1]), atp_mem[2], atp_mem[3], facecolor='#ffe0b2', edgecolor='black', lw=1.5, zorder=2))
ax.add_patch(Circle((atp_mem[0] + atp_mem[2]/2, mem_y - 1.0), 0.7, facecolor='#ffd180', edgecolor='black', lw=1.5, zorder=3))
ax.text(atp_mem[0] + atp_mem[2]/2, mem_y - 2.0, 'ATP synthase', ha='center', va='center', fontsize=10)

# Helper to draw arrows
def arrow(p1, p2, color='k', lw=2, style='-|>', ms=16, z=4, alpha=1.0):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(a)
    return a

# Light absorption arrows (yellow) to PSII and PSI
arrow((3.5, 9.2), (3.5, mem_y + mem_h + 0.15), color='#f9a825', lw=2.5, ms=18)
ax.text(3.5, 9.4, 'light', ha='center', va='bottom', color='#f9a825', fontsize=10)
arrow((14.9, 9.0), (14.9, mem_y + mem_h + 0.15), color='#f9a825', lw=2.5, ms=18)
ax.text(14.9, 9.2, 'light', ha='center', va='bottom', color='#f9a825', fontsize=10)

# Water splitting at PSII (on lumen side)
ax.text(psii[0] - 0.4, mem_y + mem_h + 0.6, 'H2O', ha='right', va='center', fontsize=10)
arrow((psii[0] - 0.25, mem_y + mem_h + 0.55), (psii[0] + 0.1, mem_y + mem_h - 0.05), color='#1976d2', lw=2)
# O2 release
arrow((psii[0] + 0.3, mem_y + mem_h + 0.2), (psii[0] + 0.3, 9.6), color='#90a4ae', lw=2)
ax.text(psii[0] + 0.3, 9.7, 'O2', ha='center', va='bottom', fontsize=10, color='#546e7a')
# Protons released into lumen
ax.text(psii[0] + psii[2]/2, mem_y + mem_h + 0.6, '2H+ (to lumen)', ha='center', va='center', fontsize=9, color='#ad1457')

# Electron transport chain (PSII -> Cyt b6f -> PSI)
mid_y = mem_y + mem_h/2
arrow((psii[0] + psii[2], mid_y), (cytb6f[0], mid_y), color='#1565c0', lw=2.5)
ax.text((psii[0] + psii[2] + cytb6f[0]) / 2, mid_y + 0.5, 'e- (via PQ)', ha='center', va='center', fontsize=9, color='#1565c0')
arrow((cytb6f[0] + cytb6f[2], mid_y), (psi[0], mid_y), color='#1565c0', lw=2.5)
ax.text((cytb6f[0] + cytb6f[2] + psi[0]) / 2, mid_y + 0.5, 'e- (via PC)', ha='center', va='center', fontsize=9, color='#1565c0')

# Proton pumping by Cyt b6f (stroma -> lumen)
arrow((cytb6f[0] + cytb6f[2]/2 - 0.4, mem_y - 0.2), (cytb6f[0] + cytb6f[2]/2 - 0.4, mem_y + mem_h + 0.2), color='#ad1457', lw=2)
arrow((cytb6f[0] + cytb6f[2]/2 + 0.4, mem_y - 0.2), (cytb6f[0] + cytb6f[2]/2 + 0.4, mem_y + mem_h + 0.2), color='#ad1457', lw=2)
ax.text(cytb6f[0] + cytb6f[2]/2, mem_y - 0.6, 'H+ pumped\ninto lumen', ha='center', va='top', fontsize=9, color='#880e4f')

# PSI to NADPH formation in stroma
arrow((psi[0] + psi[2], mid_y), (psi[0] + psi[2] + 2.0, 3.0), color='#1565c0', lw=2.5)
ax.text(psi[0] + psi[2] + 2.2, 2.8, 'NADP+ + 2e- + H+ → NADPH', ha='left', va='center', fontsize=10, color='#2e7d32')

# ATP synthase: H+ flow from lumen to stroma generates ATP
arrow((atp_mem[0] + atp_mem[2]/2 - 0.25, mem_y + mem_h + 0.2), (atp_mem[0] + atp_mem[2]/2 - 0.25, mem_y - 0.2), color='#ad1457', lw=2)
arrow((atp_mem[0] + atp_mem[2]/2 + 0.25, mem_y + mem_h + 0.2), (atp_mem[0] + atp_mem[2]/2 + 0.25, mem_y - 0.2), color='#ad1457', lw=2)
ax.text(atp_mem[0] + atp_mem[2]/2, mem_y + mem_h + 0.6, 'H+ down gradient', ha='center', va='bottom', fontsize=9, color='#880e4f')
arrow((atp_mem[0] + atp_mem[2]/2, mem_y - 1.0), (atp_mem[0] + atp_mem[2]/2 + 1.8, mem_y - 1.8), color='#2e7d32', lw=2.5)
ax.text(atp_mem[0] + atp_mem[2]/2 + 2.0, mem_y - 1.8, 'ADP + Pi → ATP', ha='left', va='center', fontsize=10, color='#2e7d32')

# Clarifying labels for electron carriers
ax.text((psii[0] + psii[2] + cytb6f[0]) / 2, mid_y - 0.7, 'plastoquinone (PQ)', ha='center', va='center', fontsize=8, color='#0d47a1')
ax.text((cytb6f[0] + cytb6f[2] + psi[0]) / 2, mid_y - 0.7, 'plastocyanin (PC)', ha='center', va='center', fontsize=8, color='#0d47a1')
ax.text(psi[0] + psi[2] + 0.8, 3.7, 'ferredoxin + FNR', ha='left', va='center', fontsize=8, color='#0d47a1')

# Title
ax.text(10, 10.0, 'Light-Dependent Reactions of Photosynthesis', ha='center', va='top', fontsize=16, fontweight='bold')
ax.text(10, 9.5, 'Conversion of light energy to ATP and NADPH across the thylakoid membrane', ha='center', va='top', fontsize=11, color='#424242')

# Save figure
out_name = 'light_dependent_reactions_photosynthesis.png'
plt.savefig(out_name, dpi=300, bbox_inches='tight')
print(f'Saved figure to {out_name}')

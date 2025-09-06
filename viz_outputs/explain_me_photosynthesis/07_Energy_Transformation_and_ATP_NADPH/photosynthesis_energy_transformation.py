import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, ArrowStyle

# Figure setup
fig, ax = plt.subplots(figsize=(12, 7))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# Colors
membrane_color = '#9BD07C'      # light green for thylakoid membrane
membrane_edge = '#3C8D2F'       # darker green edge
protein_color = '#4C7BC3'       # protein complexes
atp_color = '#F2C14E'           # ATP bubble
nadph_color = '#5CC8A1'         # NADPH bubble
highlight = '#D95F02'           # arrows/emphasis
hplus_color = '#D93A49'         # H+
text_gray = '#333333'

# Thylakoid membrane (single thick slab for clarity)
membrane = Rectangle((3.0, 3.1), 7.0, 0.8, facecolor=membrane_color, edgecolor=membrane_edge, linewidth=2)
ax.add_patch(membrane)

# Region labels
ax.text(1.0, 5.6, 'Thylakoid Lumen', fontsize=12, color=text_gray)
ax.text(1.0, 1.2, 'Stroma (chloroplast)', fontsize=12, color=text_gray)

# Sun (light source)
sun = Circle((1.2, 5.2), 0.5, facecolor='#FFD166', edgecolor='#CC9A00', linewidth=2)
ax.add_patch(sun)
# Sun rays
angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
for ang in angles:
    x0, y0 = 1.2 + 0.5*np.cos(ang), 5.2 + 0.5*np.sin(ang)
    x1, y1 = 1.2 + 0.9*np.cos(ang), 5.2 + 0.9*np.sin(ang)
    ax.plot([x0, x1], [y0, y1], color='#CC9A00', linewidth=1.8)
ax.text(1.2, 4.4, 'Light', ha='center', color=text_gray)

# Helper to add membrane protein boxes
def add_protein(x, label, width=0.9, height=1.7, color=protein_color):
    y = 3.1 + (0.8 - height)/2  # center across thickness
    box = FancyBboxPatch((x - width/2, y), width, height,
                         boxstyle='round,pad=0.02,rounding_size=0.08',
                         facecolor=color, edgecolor='#2D4A86', linewidth=2)
    ax.add_patch(box)
    ax.text(x, y + height + 0.15, label, ha='center', va='bottom', fontsize=10, color='white',
            bbox=dict(boxstyle='round,pad=0.2', fc=color, ec='none', alpha=0.9))
    return box

# Photosynthetic complexes
psii_x = 4.2
cytb6f_x = 6.0
psi_x = 7.6
atp_syn_x = 9.2

add_protein(psii_x, 'Photosystem II (PSII)')
add_protein(cytb6f_x, 'Cyt b6f')
add_protein(psi_x, 'Photosystem I (PSI)')

# ATP synthase: stylized with rotor in stroma
atp_box = add_protein(atp_syn_x, 'ATP Synthase', width=0.9, height=1.7, color='#7E57C2')
rotor = Circle((atp_syn_x, 2.2), 0.28, facecolor='#7E57C2', edgecolor='#4A2F82', linewidth=2)
ax.add_patch(rotor)
ax.text(atp_syn_x, 1.75, 'Rotor', ha='center', va='top', fontsize=8, color='#4A2F82')

# Light to PSII arrow
light_arrow = FancyArrowPatch((1.9, 5.0), (psii_x - 0.4, 4.6),
                              arrowstyle=ArrowStyle('Simple', head_length=8, head_width=8, tail_width=2),
                              mutation_scale=12, color='#F8961E', linewidth=0, alpha=0.9)
ax.add_patch(light_arrow)
ax.text(psii_x - 0.2, 4.85, 'photons', fontsize=9, color='#F8961E')

# Water splitting at PSII (on lumen side)
ax.text(psii_x - 0.95, 4.45, 'H2O', fontsize=10, color=text_gray)
ax.text(psii_x - 1.0, 4.85, '\u2192 O2 + 2 H+ + 2 e-', fontsize=9, color=text_gray)
water_arrow = FancyArrowPatch((psii_x - 0.8, 4.3), (psii_x - 0.15, 4.1),
                              arrowstyle='->', mutation_scale=12, linewidth=2, color=text_gray)
ax.add_patch(water_arrow)
# O2 bubble
o2 = Circle((psii_x - 0.6, 4.95), 0.12, facecolor='none', edgecolor=text_gray, linewidth=1.5)
ax.add_patch(o2)
ax.text(psii_x - 0.6, 4.95, 'O2', ha='center', va='center', fontsize=8, color=text_gray)

# Electron transport chain arrows (stroma side path)
# PSII -> Cyt b6f
et1 = FancyArrowPatch((psii_x + 0.45, 3.0), (cytb6f_x - 0.45, 3.0),
                      arrowstyle='-|>', mutation_scale=14, linewidth=2.5, color=highlight)
ax.add_patch(et1)
# Cyt b6f -> PSI
et2 = FancyArrowPatch((cytb6f_x + 0.45, 3.0), (psi_x - 0.45, 3.0),
                      arrowstyle='-|>', mutation_scale=14, linewidth=2.5, color=highlight)
ax.add_patch(et2)
# PSI -> NADP+ reductase -> NADPH (in stroma)
# Small FNR node
fnr_x, fnr_y = psi_x + 0.9, 2.8
fnr = Circle((fnr_x, fnr_y), 0.16, facecolor='#2A9D8F', edgecolor='#1C6E66', linewidth=1.5)
ax.add_patch(fnr)
ax.text(fnr_x, fnr_y - 0.38, 'FNR\n(NADP+ reductase)', ha='center', va='top', fontsize=8, color='#1C6E66')

et3 = FancyArrowPatch((psi_x + 0.45, 3.0), (fnr_x - 0.2, fnr_y + 0.02),
                      arrowstyle='-|>', mutation_scale=14, linewidth=2.5, color=highlight)
ax.add_patch(et3)

# NADPH bubble
nadph_pos = (fnr_x + 0.9, 2.3)
nadph = Circle(nadph_pos, 0.32, facecolor=nadph_color, edgecolor='#2D7F68', linewidth=2)
ax.add_patch(nadph)
ax.text(nadph_pos[0], nadph_pos[1], 'NADPH', ha='center', va='center', fontsize=9, color='white')
ax.text(nadph_pos[0], nadph_pos[1]-0.55, 'reducing power (e-)', ha='center', va='top', fontsize=8, color='#2D7F68')

nadp_arrow = FancyArrowPatch((fnr_x + 0.18, fnr_y - 0.02), (nadph_pos[0] - 0.35, nadph_pos[1] + 0.12),
                             arrowstyle='-|>', mutation_scale=14, linewidth=2, color='#2D7F68')
ax.add_patch(nadp_arrow)
ax.text(fnr_x + 0.55, fnr_y + 0.28, 'NADP+ + 2 e- + H+ \u2192', fontsize=8, color='#2D7F68')

# Proton pumping at Cyt b6f (stroma -> lumen)
for dx in [-0.15, 0.0, 0.15]:
    pump = FancyArrowPatch((cytb6f_x + dx, 3.15), (cytb6f_x + dx, 4.35),
                           arrowstyle='-|>', mutation_scale=14, linewidth=2, color=hplus_color)
    ax.add_patch(pump)
    ax.text(cytb6f_x + dx + 0.05, 4.45, 'H+', fontsize=9, color=hplus_color)
ax.text(cytb6f_x, 4.65, 'H+ pumped to lumen', ha='center', fontsize=8, color=hplus_color)

# H+ flow through ATP synthase (lumen -> stroma)
for dy in [0.0, 0.25]:
    flow = FancyArrowPatch((atp_syn_x - 0.2 + dy, 4.2), (atp_syn_x - 0.2 + dy, 2.5),
                           arrowstyle='-|>', mutation_scale=14, linewidth=2, color=hplus_color)
    ax.add_patch(flow)
    ax.text(atp_syn_x - 0.5 + dy, 4.35, 'H+', fontsize=9, color=hplus_color)
ax.text(atp_syn_x + 0.65, 3.35, 'H+ flow drives\nATP synthesis', ha='left', va='center', fontsize=8, color='#4A2F82')

# ATP bubble near ATP synthase (stroma side)
atp_pos = (atp_syn_x + 0.9, 2.1)
atp = Circle(atp_pos, 0.32, facecolor=atp_color, edgecolor='#B3861A', linewidth=2)
ax.add_patch(atp)
ax.text(atp_pos[0], atp_pos[1], 'ATP', ha='center', va='center', fontsize=10, color='white')
ax.text(atp_pos[0], atp_pos[1]-0.55, 'energy currency', ha='center', va='top', fontsize=8, color='#B3861A')

# Calvin Cycle representation (in stroma)
cycle_center = (5.8, 1.2)
cycle_r = 0.9
# Dashed circular arrows to suggest a cycle
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(cycle_center[0] + cycle_r*np.cos(theta),
        cycle_center[1] + cycle_r*np.sin(theta),
        color='#2E4057', linestyle='--', linewidth=2)
ax.text(cycle_center[0], cycle_center[1], 'Calvin\nCycle', ha='center', va='center', fontsize=10, color='#2E4057')
ax.text(cycle_center[0], cycle_center[1]-1.0, 'uses ATP + NADPH to fix CO2', ha='center', fontsize=9, color='#2E4057')

# Arrows from ATP and NADPH to Calvin Cycle
atp_to_cycle = FancyArrowPatch((atp_pos[0] - 0.1, atp_pos[1] - 0.1), (cycle_center[0] + 0.8, cycle_center[1] + 0.2),
                               arrowstyle='-|>', mutation_scale=14, linewidth=2.5, color=atp_color)
ax.add_patch(atp_to_cycle)
ax.text((atp_pos[0] + cycle_center[0]) / 2 + 0.6, (atp_pos[1] + cycle_center[1]) / 2 + 0.4, 'ATP', fontsize=9, color='#B3861A')

nadph_to_cycle = FancyArrowPatch((nadph_pos[0] - 0.2, nadph_pos[1] - 0.05), (cycle_center[0] + 0.2, cycle_center[1] + 0.3),
                                 arrowstyle='-|>', mutation_scale=14, linewidth=2.5, color=nadph_color)
ax.add_patch(nadph_to_cycle)
ax.text((nadph_pos[0] + cycle_center[0]) / 2 + 0.2, (nadph_pos[1] + cycle_center[1]) / 2 + 0.45, 'NADPH', fontsize=9, color='#2D7F68')

# Output sugars from Calvin Cycle
sugar_pos = (cycle_center[0] + 2.0, cycle_center[1])
sugar_arrow = FancyArrowPatch((cycle_center[0] + cycle_r, cycle_center[1]), (sugar_pos[0] - 0.4, sugar_pos[1]),
                              arrowstyle='-|>', mutation_scale=14, linewidth=2, color='#6C757D')
ax.add_patch(sugar_arrow)
ax.text(sugar_pos[0], sugar_pos[1], 'Sugars (e.g., glucose)', ha='left', va='center', fontsize=9, color='#6C757D')

# Title
ax.text(6.0, 6.6, 'Energy Transformation in Photosynthesis: Light Reactions Produce ATP and NADPH',
        ha='center', va='center', fontsize=14, color=text_gray, weight='bold')
ax.text(6.0, 6.1, 'Light energy \u2192 electron transport \u2192 H+ gradient \u2192 ATP; PSI reduces NADP+ \u2192 NADPH',
        ha='center', va='center', fontsize=10, color=text_gray)

# Save figure
outfile = 'photosynthesis_energy_transformation.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
print(f'Saved figure to {outfile}')

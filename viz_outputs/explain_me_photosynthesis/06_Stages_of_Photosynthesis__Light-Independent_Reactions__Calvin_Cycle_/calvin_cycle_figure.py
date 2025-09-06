import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch


def pol2cart(angle_deg, radius, center=(0.0, 0.0)):
    a = np.deg2rad(angle_deg)
    return center[0] + radius * np.cos(a), center[1] + radius * np.sin(a)


def add_textbox(ax, xy, text, ha='center', va='center', fontsize=11, box_fc='white', box_ec='black'):
    ax.text(
        xy[0], xy[1], text,
        ha=ha, va=va, fontsize=fontsize, color='black',
        bbox=dict(boxstyle='round,pad=0.4', fc=box_fc, ec=box_ec, lw=1)
    )


def add_arrow(ax, start, end, color='black', lw=2, rad=0.0, arrowstyle='-|>', mutation_scale=14):
    arrow = FancyArrowPatch(
        start, end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=arrowstyle,
        mutation_scale=mutation_scale,
        lw=lw,
        color=color
    )
    ax.add_patch(arrow)


def main():
    plt.rcParams.update({
        'font.size': 11,
        'axes.edgecolor': 'none'
    })

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scene bounds
    ax.set_xlim(-6, 6)
    ax.set_ylim(-4.5, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title and subtitle
    ax.text(0, 5.3, 'Calvin Cycle (Light-Independent Reactions)', ha='center', va='center', fontsize=18, fontweight='bold')
    ax.text(0, 4.7, 'Occurs in the stroma of chloroplasts; uses ATP and NADPH to fix CO2 into sugars', ha='center', va='center', fontsize=12)

    # Central cycle circle
    r = 2.2
    circ = Circle((0, 0), r, fill=False, lw=2.2, ec='forestgreen')
    ax.add_patch(circ)

    # Step anchor angles (clockwise progression): 1=90°, 2=-30°, 3=210°
    a1, a2, a3 = 90, -30, 210

    # Draw cycle arrows (clockwise) using curved connectors
    p1 = pol2cart(a1, r)
    p2 = pol2cart(a2, r)
    p3 = pol2cart(a3, r)

    # Slightly inside the circle to avoid overlapping with the circle edge
    p1i = pol2cart(a1, r * 0.98)
    p2i = pol2cart(a2, r * 0.98)
    p3i = pol2cart(a3, r * 0.98)

    # 1 -> 2, 2 -> 3, 3 -> 1
    add_arrow(ax, p1i, p2i, color='forestgreen', lw=2.5, rad=-0.22, mutation_scale=16)
    add_arrow(ax, p2i, p3i, color='forestgreen', lw=2.5, rad=-0.22, mutation_scale=16)
    add_arrow(ax, p3i, p1i, color='forestgreen', lw=2.5, rad=-0.22, mutation_scale=16)

    # Step labels placed just outside the circle
    step_radius = r + 0.9

    s1_pos = pol2cart(a1, step_radius)
    s2_pos = pol2cart(a2, step_radius)
    s3_pos = pol2cart(a3, step_radius)

    add_textbox(
        ax, s1_pos,
        '1. Carbon fixation (RuBisCO)\nCO2 + RuBP → 3-PGA',
        ha='center', va='center', fontsize=11, box_fc='#e9f7ef', box_ec='forestgreen'
    )
    add_textbox(
        ax, s2_pos,
        '2. Reduction\n3-PGA → G3P\nUses ATP + NADPH',
        ha='center', va='center', fontsize=11, box_fc='#fff3e0', box_ec='#f39c12'
    )
    add_textbox(
        ax, s3_pos,
        '3. Regeneration of RuBP\nUses ATP',
        ha='center', va='center', fontsize=11, box_fc='#e8f0fe', box_ec='#1e88e5'
    )

    # Connect step labels to the cycle with subtle arrows
    add_arrow(ax, pol2cart(a1, step_radius - 0.2), pol2cart(a1, r), color='forestgreen', lw=1.6, rad=0.0, mutation_scale=10)
    add_arrow(ax, pol2cart(a2, step_radius - 0.2), pol2cart(a2, r), color='#f39c12', lw=1.6, rad=0.0, mutation_scale=10)
    add_arrow(ax, pol2cart(a3, step_radius - 0.2), pol2cart(a3, r), color='#1e88e5', lw=1.6, rad=0.0, mutation_scale=10)

    # Internal labels
    ax.text(0, 0.2, 'RuBP (5C) pool', ha='center', va='center', fontsize=12, color='darkgreen')
    ax.text(0, -0.35, 'Cycle repeats to store energy in stable sugars', ha='center', va='center', fontsize=10, color='dimgray')

    # External inputs/outputs
    # CO2 input to Step 1 (top)
    co2_box = (-4.7, 3.2)
    add_textbox(ax, co2_box, 'CO2', ha='center', va='center', fontsize=12, box_fc='white', box_ec='black')
    add_arrow(ax, co2_box, pol2cart(a1, r*0.95), color='black', lw=2, rad=0.0, mutation_scale=14)

    # Energy carriers into Reduction (Step 2)
    energy_in_box = (4.8, -0.5)
    add_textbox(ax, energy_in_box, 'ATP + NADPH\n(from light reactions)', ha='center', va='center', fontsize=11, box_fc='white', box_ec='#f39c12')
    add_arrow(ax, energy_in_box, pol2cart(a2, r*0.95), color='#f39c12', lw=2.2, rad=0.0, mutation_scale=14)

    # Spent carriers out
    spent_box = (4.8, -2.5)
    add_textbox(ax, spent_box, 'ADP + Pi + NADP+', ha='center', va='center', fontsize=11, box_fc='white', box_ec='gray')
    add_arrow(ax, pol2cart(a2 - 10, r*0.95), spent_box, color='gray', lw=2, rad=0.0, mutation_scale=14)

    # G3P export (sugar precursor)
    g3p_box = (4.8, 1.8)
    add_textbox(ax, g3p_box, 'G3P (sugar)\n→ Glucose & starch', ha='center', va='center', fontsize=11, box_fc='white', box_ec='purple')
    add_arrow(ax, pol2cart(a2 + 10, r*0.95), g3p_box, color='purple', lw=2.2, rad=0.0, mutation_scale=14)

    # Supporting note about stoichiometry (pedagogical)
    ax.text(0, -3.7,
            'Per 3 CO2 fixed: uses 9 ATP and 6 NADPH → exports 1 G3P; remaining G3P regenerates RuBP',
            ha='center', va='center', fontsize=10, color='dimgray')

    plt.savefig('calvin_cycle_diagram.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, RegularPolygon


def draw_labeled_circle(ax, xy, radius, facecolor, edgecolor, text, count_text=None, text_color='white', linewidth=2):
    circle = Circle(xy, radius, facecolor=facecolor, edgecolor=edgecolor, linewidth=linewidth)
    ax.add_patch(circle)
    ax.text(xy[0], xy[1], text, ha='center', va='center', color=text_color, fontsize=14, weight='bold')
    if count_text:
        ax.text(xy[0] + radius + 0.25, xy[1], count_text, ha='left', va='center', color=edgecolor, fontsize=13, weight='bold')


def draw_sun(ax, center, radius_outer=0.9, rays=12, facecolor='#FFD54F', edgecolor='#F9A825'):
    # Rays as a star polygon
    star = RegularPolygon(center, numVertices=rays, radius=radius_outer, orientation=np.pi / rays,
                          facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(star)
    core = Circle(center, radius_outer * 0.45, facecolor='#FFF59D', edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(core)


def add_value_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{int(height)}", ha='center', va='bottom', fontsize=11)


def make_figure():
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 12
    })

    fig = plt.figure(figsize=(12, 6), facecolor='white')
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1], wspace=0.18)

    # Left panel: Schematic equation
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')

    react_color = '#1f77b4'  # blue
    prod_color = '#2ca02c'   # green
    accent_grey = '#616161'

    # Titles
    ax1.text(1.0, 9.4, 'Reactants', color=react_color, fontsize=14, weight='bold')
    ax1.text(7.6, 9.4, 'Products', color=prod_color, fontsize=14, weight='bold')

    # Reactants: 6 CO2 and 6 H2O + light
    draw_labeled_circle(ax1, (2.0, 6.2), radius=0.95, facecolor=react_color, edgecolor=react_color, text='CO$_2$', count_text='×6')
    draw_labeled_circle(ax1, (2.0, 3.6), radius=0.95, facecolor=react_color, edgecolor=react_color, text='H$_2$O', count_text='×6')

    # Light energy
    draw_sun(ax1, center=(4.7, 8.2), radius_outer=0.9)
    ax1.text(4.7, 6.9, 'light energy', ha='center', va='top', fontsize=12, color=accent_grey)

    # Arrow to products
    arrow = FancyArrowPatch((3.7, 5.0), (6.7, 5.0), arrowstyle='-|>', mutation_scale=30, linewidth=2.5, color=accent_grey)
    ax1.add_patch(arrow)

    # Products: Glucose and Oxygen
    # Glucose as a hexagon
    hexagon = RegularPolygon((8.0, 6.4), numVertices=6, radius=1.2, orientation=np.pi/6,
                             facecolor=prod_color, edgecolor=prod_color, linewidth=2)
    ax1.add_patch(hexagon)
    ax1.text(8.0, 6.4, 'C$_6$H$_{12}$O$_6$', ha='center', va='center', color='white', fontsize=14, weight='bold')

    # Oxygen molecules
    draw_labeled_circle(ax1, (8.0, 3.6), radius=0.95, facecolor=prod_color, edgecolor=prod_color, text='O$_2$', count_text='×6')

    # Equation text (balanced)
    eq = r'$6\,\mathrm{CO_2} + 6\,\mathrm{H_2O} + \text{light energy} \;\rightarrow\; \mathrm{C_6H_{12}O_6} + 6\,\mathrm{O_2}$'
    ax1.text(0.5, -0.10, eq, transform=ax1.transAxes, ha='center', va='top', fontsize=14)

    # Right panel: Atom balance bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    elements = ['C', 'H', 'O']
    react_counts = np.array([6, 12, 18])
    prod_counts = np.array([6, 12, 18])

    x = np.arange(len(elements))
    width = 0.35

    bars1 = ax2.bar(x - width/2, react_counts, width, label='Reactants', color=react_color)
    bars2 = ax2.bar(x + width/2, prod_counts, width, label='Products', color=prod_color)

    add_value_labels(ax2, bars1)
    add_value_labels(ax2, bars2)

    ax2.set_xticks(x)
    ax2.set_xticklabels(elements, fontsize=12)
    ax2.set_ylabel('Number of atoms', fontsize=12)
    ax2.set_title('Atom balance (stoichiometry): matter is conserved', fontsize=13)
    ax2.legend(frameon=False)
    ax2.set_ylim(0, max(react_counts.max(), prod_counts.max()) * 1.25)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # Save figure
    outfile = 'photosynthesis_chemical_equation.png'
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f'Saved figure to {outfile}')


if __name__ == '__main__':
    make_figure()

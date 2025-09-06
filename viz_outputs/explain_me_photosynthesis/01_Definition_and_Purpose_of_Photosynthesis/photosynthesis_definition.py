import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch
from matplotlib.patches import FancyArrowPatch


def add_sun(ax, center=(1.2, 5.2), radius=0.5, n_rays=12):
    x0, y0 = center
    sun = Circle((x0, y0), radius, facecolor='#f9d71c', edgecolor='#d4b013', linewidth=2)
    ax.add_patch(sun)
    angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
    for a in angles:
        r1 = radius * 1.05
        r2 = radius * 1.7
        x1, y1 = x0 + r1*np.cos(a), y0 + r1*np.sin(a)
        x2, y2 = x0 + r2*np.cos(a), y0 + r2*np.sin(a)
        ax.add_line(plt.Line2D([x1, x2], [y1, y2], color='#f3c613', linewidth=2.2))


def add_cloud(ax, center=(8.7, 5.0), scale=0.9):
    x0, y0 = center
    blobs = [(-0.9, 0.0, 0.9), (-0.3, 0.2, 0.8), (0.3, 0.1, 0.7), (0.9, 0.0, 0.85)]
    for dx, dy, s in blobs:
        c = Circle((x0 + dx*scale, y0 + dy*scale), 0.45*scale*s, facecolor='white', edgecolor='#b0c4de', linewidth=2)
        ax.add_patch(c)
    # Base
    ax.add_line(plt.Line2D([x0-1.0*scale, x0+1.05*scale], [y0-0.35*scale, y0-0.35*scale], color='white', linewidth=8, solid_capstyle='round'))
    ax.add_line(plt.Line2D([x0-1.0*scale, x0+1.05*scale], [y0-0.35*scale, y0-0.35*scale], color='#b0c4de', linewidth=2, alpha=0.8))
 

def add_leaf_and_stem(ax):
    # Stem
    ax.plot([4.0, 4.0], [1.0, 3.0], color='#2f4f4f', linewidth=5, solid_capstyle='round')
    # Leaf
    leaf = Ellipse((5.0, 3.3), width=3.2, height=1.8, angle=20, facecolor='#2e8b57', edgecolor='#1f5d3a', linewidth=2)
    ax.add_patch(leaf)
    # Midrib of leaf
    ax.add_line(plt.Line2D([3.7, 6.2], [2.9, 3.7], color='#1f5d3a', linewidth=2))
    # Few small chloroplast ovals inside leaf
    chloros = [(4.6, 3.5), (5.2, 3.2), (4.8, 3.0), (5.6, 3.55), (4.2, 3.2)]
    for (cx, cy) in chloros:
        ch = Ellipse((cx, cy), 0.35, 0.18, angle=25, facecolor='#a8e6a3', edgecolor='#5aa86a', linewidth=1)
        ax.add_patch(ch)
    ax.text(5.0, 4.35, 'Leaf (chloroplasts)', color='#1f5d3a', fontsize=12, ha='center', va='bottom')


def arrow(ax, start, end, color, label=None, label_offset=(0, 0), lw=2.8, style='-|>', mutation_scale=18, zorder=5):
    arr = FancyArrowPatch(posA=start, posB=end,
                          arrowstyle=style, mutation_scale=mutation_scale,
                          linewidth=lw, color=color, zorder=zorder)
    ax.add_patch(arr)
    if label:
        xm = (start[0] + end[0]) / 2 + label_offset[0]
        ym = (start[1] + end[1]) / 2 + label_offset[1]
        ax.text(xm, ym, label, color=color, fontsize=12, fontweight='bold', ha='center', va='center')


def add_water_droplet(ax, center=(3.0, 0.6), scale=0.15):
    x, y = center
    # Simple teardrop using polygon
    pts = np.array([
        [x, y+2.5*scale],
        [x-1.2*scale, y+0.9*scale],
        [x, y-0.8*scale],
        [x+1.2*scale, y+0.9*scale]
    ])
    drop = Polygon(pts, closed=True, facecolor='#1f77b4', edgecolor='#105183', linewidth=1.5)
    ax.add_patch(drop)


def add_hexagon(ax, center=(8.6, 2.5), radius=0.75, facecolor='#ffd39b', edgecolor='#ff8c00'):
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6  # flat-top
    verts = np.column_stack((cx + radius*np.cos(angles), cy + radius*np.sin(angles)))
    hex_poly = Polygon(verts, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(hex_poly)
    ax.text(cx, cy+0.05, 'Glucose', color='#cc6a00', fontsize=12, fontweight='bold', ha='center', va='bottom')
    ax.text(cx, cy-0.25, 'C6H12O6', color='#cc6a00', fontsize=11, ha='center', va='top')


def add_purpose_box(ax, xy=(7.1, 3.9), width=2.8, height=1.7):
    box = FancyBboxPatch(xy, width, height, boxstyle='round,pad=0.25,rounding_size=0.08',
                         facecolor='#f5f7fb', edgecolor='#7a8aa1', linewidth=1.5, alpha=0.95)
    ax.add_patch(box)
    x0, y0 = xy
    ax.text(x0 + 0.1, y0 + height - 0.25, 'Why it matters', color='#2c3e50', fontsize=12, fontweight='bold', ha='left', va='top')
    bullets = [
        'Converts light to chemical energy (glucose)',
        'Releases O2, uses CO2',
        'Basis of food chains and energy flow'
    ]
    for i, b in enumerate(bullets):
        ax.text(x0 + 0.15, y0 + height - 0.55 - i*0.42, '\u2022 ' + b, color='#2c3e50', fontsize=10.5, ha='left', va='top')


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect('equal')
    ax.axis('off')

    # Title and subtitle
    fig.suptitle('Photosynthesis: Definition and Purpose', fontsize=18, fontweight='bold', y=0.98)
    ax.text(0.5, 5.75, 'Plants convert light energy into chemical energy (glucose), releasing oxygen and sustaining life.',
            fontsize=12.5, color='#34495e', ha='left', va='center')

    # Elements
    add_sun(ax, center=(1.25, 5.1), radius=0.55)
    add_leaf_and_stem(ax)
    add_cloud(ax, center=(8.8, 5.0), scale=0.95)
    add_hexagon(ax, center=(8.6, 2.5), radius=0.85)

    # Inputs
    arrow(ax, start=(1.95, 4.9), end=(4.45, 3.85), color='#f1b70f', label='Light energy', label_offset=(0.0, 0.25), lw=3.2, mutation_scale=20)
    arrow(ax, start=(0.9, 3.0), end=(3.9, 3.0), color='dimgray', label='CO2', label_offset=(0.0, 0.25))
    add_water_droplet(ax, center=(3.0, 0.7), scale=0.18)
    arrow(ax, start=(3.0, 0.6), end=(4.0, 2.0), color='#1f77b4', label='H2O', label_offset=(-0.2, -0.2))

    # Outputs
    arrow(ax, start=(6.2, 3.95), end=(8.1, 5.0), color='#17a2b8', label='O2', label_offset=(0.0, 0.25))
    arrow(ax, start=(6.2, 2.9), end=(7.7, 2.55), color='#ff8c00', label='Glucose', label_offset=(0.0, -0.3))

    # Purpose box
    add_purpose_box(ax, xy=(6.9, 3.9), width=3.0, height=1.8)

    # Equation and context
    ax.text(5.0, 0.6,
            '6 CO2  +  6 H2O  +  light energy  →  C6H12O6  +  6 O2',
            fontsize=13, color='#2c3e50', ha='center', va='center')
    ax.text(5.0, 0.25,
            'Occurs in chloroplasts of plants, algae, and some bacteria; anchors the carbon cycle and supports most life.',
            fontsize=10.5, color='#5d6d7e', ha='center', va='center')

    # Save figure
    plt.savefig('photosynthesis_definition.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()

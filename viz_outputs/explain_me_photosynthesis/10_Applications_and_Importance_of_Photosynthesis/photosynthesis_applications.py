import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch


def draw_box(ax, center, width, height, facecolor, edgecolor, title, bullets, title_color='black'):
    cx, cy = center
    box = FancyBboxPatch((cx - width/2, cy - height/2), width, height,
                         boxstyle="round,pad=0.02,rounding_size=0.06",
                         linewidth=2, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(box)

    # Text layout inside the box
    margin_x = 0.15 * width
    top_y = cy + height/2 - 0.28
    # Title
    ax.text(cx, top_y, title, ha='center', va='top', fontsize=13, fontweight='bold', color=title_color)

    # Bullets
    y = top_y - 0.6
    line_spacing = 0.48
    for bullet in bullets:
        ax.text(cx - width/2 + margin_x, y, f"\u2022 {bullet}", ha='left', va='top', fontsize=11, color='black')
        y -= line_spacing

    return box


def arrow_from_center_to_box(ax, center, radius, box_center, box_w, box_h, color='#555555'):
    # Compute points so arrow starts at circle edge and ends just outside box edge
    cx, cy = center
    bx, by = box_center
    d = np.array([bx - cx, by - cy], dtype=float)
    norm = np.hypot(d[0], d[1])
    if norm == 0:
        return
    u = d / norm

    # Start at circle edge (slightly outside for visibility)
    start = np.array([cx, cy]) + u * (radius + 0.05)

    # End near rectangle edge along the direction from box center to circle center
    a = box_w / 2.0
    b = box_h / 2.0
    dx, dy = d
    # Ratios for intersection with rectangle boundary
    rx = abs(dx) / a if a > 0 and dx != 0 else 0.0
    ry = abs(dy) / b if b > 0 and dy != 0 else 0.0
    denom = max(rx, ry) if max(rx, ry) > 0 else 1.0
    t = 1.0 / denom
    gap = 0.08  # small gap outside the box
    end = np.array([bx, by]) - d * (t + gap)

    arrow = FancyArrowPatch(posA=(start[0], start[1]), posB=(end[0], end[1]),
                            arrowstyle='-|>', mutation_scale=16, lw=2.2, color=color)
    ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Central concept: Photosynthesis
    center = (6.5, 5.0)
    radius = 1.35
    core_circle = Circle(center, radius=radius, facecolor='#C7E9C0', edgecolor='#2B8CBE', linewidth=2.5)
    ax.add_patch(core_circle)

    # Central text
    ax.text(center[0], center[1] + 0.4, 'Photosynthesis', ha='center', va='center', fontsize=16, fontweight='bold', color='#0B5D1E')
    ax.text(center[0], center[1] - 0.1, 'CO2 + H2O + light \N{RIGHTWARDS ARROW} sugars + O2',
            ha='center', va='center', fontsize=12, color='#1A1A1A')

    # Inputs arrow (into circle)
    inputs_pt = (center[0] - 2.6, center[1] + 1.8)
    in_arrow = FancyArrowPatch(posA=inputs_pt,
                               posB=(center[0] - radius/np.sqrt(2), center[1] + radius/np.sqrt(2)),
                               arrowstyle='-|>', mutation_scale=14, lw=2, color='#FDB813')
    ax.add_patch(in_arrow)
    ax.text(inputs_pt[0] - 0.05, inputs_pt[1] + 0.2, 'Light', ha='right', va='bottom', fontsize=11, color='#B57400')
    ax.text(inputs_pt[0] - 0.05, inputs_pt[1] - 0.15, 'CO2 + H2O', ha='right', va='top', fontsize=11, color='#555555')

    # Outputs arrow (out of circle)
    outputs_pt = (center[0] + 2.7, center[1] - 1.9)
    out_arrow = FancyArrowPatch(posA=(center[0] + radius/np.sqrt(2), center[1] - radius/np.sqrt(2)),
                                posB=outputs_pt,
                                arrowstyle='-|>', mutation_scale=14, lw=2, color='#2B8CBE')
    ax.add_patch(out_arrow)
    ax.text(outputs_pt[0] + 0.05, outputs_pt[1] + 0.2, 'Sugars', ha='left', va='bottom', fontsize=11, color='#1A1A1A')
    ax.text(outputs_pt[0] + 0.05, outputs_pt[1] - 0.15, 'O2', ha='left', va='top', fontsize=11, color='#1A1A1A')

    # Title
    ax.text(0.5, 0.97, 'Applications and Importance of Photosynthesis', transform=ax.transAxes,
            ha='left', va='top', fontsize=18, fontweight='bold', color='#0B5D1E')
    ax.text(0.5, 0.93, 'Foundation for agriculture, climate solutions, ecosystem health, biofuels, and bioengineering',
            transform=ax.transAxes, ha='left', va='top', fontsize=11, color='#333333')

    # Define application boxes
    boxes = [
        {
            'center': (6.5, 9.0),
            'w': 4.8, 'h': 2.4,
            'face': '#E8F5E9', 'edge': '#2E7D32', 'title': 'Agriculture & Food Security',
            'bullets': ['Increase crop yields', 'Improve water & nitrogen-use efficiency', 'Breeding for stress-tolerant, efficient plants'],
            'tcolor': '#1B5E20'
        },
        {
            'center': (11.0, 6.8),
            'w': 3.6, 'h': 2.2,
            'face': '#FFF3E0', 'edge': '#EF6C00', 'title': 'Biofuels & Bioproducts',
            'bullets': ['Algal/plant biomass to fuels', 'Toward carbon-neutral feedstocks'],
            'tcolor': '#E65100'
        },
        {
            'center': (6.5, 1.3),
            'w': 4.8, 'h': 2.4,
            'face': '#E3F2FD', 'edge': '#1565C0', 'title': 'Climate & Carbon Cycle',
            'bullets': ['Draws down atmospheric CO2', 'Maintains O2 for aerobic life', 'Natural climate mitigation'],
            'tcolor': '#0D47A1'
        },
        {
            'center': (2.0, 6.8),
            'w': 3.6, 'h': 2.2,
            'face': '#E0F2F1', 'edge': '#00695C', 'title': 'Ecosystems & Services',
            'bullets': ['Base of food webs', 'Supports habitat & soil health'],
            'tcolor': '#004D40'
        },
        {
            'center': (11.0, 3.2),
            'w': 3.6, 'h': 2.2,
            'face': '#F3E5F5', 'edge': '#6A1B9A', 'title': 'Biotechnology & Bioengineering',
            'bullets': ['Engineered photosystems', 'Solar-to-chemical devices'],
            'tcolor': '#4A148C'
        }
    ]

    # Draw boxes and arrows
    for b in boxes:
        box_patch = draw_box(ax, b['center'], b['w'], b['h'], b['face'], b['edge'], b['title'], b['bullets'], title_color=b['tcolor'])
        arrow_from_center_to_box(ax, center, radius, b['center'], b['w'], b['h'])

    # Subtle legend-like cues (icons) using simple shapes near some boxes
    # Sun rays near inputs label
    sun_center = (center[0] - 2.95, center[1] + 2.05)
    sun_r = 0.14
    for ang in np.linspace(0, 2*np.pi, 12, endpoint=False):
        x0 = sun_center[0] + np.cos(ang) * (sun_r)
        y0 = sun_center[1] + np.sin(ang) * (sun_r)
        x1 = sun_center[0] + np.cos(ang) * (sun_r + 0.22)
        y1 = sun_center[1] + np.sin(ang) * (sun_r + 0.22)
        ax.plot([x0, x1], [y0, y1], color='#FDB813', lw=2)
    circle_sun = Circle(sun_center, sun_r, facecolor='#FDB813', edgecolor='#B57400', lw=1.5)
    ax.add_patch(circle_sun)

    # Save figure
    plt.tight_layout()
    plt.savefig('photosynthesis_applications.png', dpi=300)
    # Also offer a PDF for vector quality
    plt.savefig('photosynthesis_applications.pdf')


if __name__ == '__main__':
    main()

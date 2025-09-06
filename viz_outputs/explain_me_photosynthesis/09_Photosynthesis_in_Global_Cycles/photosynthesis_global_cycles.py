import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def add_box(ax, x, y, w, h, color, text, text_color='white', fontsize=12, alpha=1.0):
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.05",
                         linewidth=1.5, edgecolor='black', facecolor=color, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', color=text_color, fontsize=fontsize, weight='bold', zorder=3)
    return box


def add_arrow(ax, start, end, color, label='', rad=0.0, lw=2.5, label_offset=(0.0, 0.0), alpha=1.0, ls='-'):
    arrow = FancyArrowPatch(start, end,
                            connectionstyle=f"arc3,rad={rad}",
                            arrowstyle='-|>',
                            mutation_scale=15,
                            lw=lw, color=color, alpha=alpha, linestyle=ls, zorder=1)
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2.0 + label_offset[0]
        my = (start[1] + end[1]) / 2.0 + label_offset[1]
        ax.text(mx, my, label, fontsize=10, color=color, ha='center', va='center', weight='bold', zorder=3)
    return arrow


def main():
    # Figure setup
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Colors
    col_CO2 = '#95a5a6'       # Atmospheric CO2 box
    col_O2 = '#85c1e9'        # Atmospheric O2 box
    col_auto = '#27ae60'      # Autotrophs box
    col_sink = '#1e8449'      # Biomass/Soils/Oceans sink box
    col_hetero = '#f39c12'    # Consumers/Decomposers box

    col_photo = '#2ecc71'     # Photosynthesis flow
    col_o2flow = '#3498db'    # O2-related flow
    col_resp = '#7f8c8d'      # Respiration/return flows
    col_food = '#e67e22'      # Food/energy flow

    # Boxes (x, y, w, h)
    CO2 = dict(x=0.8, y=8.2, w=3.6, h=1.1)
    O2  = dict(x=5.6, y=8.2, w=3.6, h=1.1)
    AUTO = dict(x=3.3, y=5.2, w=3.4, h=1.4)
    SINK = dict(x=0.7, y=1.3, w=3.8, h=1.5)
    HET  = dict(x=6.0, y=1.3, w=3.0, h=1.5)

    # Draw boxes
    add_box(ax, **CO2, color=col_CO2, text='Atmospheric CO2')
    add_box(ax, **O2, color=col_O2, text='Atmospheric O2', text_color='black')
    add_box(ax, **AUTO, color=col_auto, text='Photosynthesis\n(Autotrophs)')
    add_box(ax, **SINK, color=col_sink, text='Organic Carbon Reservoirs\n(Biomass, Soils, Oceans)')
    add_box(ax, **HET, color=col_hetero, text='Consumers &\nDecomposers', text_color='black')

    # Helper points (centers and edges)
    def center(box):
        return (box['x'] + box['w']/2, box['y'] + box['h']/2)
    def top_center(box):
        return (box['x'] + box['w']/2, box['y'] + box['h'])
    def bottom_center(box):
        return (box['x'] + box['w']/2, box['y'])
    def left_center(box):
        return (box['x'], box['y'] + box['h']/2)
    def right_center(box):
        return (box['x'] + box['w'], box['y'] + box['h']/2)

    # Key points
    p_CO2_bottom = (center(CO2)[0], CO2['y'])
    p_O2_bottom  = (center(O2)[0], O2['y'])
    p_AUTO_top   = top_center(AUTO)
    p_AUTO_bottom = bottom_center(AUTO)
    p_AUTO_left  = left_center(AUTO)
    p_AUTO_right = right_center(AUTO)
    p_SINK_top   = top_center(SINK)
    p_SINK_center = center(SINK)
    p_HET_top    = top_center(HET)
    p_HET_center = center(HET)

    # Arrows + labels
    # CO2 -> Autotrophs (Photosynthesis)
    add_arrow(ax, p_CO2_bottom, p_AUTO_top, color=col_photo,
              label='CO2 + H2O + light → sugars (carbon fixation)',
              rad=0.0, label_offset=(0.0, -0.5))

    # Autotrophs -> O2 (Oxygen release)
    add_arrow(ax, p_AUTO_top, p_O2_bottom, color=col_o2flow,
              label='O2 release', rad=0.0, label_offset=(0.0, 0.4))

    # Autotrophs -> Organic Carbon Reservoirs (Sequestration)
    add_arrow(ax, p_AUTO_bottom, p_SINK_top, color=col_photo,
              label='Carbon sequestration', rad=0.1, label_offset=(-0.4, -0.2))

    # Reservoirs -> CO2 (Respiration & Decomposition)
    add_arrow(ax, p_SINK_top, p_CO2_bottom, color=col_resp,
              label='Respiration & decomposition', rad=0.0, label_offset=(-0.1, 0.5))

    # Autotrophs -> Consumers/Decomposers (Food/Energy)
    add_arrow(ax, p_AUTO_right, p_HET_top, color=col_food,
              label='Food & energy to food webs', rad=-0.2, label_offset=(0.6, -0.2))

    # Consumers/Decomposers -> CO2 (Respiration)
    add_arrow(ax, p_HET_top, p_CO2_bottom, color=col_resp,
              label='Respiration', rad=-0.3, label_offset=(-0.2, 0.2))

    # Atmospheric O2 -> Consumers/Decomposers (O2 used in respiration)
    add_arrow(ax, p_O2_bottom, p_HET_top, color=col_o2flow,
              label='O2 used in respiration', rad=-0.15, label_offset=(0.2, 0.2), alpha=0.9)

    # Title and annotations
    ax.text(5, 9.7, 'Photosynthesis in Global Carbon and Oxygen Cycles',
            ha='center', va='center', fontsize=18, weight='bold')

    ax.text(5, 0.6,
            'Autotrophs capture atmospheric CO2 and, using sunlight, build organic matter while releasing O2.\n'
            'This fuels food webs and returns carbon via respiration and decomposition, helping regulate climate.',
            ha='center', va='center', fontsize=10, color='#2c3e50')

    # Sub-annotations for global impacts
    ax.text(1.1, 7.35, 'Climate regulation via CO2 drawdown', fontsize=9, color='#2c3e50')
    ax.text(8.9, 7.35, 'Supports aerobic life', fontsize=9, color='#2c3e50', ha='right')

    # Save figure
    outname = 'photosynthesis_global_cycles.png'
    plt.tight_layout()
    fig.savefig(outname, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {outname}')


if __name__ == '__main__':
    main()

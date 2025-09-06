import numpy as np
import matplotlib.pyplot as plt

# Relativity of Simultaneity — Minkowski Diagram
# This script creates a single, annotated figure illustrating that two events
# (lightning strikes at opposite ends of a moving train) can be simultaneous in one
# frame (ground, S) but not in another (train, S'). The diagram shows both frames'
# axes and highlights the differing simultaneity lines.

def main():
    # Physical/setup parameters (units with c = 1)
    c = 1.0
    beta = 0.6           # v/c (train moving to the right)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    L = 2.0              # Separation of lightning strikes in S (ground frame)

    # Events: A at x = -L/2, B at x = +L/2, both at t = 0 in S
    xA, xB = -L/2, L/2
    ctA, ctB = 0.0, 0.0

    # Lorentz transform to S' (train frame) for these events (t=0)
    # ct' = gamma*(ct - beta*x), x' = gamma*(x - beta*ct)
    ctA_p = gamma * (ctA - beta * xA)
    ctB_p = gamma * (ctB - beta * xB)

    # Plot ranges
    xlim = 2.2
    ylim = 2.2
    xs = np.linspace(-xlim, xlim, 400)
    cts = np.linspace(-ylim, ylim, 400)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Ground frame S axes (black)
    ax.plot([-xlim, xlim], [0, 0], color='black', lw=2)              # x-axis (ct=0)
    ax.plot([0, 0], [-ylim, ylim], color='black', lw=2)              # ct-axis

    # Light lines (45 degrees, speed of light) for visual reference
    ax.plot(xs, xs, color='gray', lw=1, ls='--', label='Light worldlines (c)')
    ax.plot(xs, -xs, color='gray', lw=1, ls='--')

    # Train frame S' axes (red)
    # ct' axis: worldline of x'=0 => x = beta * ct
    ct_vals = np.linspace(-ylim, ylim, 400)
    x_ctprime = beta * ct_vals
    ax.plot(x_ctprime, ct_vals, color='crimson', lw=2, label="S' axes")

    # x' axis: ct' = 0 => ct = beta * x
    x_vals = np.linspace(-xlim, xlim, 400)
    ct_xprime = beta * x_vals
    ax.plot(x_vals, ct_xprime, color='crimson', lw=2)

    # Simultaneity lines in each frame
    # S: t = 0 (already the x-axis). Emphasize with dotted style overlay.
    ax.plot([-xlim, xlim], [0, 0], color='black', lw=1.2, ls=':')

    # S': t' = 0 is the x' axis (ct = beta x). Emphasize with dotted overlay.
    ax.plot(x_vals, ct_xprime, color='crimson', lw=1.2, ls=':')

    # Events A and B at t=0 in S
    ax.scatter([xA, xB], [ctA, ctB], color='dodgerblue', s=70, zorder=3)
    ax.text(xA - 0.12, ctA - 0.18, 'A', color='dodgerblue', fontsize=12)
    ax.text(xB + 0.06, ctB - 0.18, 'B', color='dodgerblue', fontsize=12)

    # Annotate their S' times
    # Place small braces/projections toward ct' axis via dashed guides parallel to ct' axis.
    # A simple and clear way: label numeric t' values next to the points.
    ax.annotate(f"t' ≈ +{ctA_p:.2f}", xy=(xA, ctA), xytext=(xA - 1.1, 0.55),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2),
                color='crimson', fontsize=11, ha='right')
    ax.annotate(f"t' ≈ {ctB_p:.2f}", xy=(xB, ctB), xytext=(xB + 1.1, -0.55),
                arrowprops=dict(arrowstyle='->', color='crimson', lw=1.2),
                color='crimson', fontsize=11, ha='left')

    # Labels for axes
    ax.text(xlim - 0.15, -0.28, 'x (S)', fontsize=12, color='black', ha='right')
    ax.text(0.2, ylim - 0.15, 'ct (S)', fontsize=12, color='black', va='top')
    ax.text(beta * (ylim - 0.15) + 0.06, ylim - 0.2, "ct' (S')", fontsize=12, color='crimson')
    ax.text(xlim - 0.35, beta * (xlim - 0.35) + 0.08, "x' (S')", fontsize=12, color='crimson')

    # Explanatory text box
    info = (
        "Relativity of simultaneity:\n"
        "In S (ground), A and B are simultaneous (t=0, black line).\n"
        "In S' (train), simultaneity is along t'=0 (red line).\n"
        f"With v=0.6c and L={L}, Lorentz transform gives:\n"
        f"t'_A ≈ +{ctA_p:.2f}, t'_B ≈ {ctB_p:.2f} (units with c=1).\n"
        "Thus B occurs earlier than A in the train frame."
    )
    ax.text(-2.15, 1.2, info, fontsize=10, color='black', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='0.7'))

    # Cosmetic settings
    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('ct')
    ax.set_title('Relativity of Simultaneity (Minkowski Diagram)')
    ax.grid(True, which='both', ls=':', lw=0.6, color='0.85')

    # Legend: indicate which lines correspond to S' and light
    # Create custom handles by plotting invisible lines (to keep legend concise)
    light_handle, = ax.plot([], [], color='gray', lw=1, ls='--', label='Light worldlines (c)')
    Sprime_handle, = ax.plot([], [], color='crimson', lw=2, label="S' axes & simultaneity (t'=0)")
    S_handle, = ax.plot([], [], color='black', lw=2, label='S axes & simultaneity (t=0)')
    ax.legend(handles=[S_handle, Sprime_handle, light_handle], loc='lower right', frameon=True)

    out_name = 'relativity_of_simultaneity.png'
    plt.tight_layout()
    plt.savefig(out_name, dpi=200)
    print(f'Saved figure to {out_name}')


if __name__ == '__main__':
    main()

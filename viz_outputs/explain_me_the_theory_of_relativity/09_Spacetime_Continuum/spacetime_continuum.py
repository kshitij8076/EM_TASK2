import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D projection


def embed_z(x, y, depth=1.2, softening=0.7):
    r = np.sqrt(x**2 + y**2)
    return -depth / (r + softening)


def main():
    # Figure setup
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25], wspace=0.25)

    # Left: Minkowski spacetime diagram (ct vs x)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("Minkowski spacetime: events and light cone", fontsize=11)
    ax0.set_xlabel("Space (x)")
    ax0.set_ylabel("Time (ct)")

    # Limits and aspect for 45-degree light-cone lines
    xlim = (-1.6, 1.6)
    ylim = (-1.6, 1.6)
    ax0.set_xlim(*xlim)
    ax0.set_ylim(*ylim)
    ax0.set_aspect('equal', adjustable='box')
    ax0.grid(alpha=0.3, lw=0.5)

    # Axes lines
    ax0.axhline(0, color='k', lw=0.8, alpha=0.6)
    ax0.axvline(0, color='k', lw=0.8, alpha=0.6)

    # Light cone lines y = ±x
    xs = np.linspace(xlim[0], xlim[1], 400)
    lc_color = 'tab:orange'
    ax0.plot(xs, xs, color=lc_color, lw=2, label='Light cone (c)')
    ax0.plot(xs, -xs, color=lc_color, lw=2)

    # Event at origin
    ax0.scatter([0], [0], s=60, color='crimson', zorder=5, label='Event')

    # Timelike worldline (inertial)
    yline = np.linspace(ylim[0], ylim[1], 300)
    x_world = np.full_like(yline, 0.5)
    ax0.plot(x_world, yline, color='tab:blue', lw=2, label='Worldline (timelike)')

    # Accelerated worldline (curved but inside light cone)
    ycurve = np.linspace(ylim[0], ylim[1], 400)
    xcurve = 0.6 * np.tanh(1.2 * ycurve)
    ax0.plot(xcurve, ycurve, color='tab:green', lw=2, label='Accelerated worldline')

    # Annotation for light cone
    ax0.annotate('Light cone', xy=(1.0, 1.0), xytext=(0.5, 1.35),
                 arrowprops=dict(arrowstyle='->', color=lc_color, lw=1.2),
                 color=lc_color)

    ax0.legend(loc='upper left', fontsize=9, frameon=True)

    # Right: Curved spacetime (2D spatial slice, embedded in 3D)
    ax1 = fig.add_subplot(gs[0, 1], projection='3d')
    ax1.set_title("Curved spacetime near mass: warping and paths", fontsize=11)

    # Grid and surface (embedding diagram)
    n = 90
    gx = np.linspace(-3, 3, n)
    gy = np.linspace(-3, 3, n)
    X, Y = np.meshgrid(gx, gy)
    Z = embed_z(X, Y)

    # Wireframe for clarity
    ax1.plot_wireframe(X, Y, Z, rstride=3, cstride=3, color='0.4', linewidth=0.5, alpha=0.7)

    # Mass at the center (warping spacetime)
    z0 = embed_z(0.0, 0.0)
    ax1.scatter([0], [0], [z0], color='k', s=50, label='Mass (warps spacetime)')

    # Geodesic-like orbit (approximate) around the mass
    theta = np.linspace(0, 2 * np.pi, 500)
    r_orbit = 2.0
    xo = r_orbit * np.cos(theta)
    yo = r_orbit * np.sin(theta)
    zo = embed_z(xo, yo)
    ax1.plot(xo, yo, zo, color='tab:blue', lw=2.2, label='Geodesic (orbit)')

    # Light path bent by gravity (heuristic deflection)
    xl = np.linspace(-3, 3, 500)
    b = 1.2   # impact parameter
    k = 0.6   # deflection strength (illustrative)
    a = 1.0   # smoothing
    yl = b - k * xl / (xl**2 + a**2)
    zl = embed_z(xl, yl)
    ax1.plot(xl, yl, zl, color='tab:orange', lw=2.2, label="Light path (bent)")

    # Axis labels and view
    ax1.set_xlabel('x (space)')
    ax1.set_ylabel('y (space)')
    ax1.set_zlabel('embedding height (curvature)')
    ax1.view_init(elev=28, azim=-60)

    # Limits for a clean view
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_zlim(np.min(Z), -0.2)

    ax1.legend(loc='upper left', fontsize=9, frameon=True)

    # Overall title
    fig.suptitle('Spacetime Continuum: unified spacetime and curvature from mass-energy', fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    out_name = 'spacetime_continuum.png'
    plt.savefig(out_name, dpi=200, bbox_inches='tight')
    # Optionally display (commented out for script usage)
    # plt.show()


if __name__ == '__main__':
    main()

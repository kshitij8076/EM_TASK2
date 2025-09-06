import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)


def draw_precession_schematic(ax):
    # Parameters for the top and geometry
    theta = np.deg2rad(25)   # tilt of the top's axis
    phi = np.deg2rad(40)     # current azimuthal orientation
    axis_length = 1.0
    com_length = 0.6         # COM along the axis from pivot

    # Unit axis direction (tilted and rotated around z by phi)
    u = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    O = np.array([0.0, 0.0, 0.0])           # pivot point
    tip = O + axis_length * u                # tip of the axis
    com = O + com_length * u                 # center of mass position

    # Vectors: r (pivot->COM), mg (downward), tau = r x mg, L ~ along axis
    r_vec = com - O
    mg_vec = np.array([0.0, 0.0, -0.6])      # scaled for illustration
    tau_vec_raw = np.cross(r_vec, mg_vec)
    tau_dir = tau_vec_raw / (np.linalg.norm(tau_vec_raw) + 1e-12)
    tau_vec = 0.5 * tau_dir                  # scaled torque arrow
    L_vec = 0.8 * u

    # Draw ground reference circle at z = 0
    Rg = 1.2
    t_g = np.linspace(0, 2*np.pi, 200)
    ax.plot(Rg*np.cos(t_g), Rg*np.sin(t_g), np.zeros_like(t_g), color='#cccccc', lw=1)

    # Draw vertical (gravity) axis
    ax.plot([0, 0], [0, 0], [0, 1.2], color='k', lw=1.2)
    ax.text3D(0.05, 0.05, 1.18, 'vertical (gravity)', color='k')

    # Draw precession path: circle traced by the tip of the axis
    R = axis_length * np.sin(theta)
    zc = axis_length * np.cos(theta)
    t = np.linspace(0, 2*np.pi, 400)
    xc, yc, zc_arr = R*np.cos(t), R*np.sin(t), zc*np.ones_like(t)
    ax.plot(xc, yc, zc_arr, ls='--', color='0.5', lw=1.5, label='Precession path')
    ax.text3D(R*1.05, 0, zc+0.03, 'precession path', color='0.3')

    # Draw the top's axis
    ax.plot([O[0], tip[0]], [O[1], tip[1]], [O[2], tip[2]], color='#444444', lw=3)

    # Mark the tip
    ax.scatter([tip[0]], [tip[1]], [tip[2]], color='#444444', s=18)

    # Draw vectors with quivers
    # Angular momentum L (along axis)
    ax.quiver(O[0], O[1], O[2], L_vec[0], L_vec[1], L_vec[2], color='#1f77b4', lw=2, arrow_length_ratio=0.12)
    L_label_pos = L_vec * 1.1
    ax.text3D(L_label_pos[0], L_label_pos[1], L_label_pos[2], 'L (angular momentum)', color='#1f77b4')

    # Lever arm r (pivot to COM)
    ax.quiver(O[0], O[1], O[2], r_vec[0], r_vec[1], r_vec[2], color='#2ca02c', lw=2, arrow_length_ratio=0.12)
    r_mid = r_vec * 0.55
    ax.text3D(r_mid[0], r_mid[1], r_mid[2]+0.04, 'r', color='#2ca02c')

    # Weight mg at COM (downward)
    ax.quiver(com[0], com[1], com[2], mg_vec[0], mg_vec[1], mg_vec[2], color='black', lw=2, arrow_length_ratio=0.12)
    mg_tip = com + mg_vec
    ax.text3D(mg_tip[0]+0.03, mg_tip[1], mg_tip[2]-0.02, 'mg', color='black')

    # Torque tau about the pivot
    ax.quiver(O[0], O[1], O[2], tau_vec[0], tau_vec[1], tau_vec[2], color='#d62728', lw=2, arrow_length_ratio=0.12)
    tau_text_pos = tau_vec * 1.1
    ax.text3D(tau_text_pos[0], tau_text_pos[1], tau_text_pos[2]+0.02, 'τ = r × mg', color='#d62728')

    # Precession direction arrow (tangent to tip circle at current azimuth phi)
    # Point on the circle matching the axis azimuth
    x_tip_c = R * np.cos(phi)
    y_tip_c = R * np.sin(phi)
    z_tip_c = zc
    # Tangent vector to the circle at phi
    tan = np.array([-R*np.sin(phi), R*np.cos(phi), 0.0])
    tan_dir = tan / (np.linalg.norm(tan) + 1e-12)
    tan_vec = 0.35 * tan_dir
    ax.quiver(x_tip_c, y_tip_c, z_tip_c, tan_vec[0], tan_vec[1], tan_vec[2], color='#ff7f0e', lw=2, arrow_length_ratio=0.25)
    ax.text3D(x_tip_c + 0.1*tan_dir[0], y_tip_c + 0.1*tan_dir[1], z_tip_c + 0.02, 'Ω_p (precession)', color='#ff7f0e')

    # Cosmetic 3D settings
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_zlim(0.0, 1.25)
    ax.set_box_aspect((1, 1, 0.8))
    ax.view_init(elev=22, azim=-55)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_title('Spinning top precession (schematic)', pad=10)


def draw_rate_vs_L(ax):
    # Relationship: Ω_p ≈ τ / L (for steady precession, τ fixed)
    tau_mag = 1.0
    L = np.linspace(0.2, 4.0, 400)
    Omega_p = tau_mag / L

    ax.plot(L, Omega_p, color='#6a3d9a', lw=2)
    ax.set_xlabel('L (angular momentum magnitude)')
    ax.set_ylabel('Ω_p (precession rate)')
    ax.set_title('Precession rate falls as L increases (τ fixed)')

    # Highlight two example points
    L_fast = 0.5
    L_slow = 3.0
    Ofast = tau_mag / L_fast
    Oslow = tau_mag / L_slow
    ax.scatter([L_fast, L_slow], [Ofast, Oslow], color='#6a3d9a', zorder=3)
    ax.annotate('small L → faster precession', xy=(L_fast, Ofast), xytext=(1.0, Ofast+0.8),
                arrowprops=dict(arrowstyle='->', color='0.3'), color='0.2')
    ax.annotate('large L → slower precession', xy=(L_slow, Oslow), xytext=(2.1, Oslow+0.6),
                arrowprops=dict(arrowstyle='->', color='0.3'), color='0.2')

    ax.text(0.25, max(Omega_p)*0.9, 'Ω_p ≈ τ / L', color='#6a3d9a')
    ax.grid(True, alpha=0.25)


def main():
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.8])
    ax3d = fig.add_subplot(gs[0, 0], projection='3d')
    ax2d = fig.add_subplot(gs[0, 1])

    draw_precession_schematic(ax3d)
    draw_rate_vs_L(ax2d)

    fig.suptitle('Precession of a Spinning Top: axis wobbles (precesses) under torque; rate depends on L', fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])

    outname = 'precession_spinning_top.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')
    print(f'Saved figure to {outname}')


if __name__ == '__main__':
    main()

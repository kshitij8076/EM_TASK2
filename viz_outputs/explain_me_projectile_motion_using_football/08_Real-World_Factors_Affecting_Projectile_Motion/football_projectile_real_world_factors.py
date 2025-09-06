import numpy as np
import matplotlib.pyplot as plt


def simulate_trajectory(v0, launch_angle_deg, rho=1.225, Cd=0.25, A=0.038, m=0.43,
                        CL=0.0, spin_axis=None, wind=np.zeros(3), g=9.81,
                        dt=0.005, t_max=8.0):
    """
    Simulate a football's flight in 3D with gravity, quadratic drag, Magnus (spin), and wind.
    Coordinates: x (downfield), y (lateral), z (vertical).

    Parameters
    - v0: initial speed (m/s)
    - launch_angle_deg: elevation angle above horizontal (degrees)
    - rho: air density (kg/m^3)
    - Cd: drag coefficient (dimensionless)
    - A: cross-sectional area (m^2)
    - m: mass (kg)
    - CL: lift (Magnus) coefficient (dimensionless). Set >0 to enable spin lift.
    - spin_axis: 3-vector giving spin axis direction (arbitrary magnitude). If None -> no spin.
                 Only direction matters; CL controls strength.
    - wind: 3-vector wind velocity of air relative to ground (m/s)
    - g: gravitational acceleration (m/s^2)
    - dt: integration time step (s)
    - t_max: safety maximum time (s)

    Returns arrays x, y, z of the trajectory until ground impact (z=0) via linear interpolation.
    """
    # Initial state
    theta = np.deg2rad(launch_angle_deg)
    v = np.array([v0 * np.cos(theta), 0.0, v0 * np.sin(theta)], dtype=float)
    r = np.array([0.0, 0.0, 0.0], dtype=float)

    # Precompute constants
    k_drag = 0.5 * rho * Cd * A / m  # drag coefficient for acceleration

    # Normalize spin axis if provided
    omega_hat = None
    if spin_axis is not None:
        sa_norm = np.linalg.norm(spin_axis)
        if sa_norm > 0:
            omega_hat = spin_axis / sa_norm

    xs, ys, zs = [r[0]], [r[1]], [r[2]]

    t = 0.0
    last_r = r.copy()
    last_v = v.copy()
    while t < t_max:
        v_rel = v - wind
        speed_rel = np.linalg.norm(v_rel)

        # Gravity
        a_g = np.array([0.0, 0.0, -g])

        # Quadratic drag
        if rho > 0.0 and speed_rel > 1e-9:
            a_d = -k_drag * speed_rel * v_rel
        else:
            a_d = np.zeros(3)

        # Magnus (spin) lift: magnitude ~ 0.5*rho*A*CL*v^2/m, direction ~ (omega_hat x v_hat)
        a_m = np.zeros(3)
        if (rho > 0.0) and (CL > 0.0) and (omega_hat is not None) and (speed_rel > 1e-9):
            v_hat = v_rel / speed_rel
            n = np.cross(omega_hat, v_hat)
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-9:
                n_hat = n / n_norm
                a_m = (0.5 * rho * A * CL * (speed_rel ** 2) / m) * n_hat

        a = a_g + a_d + a_m

        # Integrate (explicit Euler)
        last_r[:] = r
        last_v[:] = v
        v = v + a * dt
        r = r + v * dt
        t += dt

        xs.append(r[0])
        ys.append(r[1])
        zs.append(r[2])

        # Ground impact detection and linear interpolation to z=0
        if r[2] < 0.0 and len(zs) > 1:
            z_prev = zs[-2]
            z_curr = zs[-1]
            if (z_prev - z_curr) != 0:
                frac = z_prev / (z_prev - z_curr)
                x_prev, y_prev = xs[-2], ys[-2]
                x_curr, y_curr = xs[-1], ys[-1]
                x_hit = x_prev + frac * (x_curr - x_prev)
                y_hit = y_prev + frac * (y_curr - y_prev)
                xs[-1] = x_hit
                ys[-1] = y_hit
                zs[-1] = 0.0
            break

    return np.array(xs), np.array(ys), np.array(zs)


def main():
    # Physical parameters for a standard football (soccer ball)
    m = 0.43              # kg
    r_ball = 0.11         # m radius
    A = np.pi * r_ball**2 # cross-sectional area
    Cd = 0.25             # drag coefficient

    # Environments
    rho_sea = 1.225  # kg/m^3 (sea level)
    rho_alt = 0.90   # kg/m^3 (high altitude ~ 3000 m)

    # Launch conditions
    v0 = 25.0         # m/s (strong kick)
    angle = 35.0      # degrees

    # Spin setup: sidespin -> spin axis vertical (z), curves sideways (y)
    spin_axis_vertical = np.array([0.0, 0.0, 1.0])
    CL_spin = 0.05  # modest Magnus lift coefficient for a curving kick

    # Wind setups
    wind_calm = np.array([0.0, 0.0, 0.0])
    wind_cross = np.array([0.0, 5.0, 0.0])    # 5 m/s crosswind to the right (+y)

    # Simulations
    # Panel A: Ideal (vacuum) vs with air drag (sea level)
    x_vac, y_vac, z_vac = simulate_trajectory(v0, angle, rho=0.0, Cd=Cd, A=A, m=m, CL=0.0,
                                               spin_axis=None, wind=wind_calm)
    x_drag, y_drag, z_drag = simulate_trajectory(v0, angle, rho=rho_sea, Cd=Cd, A=A, m=m, CL=0.0,
                                                 spin_axis=None, wind=wind_calm)

    # Panel B: Sea level vs high altitude
    x_alt, y_alt, z_alt = simulate_trajectory(v0, angle, rho=rho_alt, Cd=Cd, A=A, m=m, CL=0.0,
                                              spin_axis=None, wind=wind_calm)

    # Panel C: Spin (Magnus) vs no spin (top view)
    x_nospin, y_nospin, z_nospin = simulate_trajectory(v0, angle, rho=rho_sea, Cd=Cd, A=A, m=m, CL=0.0,
                                                       spin_axis=None, wind=wind_calm)
    x_spin, y_spin, z_spin = simulate_trajectory(v0, angle, rho=rho_sea, Cd=Cd, A=A, m=m, CL=CL_spin,
                                                 spin_axis=spin_axis_vertical, wind=wind_calm)

    # Panel D: Crosswind drift (top view)
    x_calm, y_calm, z_calm = x_nospin, y_nospin, z_nospin  # reuse calm, no spin
    x_wind, y_wind, z_wind = simulate_trajectory(v0, angle, rho=rho_sea, Cd=Cd, A=A, m=m, CL=0.0,
                                                 spin_axis=None, wind=wind_cross)

    # Create figure
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'legend.fontsize': 9
    })

    fig, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    # Panel A: Side view - Ideal vs Drag
    ax = axs[0, 0]
    ax.plot(x_vac, z_vac, color='k', linestyle='--', linewidth=2, label='Ideal (vacuum)')
    ax.plot(x_drag, z_drag, color='tab:red', linewidth=2, label='With air drag (sea level)')
    ax.set_title('A) Air Drag Lowers Range')
    ax.set_xlabel('Downfield distance x (m)')
    ax.set_ylabel('Height z (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.annotate('No air resistance → longer, symmetric parabola', xy=(0.55, 0.90), xycoords='axes fraction',
                fontsize=9, color='k')
    ax.annotate('Drag slows ball → shorter range', xy=(0.55, 0.80), xycoords='axes fraction',
                fontsize=9, color='tab:red')

    # Panel B: Side view - Altitude effect
    ax = axs[0, 1]
    ax.plot(x_drag, z_drag, color='tab:red', linewidth=2, label=f'Sea level (ρ={rho_sea:.3g} kg/m³)')
    ax.plot(x_alt, z_alt, color='tab:orange', linewidth=2, label=f'High altitude (ρ={rho_alt:.2f} kg/m³)')
    ax.set_title('B) Thinner Air Increases Range')
    ax.set_xlabel('Downfield distance x (m)')
    ax.set_ylabel('Height z (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    ax.annotate('Lower ρ → less drag', xy=(0.55, 0.85), xycoords='axes fraction', fontsize=9, color='tab:orange')

    # Panel C: Top view - Spin (Magnus effect)
    ax = axs[1, 0]
    ax.plot(x_nospin, y_nospin, color='0.4', linewidth=2, label='No spin (calm air)')
    ax.plot(x_spin, y_spin, color='tab:blue', linewidth=2, label='Sidespin (Magnus)')
    ax.set_title('C) Spin Curves the Path (Magnus Effect)')
    ax.set_xlabel('Downfield distance x (m)')
    ax.set_ylabel('Lateral displacement y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.annotate('Vertical spin axis → sideways lift', xy=(0.45, 0.12), xycoords='axes fraction', fontsize=9, color='tab:blue')

    # Panel D: Top view - Crosswind drift
    ax = axs[1, 1]
    ax.plot(x_calm, y_calm, color='0.4', linewidth=2, label='Calm air')
    ax.plot(x_wind, y_wind, color='tab:purple', linewidth=2, label='Crosswind 5 m/s')
    ax.set_title('D) Wind Pushes the Ball Sideways')
    ax.set_xlabel('Downfield distance x (m)')
    ax.set_ylabel('Lateral displacement y (m)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax.annotate('Wind acts on air-relative speed', xy=(0.45, 0.12), xycoords='axes fraction', fontsize=9, color='tab:purple')

    # Global title
    fig.suptitle('Real-World Factors Affecting a Football\'s Projectile Motion', fontsize=14)

    # Save figure
    out_name = 'football_projectile_real_world_factors.png'
    plt.savefig(out_name, dpi=150)
    print(f'Saved figure to {out_name}')


if __name__ == '__main__':
    main()

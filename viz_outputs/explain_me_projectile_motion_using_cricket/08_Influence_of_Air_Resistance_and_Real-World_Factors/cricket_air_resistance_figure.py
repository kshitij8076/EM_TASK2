import numpy as np
import matplotlib.pyplot as plt


def simulate_with_drag(v0, angle_deg, m, Cd, A, rho, wind=(0.0, 0.0), g=9.81, dt=0.002, t_max=15.0):
    """
    Simulate 2D projectile motion with quadratic air drag and constant wind using RK4.
    Returns time, x, y arrays that end exactly at ground (y=0) by linear interpolation.
    """
    th = np.deg2rad(angle_deg)
    vx0 = v0 * np.cos(th)
    vy0 = v0 * np.sin(th)

    def deriv(state):
        x, y, vx, vy = state
        v_rel_x = vx - wind[0]
        v_rel_y = vy - wind[1]
        speed_rel = np.hypot(v_rel_x, v_rel_y)
        # Drag force vector (opposes relative motion)
        Fx = -0.5 * rho * Cd * A * speed_rel * v_rel_x
        Fy = -0.5 * rho * Cd * A * speed_rel * v_rel_y
        ax = Fx / m
        ay = Fy / m - g
        return np.array([vx, vy, ax, ay])

    # Initialize
    t = 0.0
    state = np.array([0.0, 0.0, vx0, vy0])
    Ts = [t]
    Xs = [state[0]]
    Ys = [state[1]]

    while t < t_max:
        # RK4 step
        k1 = deriv(state)
        k2 = deriv(state + 0.5 * dt * k1)
        k3 = deriv(state + 0.5 * dt * k2)
        k4 = deriv(state + dt * k3)
        state_next = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        t_next = t + dt

        # Record
        Ts.append(t_next)
        Xs.append(state_next[0])
        Ys.append(state_next[1])

        # Check ground crossing
        if Ys[-2] >= 0.0 and Ys[-1] < 0.0:
            # Linear interpolation between the last two points to find exact landing
            y1, y2 = Ys[-2], Ys[-1]
            x1, x2 = Xs[-2], Xs[-1]
            t1, t2 = Ts[-2], Ts[-1]
            frac = y1 / (y1 - y2 + 1e-12)
            x_land = x1 + frac * (x2 - x1)
            t_land = t1 + frac * (t2 - t1)
            # Replace the last point with the interpolated landing point at y=0
            Ts[-1] = t_land
            Xs[-1] = x_land
            Ys[-1] = 0.0
            break

        # Prepare next step
        state = state_next
        t = t_next

    return np.array(Ts), np.array(Xs), np.array(Ys)


def ideal_projectile(v0, angle_deg, g=9.81, num=400):
    th = np.deg2rad(angle_deg)
    t_f = 2 * v0 * np.sin(th) / g
    t = np.linspace(0, t_f, num)
    x = v0 * np.cos(th) * t
    y = v0 * np.sin(th) * t - 0.5 * g * t**2
    return t, x, y, t_f


def main():
    # Physical parameters for a cricket ball
    g = 9.81
    m = 0.156  # kg
    r = 0.036  # m (approx radius)
    A = np.pi * r**2  # cross-sectional area
    rho = 1.225  # kg/m^3 (sea level, dry air)
    Cd_smooth = 0.45  # representative for smoother/shiny side

    # Launch conditions (typical strong throw/shot)
    v0 = 35.0  # m/s
    angle = 35.0  # degrees

    # Compute ideal (no drag)
    t_id, x_id, y_id, t_id_land = ideal_projectile(v0, angle, g=g)
    R_id = x_id[-1]

    # Drag cases
    cases = [
        {
            'label': 'Drag (no wind)',
            'wind': (0.0, 0.0),
            'Cd': Cd_smooth,
            'rho': rho,
            'color': 'tab:blue',
            'z': 3
        },
        {
            'label': 'Drag + headwind 5 m/s',
            'wind': (-5.0, 0.0),  # air moving opposite to ball flight
            'Cd': Cd_smooth,
            'rho': rho,
            'color': 'tab:red',
            'z': 3
        },
        {
            'label': 'Drag + tailwind 5 m/s',
            'wind': (5.0, 0.0),  # air moving with the ball
            'Cd': Cd_smooth,
            'rho': rho,
            'color': 'tab:green',
            'z': 3
        }
    ]

    sim_results = []
    for case in cases:
        t, x, y = simulate_with_drag(v0, angle, m, case['Cd'], A, case['rho'], wind=case['wind'], g=g, dt=0.002, t_max=15.0)
        sim_results.append({
            'label': case['label'],
            'x': x,
            'y': y,
            't': t,
            'color': case['color']
        })

    # Prepare figure
    plt.rcParams.update({
        'figure.dpi': 120,
        'font.size': 11
    })
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot ideal curve
    ax.plot(x_id, y_id, linestyle='--', color='gray', linewidth=2.0, label='Ideal (no air resistance)', zorder=2)

    # Plot drag curves and landing markers
    landing_points = []
    for res in sim_results:
        ax.plot(res['x'], res['y'], color=res['color'], linewidth=2.2, label=res['label'], zorder=3)
        x_land = res['x'][-1]
        t_land = res['t'][-1]
        landing_points.append((res['label'], x_land, t_land, res['color']))
        ax.plot([x_land], [0.0], marker='o', color=res['color'], markersize=6, zorder=4)

    # Ideal landing marker
    ax.plot([R_id], [0.0], marker='s', color='gray', markersize=6, zorder=3)

    # Ground line
    xmax_all = max([R_id] + [lp[1] for lp in landing_points])
    ax.hlines(0, 0, xmax_all * 1.05, colors='k', linestyles='-', linewidth=0.8, alpha=0.4)

    # Annotate shorter range and flight time (ideal vs drag no wind)
    # find the no-wind case landing
    x_land_nowind = [lp[1] for lp in landing_points if 'no wind' in lp[0]][0]
    t_land_nowind = [lp[2] for lp in landing_points if 'no wind' in lp[0]][0]

    ax.annotate('', xy=(x_land_nowind, 0.0), xytext=(R_id, 0.0),
                arrowprops=dict(arrowstyle='<->', color='black', lw=1.2))
    mid_x = 0.5 * (x_land_nowind + R_id)
    ax.text(mid_x, 0.8, 'Range reduced by drag', ha='center', va='bottom')

    # Annotate steeper descent on drag (no wind) curve
    # pick a point late in flight on the no-wind drag trajectory
    nowind_curve = [res for res in sim_results if res['label'] == 'Drag (no wind)'][0]
    idx = int(0.75 * len(nowind_curve['x']))
    ax.annotate('Steeper downward path
with air resistance',
                xy=(nowind_curve['x'][idx], nowind_curve['y'][idx]),
                xytext=(nowind_curve['x'][idx] - 35, nowind_curve['y'][idx] + 8),
                arrowprops=dict(arrowstyle='->', color='tab:blue', lw=1.2),
                color='tab:blue', ha='left', va='bottom')

    # Add small text notes about real-world factors
    ax.text(xmax_all * 0.55, 13,
            'Wind alters relative air speed (headwind reduces range,\n'
            'tailwind increases it). Humidity and ball roughness\n'
            'change air density and drag coefficient, shifting curves.',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, ec='gray'))

    # Axes labels and title
    ax.set_title('Cricket Ball Trajectories: Effect of Air Resistance and Wind')
    ax.set_xlabel('Horizontal distance (m)')
    ax.set_ylabel('Height (m)')

    # Limits
    ax.set_xlim(0, max(130, xmax_all * 1.05))
    ax.set_ylim(0, 40)

    # Legend with landing info
    legend_labels = ['Ideal (no air resistance)']
    for label, xL, tL, color in landing_points:
        legend_labels.append(f"{label} — range: {xL:.1f} m, time: {tL:.2f} s")
    ax.legend(legend_labels, loc='upper right', framealpha=0.95)

    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)

    # Save figure
    outfile = 'cricket_projectile_air_resistance.png'
    plt.tight_layout()
    plt.savefig(outfile, dpi=200)
    print(f'Figure saved to {outfile}')


if __name__ == '__main__':
    main()

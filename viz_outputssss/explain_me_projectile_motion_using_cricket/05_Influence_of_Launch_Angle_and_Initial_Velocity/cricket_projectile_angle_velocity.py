import numpy as np
import matplotlib.pyplot as plt

# Projectile motion utilities
G = 9.81  # m/s^2

def trajectory(v0, angle_deg, g=G, n=400):
    theta = np.deg2rad(angle_deg)
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, n)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    return x, y

def apex(v0, angle_deg, g=G):
    theta = np.deg2rad(angle_deg)
    t_peak = v0 * np.sin(theta) / g
    x_peak = v0 * np.cos(theta) * t_peak
    y_peak = v0 * np.sin(theta) * t_peak - 0.5 * g * t_peak**2
    return x_peak, y_peak

def range_ideal(v0, angle_deg, g=G):
    theta = np.deg2rad(angle_deg)
    return (v0**2) * np.sin(2 * theta) / g

if __name__ == "__main__":
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    # Left: Effect of launch angle at fixed speed
    ax0 = axes[0]
    v0_fixed = 35  # m/s, typical big hit/throw speed
    angles = [15, 30, 45, 60]

    y_peaks = []
    ranges = []
    lines = []
    for ang in angles:
        x, y = trajectory(v0_fixed, ang)
        line, = ax0.plot(x, y, lw=2, label=f"{ang}\N{DEGREE SIGN}")
        lines.append(line)
        xp, yp = apex(v0_fixed, ang)
        y_peaks.append(yp)
        ranges.append(x[-1])
        ax0.plot(xp, yp, 'o', color=line.get_color(), ms=4)

    # Highlight max range at 45 degrees (idealized)
    R45 = range_ideal(v0_fixed, 45)
    ymax = max(y_peaks) if y_peaks else 1
    ax0.axvline(R45, ls='--', color='gray', alpha=0.5)
    ax0.annotate(
        "Farthest range at 45\N{DEGREE SIGN} (ideal)",
        xy=(R45, 0), xytext=(R45 * 0.55, ymax * 0.6),
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center', va='center', color='dimgray'
    )

    # Ground line
    ax0.axhline(0, color='k', lw=1)

    ax0.set_title("Varying launch angle (v0 = 35 m/s)")
    ax0.set_xlabel("Horizontal distance (m)")
    ax0.set_ylabel("Height (m)")
    ax0.grid(True, ls=':', alpha=0.6)

    # Limits with margin
    ax0.set_xlim(0, max(ranges) * 1.08)
    ax0.set_ylim(0, max(y_peaks) * 1.2)
    ax0.legend(title="Angle")

    # Right: Effect of initial speed at fixed angle (45 degrees)
    ax1 = axes[1]
    angle_fixed = 45
    speeds = [20, 30, 40]  # m/s

    y_peaks_r = []
    ranges_r = []
    for v in speeds:
        x, y = trajectory(v, angle_fixed)
        line, = ax1.plot(x, y, lw=2, label=f"v0 = {v} m/s")
        xp, yp = apex(v, angle_fixed)
        y_peaks_r.append(yp)
        ranges_r.append(x[-1])
        ax1.plot(xp, yp, 'o', color=line.get_color(), ms=4)

    # Annotation indicating effect of speed
    # Arrow from medium speed apex to high speed apex
    xp_med, yp_med = apex(speeds[1], angle_fixed)
    xp_hi, yp_hi = apex(speeds[2], angle_fixed)
    ax1.annotate(
        "Higher speed -> farther and higher",
        xy=(xp_hi, yp_hi), xytext=(xp_med * 0.6, yp_hi * 0.6),
        arrowprops=dict(arrowstyle='->'),
        ha='left', va='center'
    )

    ax1.axhline(0, color='k', lw=1)
    ax1.set_title("Varying initial speed (angle = 45\N{DEGREE SIGN})")
    ax1.set_xlabel("Horizontal distance (m)")
    ax1.grid(True, ls=':', alpha=0.6)

    ax1.set_xlim(0, max(ranges_r) * 1.08)
    ax1.set_ylim(0, max(y_peaks_r) * 1.2)
    ax1.legend(title="Initial speed")

    # Overall context title and note
    fig.suptitle("Cricket ball flight: how launch angle and speed shape distance and height")
    fig.text(0.5, 0.01, "Idealized: level ground, no air resistance or spin", ha='center', color='dimgray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_name = "cricket_projectile_launch_angle_velocity.png"
    plt.savefig(out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

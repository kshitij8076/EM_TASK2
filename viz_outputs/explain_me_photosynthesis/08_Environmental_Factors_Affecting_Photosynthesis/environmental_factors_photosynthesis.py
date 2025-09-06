import numpy as np
import matplotlib.pyplot as plt

# Generate illustrative response curves for key environmental factors
# All rates are normalized to a maximum of 1 for pedagogical clarity.

# 1) Light intensity (saturating response)
I = np.linspace(0, 2000, 500)  # μmol photons m⁻² s⁻¹
Ik = 300.0
rI = I / (Ik + I)

# 2) CO₂ concentration (saturating response)
C = np.linspace(0, 1200, 500)  # ppm
Kc = 200.0
rC = C / (Kc + C)

# 3) Temperature (peaked response with high-T denaturation)
T = np.linspace(0, 50, 500)  # °C
Topt = 28.0
sigma = 8.0
base_peak = np.exp(-0.5 * ((T - Topt) / sigma) ** 2)
heat_drop = 1.0 / (1.0 + np.exp((T - 38.0) / 1.5))  # sharp decline after ~38°C
rT = base_peak * heat_drop
rT /= rT.max()  # normalize to 1

# 4) Water availability (monotonic increase to saturation)
W = np.linspace(0, 100, 500)  # % relative soil water content
k = 0.12
w50 = 45.0
rW = 1.0 / (1.0 + np.exp(-k * (W - w50)))

# Build the figure
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.ravel()

for ax in axes:
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    ax.axhline(1.0, color='0.3', linestyle=':', linewidth=1)

# Light intensity subplot
ax = axes[0]
ax.set_title('Light intensity')
ax.axvspan(0, 400, color='grey', alpha=0.15)
ax.plot(I, rI, color='forestgreen', linewidth=2.5)
ax.set_xlabel('Light intensity (μmol photons m⁻² s⁻¹)')
ax.set_ylabel('Relative photosynthetic rate')
y200 = np.interp(200, I, rI)
ax.annotate('Light-limited', xy=(200, y200), xytext=(650, 0.35),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')
y1300 = np.interp(1300, I, rI)
ax.annotate('Saturation', xy=(1300, y1300), xytext=(900, 0.9),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')

# CO₂ concentration subplot
ax = axes[1]
ax.set_title('CO₂ concentration')
ax.axvspan(0, 300, color='grey', alpha=0.15)
ax.plot(C, rC, color='forestgreen', linewidth=2.5)
ax.set_xlabel('CO₂ concentration (ppm)')
y150 = np.interp(150, C, rC)
ax.annotate('CO₂-limited', xy=(150, y150), xytext=(450, 0.35),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')
y900 = np.interp(900, C, rC)
ax.annotate('Approaches max', xy=(900, y900), xytext=(600, 0.85),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')

# Temperature subplot
ax = axes[2]
ax.set_title('Temperature')
ax.axvspan(20, 30, color='lightgreen', alpha=0.25)
ax.axvspan(38, 50, color='tomato', alpha=0.15)
ax.plot(T, rT, color='forestgreen', linewidth=2.5)
ax.set_xlabel('Temperature (°C)')
yopt = np.interp(Topt, T, rT)
ax.annotate('Optimal range', xy=(Topt, yopt), xytext=(10, 0.9),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')
y40 = np.interp(40, T, rT)
ax.annotate('Heat stress / enzyme denaturation', xy=(40, y40), xytext=(31, 0.45),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')

# Water availability subplot
ax = axes[3]
ax.set_title('Water availability')
ax.axvspan(0, 40, color='grey', alpha=0.15)
ax.plot(W, rW, color='forestgreen', linewidth=2.5)
ax.set_xlabel('Relative soil water content (%)')
y20 = np.interp(20, W, rW)
ax.annotate('Water stress', xy=(20, y20), xytext=(55, 0.35),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')
y80 = np.interp(80, W, rW)
ax.annotate('Sufficient water', xy=(80, y80), xytext=(60, 0.85),
            arrowprops=dict(arrowstyle='->', color='grey'), color='grey')

fig.suptitle('Environmental factors affecting photosynthesis', fontsize=16, y=0.98)
fig.text(0.5, 0.01, 'Illustrative relationships (not to scale). Curves show how each factor can limit or optimize the photosynthetic rate.',
         ha='center', fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('photosynthesis_environmental_factors.png', dpi=300)

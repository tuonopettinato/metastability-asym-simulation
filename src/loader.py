"""
loader.py 
This script loads simulation results from .npy files and generates plots to visualize the data.
Plots will be saved in the 'loaded_results' directory and the script assumes the data to be
in the 'simulation_results/npy' directory.

It creates a 2x2 subplot figure with:
1. Neural currents of the first few neurons.
2. Firing rates of the first few neurons.
3. Memory pattern overlaps if available.
4. Control signal ζ(t) from the Ornstein-Uhlenbeck process.
It also saves individual plots for each subplot and a heatmap of firing rates for all neurons.

It assumes the following directory structure:
- src/loader.py
- simulation_results/npy/ (contains .npy files with simulation data)
- loaded_results/ (where the output plots will be saved)
Note that both directories simulation_results/npy and loaded_results are in the .gitignore file.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Set paths
base_dir = os.path.dirname(__file__)
npy_dir = os.path.join(base_dir, "..", "simulation_results", "npy")
output_dir = os.path.join(base_dir, "..", "loaded_results")
os.makedirs(output_dir, exist_ok=True)

# Load data
t = np.load(os.path.join(npy_dir, "time.npy"))
u = np.load(os.path.join(npy_dir, "neural_currents.npy"))
phi_u = np.load(os.path.join(npy_dir, "firing_rates.npy"))
zeta = np.load(os.path.join(npy_dir, "ou_process.npy"))
params = np.load(os.path.join(npy_dir, "simulation_parameters.npy"), allow_pickle=True).item()

N = params['N']
n_display = params.get('n_display', min(5, N))
p = params.get('p', 0)

# Try to load pattern overlaps if available
overlaps = None
if p > 0:
    overlaps_path = os.path.join(npy_dir, "pattern_overlaps.npy")
    if os.path.exists(overlaps_path):
        overlaps = np.load(overlaps_path)

# 2x2 subplot figure
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# 1. Neural Currents (top-left)
n_plot = min(n_display, N)
ax = axs[0, 0]
for i in range(n_plot):
    ax.plot(t, u[:, i], alpha=0.7, label=f'u_{i+1}')
ax.set_xlabel('Time')
ax.set_ylabel('Current')
ax.set_title(f'Neural Currents $u_i$ (first {n_plot} neurons)')
ax.grid(True)
if n_plot <= 5:
    ax.legend()

# 2. Firing Rates (top-right)
ax = axs[0, 1]
for i in range(n_plot):
    ax.plot(t, phi_u[:, i], alpha=0.7, label=f'φ(u_{i+1})')
ax.set_xlabel('Time')
ax.set_ylabel('FR')
ax.set_title(f'Firing Rates φ($u_i$) (first {n_plot} neurons)')
ax.grid(True)
if n_plot <= 5:
    ax.legend()

# 3. Pattern Overlaps (bottom-left)
ax = axs[1, 0]
if overlaps is not None:
    for i in range(overlaps.shape[1]):
        ax.plot(t, overlaps[:, i], label=f'Pattern {i+1}', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Pattern Overlap')
    ax.set_title('Memory Pattern Overlaps')
    ax.grid(True)
    ax.legend()
else:
    ax.text(0.5, 0.5, 'No patterns to display', ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Pattern Overlaps')

# 4. OU Process (bottom-right)
ax = axs[1, 1]
if np.all(zeta == zeta[0]):
    zeta_val = float(zeta[0])
    ax.axhline(y=zeta_val, color='r', linestyle='-', linewidth=2)
    ax.set_ylim([zeta_val - 0.1, zeta_val + 0.1])
    ax.set_title(f'Control Signal ζ(t) = {zeta_val:.2f} (constant)')
else:
    ax.plot(t, zeta, color='red', linewidth=1.5, label='ζ(t)')
    ax.set_title('Control Signal ζ(t)')
    ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('ζ(t)')
ax.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "complete_simulation_results_from_loader.png"), dpi=300)

# Save individual plots
titles = ['neural_currents', 'firing_rates', 'pattern_overlaps', 'ou_process']
for i, ax in enumerate(axs.flat):
    fig_single, ax_single = plt.subplots(figsize=(10, 6))
    for line in ax.get_lines():
        ax_single.plot(line.get_xdata(), line.get_ydata(),
                       label=line.get_label(), color=line.get_color(),
                       alpha=line.get_alpha() if line.get_alpha() is not None else 1.0,
                       linewidth=line.get_linewidth())
    ax_single.set_title(ax.get_title())
    ax_single.set_xlabel(ax.get_xlabel())
    ax_single.set_ylabel(ax.get_ylabel())
    ax_single.grid(True)
    if ax.get_legend() is not None:
        ax_single.legend()
    for text in ax.texts:
        ax_single.text(text.get_position()[0], text.get_position()[1],
                       text.get_text(), ha=text.get_ha(), va=text.get_va(),
                       transform=ax_single.transAxes if text.get_transform() == ax.transAxes else ax_single.transData)
    ax_single.set_xlim(ax.get_xlim())
    ax_single.set_ylim(ax.get_ylim())
    plt.tight_layout()
    fig_single.savefig(os.path.join(output_dir, f"{titles[i]}_from_loader.png"), dpi=300)
    plt.close(fig_single)

# Firing rates heatmap
plt.figure(figsize=(10, 6))
plt.imshow(phi_u.T, aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='FR')
plt.title(f'Firing Rates of all {N} neurons')
plt.xlabel('Time')
plt.ylabel('Neuron Index')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "firing_rates_heatmap_from_loader.png"), dpi=300)
plt.close()

print("Plots recreated from saved data in 'loaded_results'.")
plt.show()
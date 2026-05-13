import os
import numpy as np
import matplotlib.pyplot as plt

from parameters import N, multiple_dir_name, test_set

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
t_start = 6000
t_end = 6400
neuron_stride = 1

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")

if test_set:
    base_dir = os.path.join(base_dir, "..", "test_set", "npy")

firing_dir = os.path.join(base_dir, "firing_rates")
plots_dir = os.path.join(base_dir, "..", "plot_batch")

rates = np.load(os.path.join(firing_dir, "all_firing_rates.npy"))
overlaps = np.load(os.path.join(plots_dir, "batch_overlaps.npy"))

# ----------------------------------------------------
# SELECT WINDOW
# ----------------------------------------------------
rates_window = rates[t_start:t_end, ::neuron_stride]
overlaps_window = overlaps[t_start:t_end]

heatmap_data = rates_window.T
t = np.arange(t_start, t_end)

# ----------------------------------------------------
# FIGURE + GRIDSPEC
# ----------------------------------------------------
fig = plt.figure(figsize=(12,8))

gs = fig.add_gridspec(
    2, 2,
    width_ratios=[40,1],   # colonna stretta per colorbar
    height_ratios=[1,2]
)

ax_over = fig.add_subplot(gs[0,0])
ax_heat = fig.add_subplot(gs[1,0], sharex=ax_over)
cax = fig.add_subplot(gs[1,1])   # asse colorbar

# ----------------------------------------------------
# OVERLAPS
# ----------------------------------------------------
P = overlaps_window.shape[1]

for i in range(P):
    ax_over.plot(t, overlaps_window[:,i], label=f"P {i+1}")

ax_over.set_ylabel("Overlaps", fontsize=20)
ax_over.tick_params(axis='both', labelsize=18)
ax_over.set_xlim(t_start, t_end)

# ----------------------------------------------------
# HEATMAP
# ----------------------------------------------------
im = ax_heat.imshow(
    heatmap_data,
    aspect="auto",
    cmap="gray_r",
    origin="lower",
    extent=[t_start, t_end, 0, heatmap_data.shape[0]]
)

ax_heat.set_ylabel("Neuron", fontsize=20)
ax_heat.set_xlabel("t", fontsize=20)
ax_heat.tick_params(axis='both', labelsize=18)

# ----------------------------------------------------
# COLORBAR (separata)
# ----------------------------------------------------
cbar = fig.colorbar(im, cax=cax)
cbar.set_label("Firing rate", fontsize=16)

# ----------------------------------------------------
# SAVE
# ----------------------------------------------------
plt.tight_layout()

out_png = os.path.join(plots_dir, f"overlaps_activity_{t_start}_{t_end}.png")
plt.savefig(out_png, dpi=300)

plt.show()
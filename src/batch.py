"""
This script processes multiple simulation runs by loading their firing rates,
computing pattern overlaps, concatenating the results, and saving both the
concatenated firing rates and trial boundaries. It also generates an HTML
plot of the overlaps using Plotly.
"""

import os
import numpy as np
import plotly.graph_objects as go

from modules.dynamics import calculate_pattern_overlaps
from parameters import (
    N, dt, phi_beta, phi_r_m, phi_x_r,
    g_q, g_x, use_numba, use_g, multiple_dir_name, test_set
)

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
FILES = None          # se None → usa tutti i firing_rates_*.npy
pivot = 150           # se 0 o None → usa boundaries reali

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------

base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
patterns_path = os.path.join(base_dir, "memory_patterns.npy")
if test_set: 
    base_dir = os.path.join(base_dir, "test_set")
else:
    base_dir = base_dir
firing_dir = os.path.join(base_dir, "firing_rates")

plots_dir = os.path.join(base_dir, "..", "plot_batch")
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------------------------------
# LOAD PATTERNS
# ----------------------------------------------------
eta = np.load(patterns_path)
phi_params = {"r_m": phi_r_m, "beta": phi_beta, "x_r": phi_x_r}
g_params = {"q_f": g_q, "x_f": g_x}

# ----------------------------------------------------
# DETECT FILES IF NEEDED
# ----------------------------------------------------
if FILES is None:
    detected = []
    for name in os.listdir(firing_dir):
        if name.startswith("firing_rates_") and name.endswith(".npy"):
            try:
                idx = int(name[len("firing_rates_"):-4])
                detected.append(idx)
            except ValueError:
                pass

    FILES = sorted(detected)
    print(f"Detected files: {FILES}")

if len(FILES) == 0:
    raise RuntimeError("No firing_rates_*.npy files found in directory.")

# ----------------------------------------------------
# CONCATENATION STORAGE
# ----------------------------------------------------
all_rates = []
all_overlaps = []
boundaries = []   # per boundaries reali

current_start = 0
current_end = -1

# ----------------------------------------------------
# PROCESS EACH FILE
# ----------------------------------------------------
for idx in FILES:
    print(f"\nProcessing file {idx} ...")

    path = os.path.join(firing_dir, f"firing_rates_{idx}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    rates = np.load(path)  # shape (T, N)
    overlaps = calculate_pattern_overlaps(
        rates, eta, phi_params, g_params,
        use_numba=use_numba, use_g=use_g
    )

    T = rates.shape[0]

    all_rates.append(rates)
    all_overlaps.append(overlaps)

    if not pivot:
        current_start = current_end + 1
        current_end = current_start + T - 1
        boundaries.append((current_start, current_end))

    print(f"Loaded {T} timesteps.")

# ----------------------------------------------------
# BUILD CONCATENATED MATRICES
# ----------------------------------------------------
all_rates = np.concatenate(all_rates, axis=0)
all_overlaps = np.concatenate(all_overlaps, axis=0)

total_T = all_rates.shape[0]
print(f"\nTotal concatenated timesteps: {total_T}")

# ----------------------------------------------------
# SAVE CONCATENATED FIRING RATES
# ----------------------------------------------------
out_batch = os.path.join(firing_dir, "all_firing_rates.npy")
np.save(out_batch, all_rates.astype(np.float32))
print(f"Saved concatenated firing rates to {out_batch}")

# ----------------------------------------------------
# SAVE TRIAL BOUNDARIES
# ----------------------------------------------------
txt_path = os.path.join(firing_dir, "trials.txt")
with open(txt_path, "w") as f:

    if pivot:   # intervalli regolari
        start = 0
        while start < total_T:
            end = min(start + pivot - 1, total_T - 1)
            f.write(f"{start} {end} 0\n")
            start = end + 1

    else:       # boundaries reali per ogni file
        for s, e in boundaries:
            f.write(f"{s} {e} 0\n")

print(f"Saved trial boundaries to {txt_path}")

# ----------------------------------------------------
# PLOTLY PLOT OF OVERLAPS
# ----------------------------------------------------
time = np.arange(total_T)
P = all_overlaps.shape[1]

fig = go.Figure()

for p in range(P):
    fig.add_trace(go.Scatter(
        x=time,
        y=all_overlaps[:, p],
        mode="lines",
        name=f"Pattern {p}"
    ))

fig.update_layout(
    title="Batch Overlaps (All Runs)",
    xaxis_title="Time step",
    yaxis_title="Overlap",
    height=700,
    width=1200
)

html_path = os.path.join(plots_dir, "batch_overlaps.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print(f"\nSaved Plotly HTML to {html_path}")

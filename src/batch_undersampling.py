#!/usr/bin/env python3

import os
import numpy as np
import plotly.graph_objects as go

from modules.dynamics import calculate_pattern_overlaps
from parameters import (
    N, dt, phi_beta, phi_r_m, phi_x_r,
    g_q, g_x, use_numba, use_g, multiple_dir_name
)

# ----------------------------------------------------
# SETTINGS
# ----------------------------------------------------
FILES = [19, 20, 21, 22, 335, 24, 29, 32, 40, 41, 43, 333]
UNDERSAMPLING = 200

# ----------------------------------------------------
# PATHS
# ----------------------------------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
firing_dir = os.path.join(base_dir, "firing_rates")
patterns_path = os.path.join(base_dir, "memory_patterns.npy")

plots_dir = os.path.join(base_dir, "..", "plot_batch")
os.makedirs(plots_dir, exist_ok=True)

# ----------------------------------------------------
# LOAD PATTERNS
# ----------------------------------------------------
eta = np.load(patterns_path)
phi_params = {"r_m": phi_r_m, "beta": phi_beta, "x_r": phi_x_r}
g_params = {"q_f": g_q, "x_f": g_x}

# ----------------------------------------------------
# STORAGE FOR GLOBAL CONCATENATION
# ----------------------------------------------------
batch_rates = []
batch_overlaps = []
trial_boundaries = []

global_start = 1
global_end = 0

# ----------------------------------------------------
# PROCESS EACH RUN
# ----------------------------------------------------
for idx in FILES:
    print(f"\nProcessing file {idx} ...")

    path = os.path.join(firing_dir, f"firing_rates_{idx}.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Load full run
    rates = np.load(path)          # (T, N)
    overlaps = calculate_pattern_overlaps(
        rates, eta, phi_params, g_params,
        use_numba=use_numba, use_g=use_g
    )

    # Undersampling
    rates_us = rates[::UNDERSAMPLING]
    overlaps_us = overlaps[::UNDERSAMPLING]

    # Store for Batch data
    batch_rates.append(rates_us)
    batch_overlaps.append(overlaps_us)

    # Update trial boundaries (in undersampled grid)
    run_length = rates_us.shape[0]
    global_end = global_start + run_length - 1
    trial_boundaries.append((global_start, global_end))
    global_start = global_end + 1

    print(f"Undersampled points for {idx}: {run_length}")

# ----------------------------------------------------
# CONSTRUCT BATCH DATA
# ----------------------------------------------------
batch_rates = np.concatenate(batch_rates, axis=0)
batch_overlaps = np.concatenate(batch_overlaps, axis=0)

# Save batch firing rates
out_batch = os.path.join(firing_dir, "all_firing_rates_undersampled.npy")
np.save(out_batch, batch_rates.astype(np.float32))

print("\nSaved Batch undersampled firing rates:")
print(out_batch)
# ----------------------------------------------------
# SAVE TRIAL BOUNDARIES FILE
# ----------------------------------------------------
txt_path = os.path.join(firing_dir, "trials.txt")

with open(txt_path, "w") as f:
    for start, end in trial_boundaries:
        f.write(f"{start} {end} 0\n")

print(f"\nSaved trial boundaries to {txt_path}")

# ----------------------------------------------------
# CREATE ONE BIG HTML PLOT OF UNDERSAMPLED OVERLAPS
# ----------------------------------------------------
time = np.arange(1, batch_overlaps.shape[0] + 1)

fig = go.Figure()
P = batch_overlaps.shape[1]

for p in range(P):
    fig.add_trace(go.Scatter(
        x=time,
        y=batch_overlaps[:, p],
        mode="lines",
        name=f"Pattern {p}"
    ))

fig.update_layout(
    title="Batch Undersampled Overlaps (All Runs)",
    xaxis_title="Time (undersampled, 1-based)",
    yaxis_title="Overlap",
    height=700,
    width=1200
)

html_path = os.path.join(plots_dir, "batch_overlaps.html")
fig.write_html(html_path, include_plotlyjs="cdn")
print(f"\nSaved Plotly HTML to {html_path}")

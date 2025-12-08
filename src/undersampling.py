#!/usr/bin/env python3
"""
Plot full-resolution and undersampled overlaps for firing_rate_22.npy
and save an interactive Plotly HTML. Also save the undersampled firing rates.
"""

import os
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from modules.dynamics import calculate_pattern_overlaps
from parameters import (
    N, dt, phi_beta, phi_r_m, phi_x_r,
    g_q, g_x, use_numba, use_g, multiple_dir_name
)

# -----------------------
# User-configurable
# -----------------------
file_index = 22
undersampling = 200  # keep 1 every undersampling steps
save_undersampled_filename = True

# -----------------------
# Paths
# -----------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
firing_rates_dir = os.path.join(base_dir, "firing_rates")
plots_dir = os.path.join(base_dir, "..", "plots")
os.makedirs(plots_dir, exist_ok=True)

input_fname = os.path.join(firing_rates_dir, f"firing_rates_{file_index}.npy")
patterns_path = os.path.join(base_dir, "memory_patterns.npy")
undersampled_out = os.path.join(firing_rates_dir, f"firing_rates_{file_index}_undersampled.npy")
html_out = os.path.join(plots_dir, f"overlaps_firing_rates_{file_index}.html")

# -----------------------
# Sanity checks
# -----------------------
if not os.path.exists(input_fname):
    raise FileNotFoundError(f"Input firing-rate file not found: {input_fname}")
if not os.path.exists(patterns_path):
    raise FileNotFoundError(f"Patterns file not found: {patterns_path}")

# -----------------------
# Load data
# -----------------------
print(f"Loading firing rates from {input_fname} ...")
rates = np.load(input_fname)  # shape (T, Nfeatures) or (T,)
print(f"Rates shape: {rates.shape}")

print(f"Loading memory patterns from {patterns_path} ...")
eta = np.load(patterns_path)

# Prepare phi/g params for calculate_pattern_overlaps if needed by your implementation
phi_params = {"r_m": phi_r_m, "beta": phi_beta, "x_r": phi_x_r}
g_params = {"q_f": g_q, "x_f": g_x}

# -----------------------
# Compute overlaps (full resolution)
# -----------------------
print("Computing overlaps (full resolution)...")
overlaps = calculate_pattern_overlaps(rates, eta, phi_params, g_params, use_numba=use_numba, use_g=use_g)
# overlaps shape: (T, P)
T, P = overlaps.shape
print(f"Overlaps shape: {overlaps.shape}")

# -----------------------
# Undersample (after detection as requested)
# -----------------------
if undersampling <= 1:
    undersampled_rates = rates.copy()
    undersampled_overlaps = overlaps.copy()
else:
    undersampled_rates = rates[::undersampling]
    undersampled_overlaps = overlaps[::undersampling]

print(f"Undersampled length: {undersampled_rates.shape[0]} (factor {undersampling})")

# Save undersampled firing rates as float32 to save space
if save_undersampled_filename:
    undersampled_to_save = undersampled_rates.astype(np.float32, copy=False)
    np.save(undersampled_out, undersampled_to_save)
    print(f"Saved undersampled firing rates to {undersampled_out} (dtype float32)")

# -----------------------
# Build Plotly figure
# -----------------------
title = f"Pattern overlaps — file {file_index} (full vs undersampled x{undersampling})"

fig = make_subplots(rows=2, cols=1, shared_xaxes=False,
                    vertical_spacing=0.12,
                    subplot_titles=("Full-resolution overlaps", f"Undersampled overlaps (1 every {undersampling} steps)"))

# Full-resolution plot: time axis in 1-based steps
time_full = np.arange(1, T + 1)

for p in range(P):
    fig.add_trace(
        go.Scatter(
            x=time_full,
            y=overlaps[:, p],
            mode="lines",
            name=f"Pattern {p}",
            legendgroup=f"P{p}",
            showlegend=True if p == 0 else False  # show legend entry once (we'll add full legend below)
        ),
        row=1, col=1
    )

# add a (single) legend for all patterns on top plot (multiple traces share legendgroup)
for p in range(P):
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines",
            name=f"Pattern {p}",
            legendgroup=f"P{p}"
        ),
        row=1, col=1
    )

# Undersampled plot: 1-based time grid for undersampled sequence
Tu = undersampled_overlaps.shape[0]
time_us = np.arange(1, Tu + 1)

for p in range(P):
    fig.add_trace(
        go.Scatter(
            x=time_us,
            y=undersampled_overlaps[:, p],
            mode="lines+markers",
            marker=dict(size=4),
            name=f"Pattern {p}",
            legendgroup=f"P{p}",
            showlegend=False
        ),
        row=2, col=1
    )

# Layout tweaks
fig.update_xaxes(title_text="Time (steps, 1-based)", row=1, col=1)
fig.update_xaxes(title_text="Undersampled time (1-based)", row=2, col=1)
fig.update_yaxes(title_text="Overlap", row=1, col=1)
fig.update_yaxes(title_text="Overlap", row=2, col=1)

fig.update_layout(
    height=800,
    width=1200,
    title_text=title,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.01)
)

# -----------------------
# Save interactive HTML
# -----------------------
fig.write_html(html_out, include_plotlyjs="cdn")
print(f"Saved interactive HTML to {html_out}")

# Also show in a blocking window if desired (comment out in headless machines)
# fig.show()


import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parameters import N, multiple_dir_name

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
firing_file = "all_firing_rates_undersampled_subset.npy"
overlap_file = "batch_overlaps.npy"

# --------------------------------------------------
# PATHS
# --------------------------------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
firing_dir = os.path.join(base_dir, "firing_rates")
plot_dir = os.path.join(base_dir, "..", "plot_batch")

firing_path = os.path.join(firing_dir, firing_file)
overlap_path = os.path.join(plot_dir, overlap_file)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
rates = np.load(firing_path)        # (T, N_sel)
overlaps = np.load(overlap_path)    # (T, P)

T, N_sel = rates.shape
P = overlaps.shape[1]

time = np.arange(T)

print(f"Loaded rates: {rates.shape}")
print(f"Loaded overlaps: {overlaps.shape}")

# --------------------------------------------------
# CREATE FIGURE
# --------------------------------------------------
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.35, 0.65],
    vertical_spacing=0.04,
    subplot_titles=("Pattern Overlaps", "Neuron Firing Rates")
)

# --------------------------------------------------
# TOP PANEL: OVERLAPS (ON by default)
# --------------------------------------------------
for p in range(P):
    fig.add_trace(
        go.Scatter(
            x=time,
            y=overlaps[:, p],
            mode="lines",
            name=f"Pattern {p}",
            line=dict(width=2)
        ),
        row=1,
        col=1
    )

# --------------------------------------------------
# BOTTOM PANEL: FIRING RATES (OFF by default)
# --------------------------------------------------
for i in range(N_sel):
    fig.add_trace(
        go.Scatter(
            x=time,
            y=rates[:, i],
            mode="lines",
            name=f"Neuron {i}",
            line=dict(width=1),
            visible="legendonly"   # <<< THIS IS THE KEY
        ),
        row=2,
        col=1
    )

# --------------------------------------------------
# LAYOUT
# --------------------------------------------------
fig.update_layout(
    title="Firing Rates and Pattern Overlaps",
    height=900,
    width=1500,
    hovermode="x unified",
    legend=dict(
        orientation="v",
        x=1.02,
        y=1,
        font=dict(size=11)
    )
)

fig.update_xaxes(title_text="Time step", row=2, col=1)
fig.update_yaxes(title_text="Overlap", row=1, col=1)
fig.update_yaxes(title_text="Firing rate", row=2, col=1)

# --------------------------------------------------
# SAVE + SHOW
# --------------------------------------------------
out_html = os.path.join(plot_dir, "firing_rates_selectable_neurons.html")
fig.write_html(out_html, include_plotlyjs="cdn")
fig.show()

print(f"Saved interactive plot to:\n{out_html}")

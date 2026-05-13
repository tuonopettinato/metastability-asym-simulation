"""
This script processes multiple simulation runs by loading their firing rates and OU processes,
computing pattern overlaps, concatenating the results, and saving both the concatenated firing rates,
OU processes, and trial boundaries. It also generates an HTML plot of the overlaps using Plotly.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib import cm

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
    base_dir = os.path.join(base_dir, "..", "test_set", "npy")
else:
    base_dir = base_dir

firing_dir = os.path.join(base_dir, "firing_rates")
ou_dir = os.path.join(base_dir, "ou_process")
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
all_ou = []
boundaries = []   # per boundaries reali

current_start = 0
current_end = -1

# ----------------------------------------------------
# PROCESS EACH FILE
# ----------------------------------------------------
for idx in FILES:
    print(f"\nProcessing file {idx} ...")

    # --- firing rates ---
    path_rates = os.path.join(firing_dir, f"firing_rates_{idx}.npy")
    if not os.path.exists(path_rates):
        raise FileNotFoundError(path_rates)
    rates = np.load(path_rates)  # shape (T, N)
    
    # --- OU process ---
    path_ou = os.path.join(ou_dir, f"ou_process_{idx}.npy")
    if not os.path.exists(path_ou):
        raise FileNotFoundError(path_ou)
    ou = np.load(path_ou)  # shape (T, N) o (T, dim OU)
    
    # --- overlaps ---
    overlaps = calculate_pattern_overlaps(
        rates, eta, phi_params, g_params,
        use_numba=use_numba, use_g=use_g
    )

    T = rates.shape[0]

    all_rates.append(rates)
    all_overlaps.append(overlaps)
    all_ou.append(ou)

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
all_ou = np.concatenate(all_ou, axis=0)

total_T = all_rates.shape[0]
total_T_ou = all_ou.shape[0]
print(f"\nTotal concatenated timesteps (firing rates): {total_T}")
print(f"Total concatenated timesteps (OU process): {total_T_ou}")

# ----------------------------------------------------
# SAVE CONCATENATED FIRING RATES
# ----------------------------------------------------
out_batch = os.path.join(firing_dir, "all_firing_rates.npy")
np.save(out_batch, all_rates.astype(np.float32))
print(f"Saved concatenated firing rates to {out_batch}")

# ----------------------------------------------------
# SAVE CONCATENATED OU PROCESS
# ----------------------------------------------------
out_ou = os.path.join(ou_dir, "all_ou_process.npy")
np.save(out_ou, all_ou.astype(np.float32))
print(f"Saved concatenated OU process to {out_ou}")

# ----------------------------------------------------
# SAVE TRIAL BOUNDARIES
# ----------------------------------------------------
txt_path = os.path.join(firing_dir, "trials.txt") if not test_set else os.path.join(firing_dir, "test_trials.txt")
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

# save npy of all overlaps too
npy_path = os.path.join(plots_dir, "batch_overlaps.npy")
np.save(npy_path, all_overlaps.astype(np.float32))
print(f"Saved overlaps numpy to {npy_path}")

# =====================================================
# TRANSITION ANALYSIS
# =====================================================
print("\n================ TRANSITION ANALYSIS ================\n")

T, P = all_overlaps.shape

# ----------------------------------------------------
# PARAMETRI
# ----------------------------------------------------
USE_THRESHOLD = True
THRESHOLD = 0.5   # stato valido solo se overlap > THRESHOLD

# ----------------------------------------------------
# STATO DISCRETO
# ----------------------------------------------------
states = np.argmax(all_overlaps, axis=1) + 1  # shift 0->1, 1->2, ...

if USE_THRESHOLD:
    max_vals = np.max(all_overlaps, axis=1)
    states[max_vals < THRESHOLD] = -1  # stato indefinito

# ----------------------------------------------------
# RIMUOVI STATI INVALIDI
# ----------------------------------------------------
valid_idx = states != -1
states_clean = states[valid_idx]
print(f"Total valid timesteps: {len(states_clean)} / {T}")

# ----------------------------------------------------
# TROVA TRANSIZIONI
# ----------------------------------------------------
transitions = np.where(np.diff(states_clean) != 0)[0] + 1
print(f"Number of transitions: {len(transitions)}")

# ----------------------------------------------------
# DWELL TIMES GLOBALI
# ----------------------------------------------------
dwell_times = np.diff(np.concatenate(([0], transitions, [len(states_clean)])))
mean_dwell = np.mean(dwell_times)
std_dwell = np.std(dwell_times)
print(f"Mean dwell time: {mean_dwell:.2f}")
print(f"Std dwell time: {std_dwell:.2f}")

# ----------------------------------------------------
# ISTOGRAMMA DWELL TIMES GLOBALI (grigio)
# ----------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(dwell_times, bins=30, alpha=0.7, color='gray')
plt.axvline(mean_dwell, linestyle='--', linewidth=2, color = 'k',
            label=f"Mean = {mean_dwell:.2f}\nStd = {std_dwell:.2f}")
plt.xlabel("Dwell time", fontsize=20)
plt.ylabel("Count", fontsize=20)
#plt.title("Distribution of dwell times (all patterns)", fontsize=20)
plt.legend(fontsize = 18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# DISTRIBUZIONE TEMPORALE DELLE TRANSIZIONI
# ----------------------------------------------------
plt.figure(figsize=(6,4))
plt.hist(transitions, bins=50, alpha=0.7)
plt.xlabel("Time")
plt.ylabel("Number of transitions")
plt.title("Temporal distribution of transitions")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# MATRICE DI TRANSIZIONE
# ----------------------------------------------------
transition_matrix = np.zeros((P, P))
for i in range(len(states_clean) - 1):
    s1 = states_clean[i] - 1  # map 1..P -> 0..P-1
    s2 = states_clean[i + 1] - 1
    if s1 != s2:
        transition_matrix[s1, s2] += 1

# normalizzazione righe
row_sums = transition_matrix.sum(axis=1, keepdims=True)
transition_matrix = np.divide(
    transition_matrix,
    row_sums,
    where=row_sums != 0
)
print("\nTransition matrix (row-normalized):\n")
print(transition_matrix)

# colori tab10
colors = plt.get_cmap('tab10').colors[:P]

plt.figure(figsize=(6,5))
plt.imshow(transition_matrix, cmap='Greys')  # bianco-nero
plt.colorbar(label="Transition probability")
plt.xticks(ticks=np.arange(P), labels=[f"P{p+1}" for p in range(P)])
plt.yticks(ticks=np.arange(P), labels=[f"P{p+1}" for p in range(P)])
plt.xlabel("To state", fontsize=20)
plt.ylabel("From state", fontsize=20)
#plt.title("Transition matrix", fontsize=20)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# RATE DI TRANSIZIONE
# ----------------------------------------------------
total_time = len(states_clean)
transition_rate = len(transitions) / total_time
print(f"\nTransition rate: {transition_rate:.4f} transitions per timestep")

# ----------------------------------------------------
# DWELL TIMES PER PATTERN
# ----------------------------------------------------
dwell_start_indices = np.concatenate(([0], transitions))
dwell_states = states_clean[dwell_start_indices]

dwell_per_pattern = {p+1: [] for p in range(P)}
for dt, st in zip(dwell_times, dwell_states):
    if st != -1:
        dwell_per_pattern[st].append(dt)

print("\n=========== DWELL TIMES PER PATTERN ===========\n")
for p in range(1, P+1):
    values = dwell_per_pattern[p]
    if len(values) == 0:
        continue
    mean_p = np.mean(values)
    std_p = np.std(values)
    print(f"Pattern {p}: count={len(values)}, mean={mean_p:.2f}, std={std_p:.2f}")

# ----------------------------------------------------
# ISTOGRAMMI PER PATTERN
# ----------------------------------------------------
for p in range(1, P+1):
    values = dwell_per_pattern[p]
    if len(values) == 0:
        continue
    mean_p = np.mean(values)
    std_p = np.std(values)

    plt.figure(figsize=(5,4))
    plt.hist(values, bins=20, alpha=0.7, color=colors[p-1])
    plt.axvline(mean_p, linestyle='--', linewidth=2, color = 'k',
                label=f"Mean = {mean_p:.2f}\nStd = {std_p:.2f}")
    plt.xlabel("Dwell time", fontsize=20)
    plt.ylabel("Count", fontsize=20)
    plt.legend(title = f"P{p}", fontsize = 18)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------
# BOXPLOT COMPARATIVO (ordine personalizzato)
# ----------------------------------------------------
order = [3, 4, 1, 2]

data = []
labels = []
valid_patterns = []

for p in order:
    values = dwell_per_pattern[p]
    if len(values) > 0:
        data.append(values)
        labels.append(f"P{p}")
        valid_patterns.append(p)

plt.figure(figsize=(8,5))

box = plt.boxplot(
    data,
    labels=labels,
    patch_artist=True,
    medianprops=dict(color='black', linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    showfliers=False,
    capprops=dict(linewidth=1.5)
)

# colori coerenti con i pattern
ordered_colors = [colors[p-1] for p in valid_patterns]

# scatter points
for i, p in enumerate(valid_patterns):
    values = dwell_per_pattern[p]
    x_jitter = np.random.normal(loc=i+1, scale=0.05, size=len(values))
    plt.scatter(
        x_jitter,
        values,
        color=colors[p-1],
        alpha=0.7,
        s=15,
        edgecolor='k',
        linewidth=0.3
    )

# colora le box
for patch, color in zip(box['boxes'], ordered_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.3)

plt.ylabel("Dwell time", fontsize=20)
plt.xticks(fontsize=18)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ----------------------------------------------------
# SALVATAGGIO DEI RISULTATI
# ----------------------------------------------------
analysis_dir = os.path.join(plots_dir, "transition_analysis")
os.makedirs(analysis_dir, exist_ok=True)

np.save(os.path.join(analysis_dir, "states.npy"), states_clean)
np.save(os.path.join(analysis_dir, "transitions.npy"), transitions)
np.save(os.path.join(analysis_dir, "dwell_times.npy"), dwell_times)
np.save(os.path.join(analysis_dir, "transition_matrix.npy"), transition_matrix)

print(f"\nSaved transition analysis to {analysis_dir}")
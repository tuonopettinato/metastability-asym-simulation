import os
import numpy as np
from parameters import N, multiple_dir_name

# -----------------------------
# SETTINGS
# -----------------------------
input_file_name = "all_firing_rates_undersampled.npy"
output_file_name = "all_firing_rates_undersampled_subset.npy"
neuron_indices = [0, 1, 23, 4, 7]  # neuroni da selezionare

# -----------------------------
# PATHS
# -----------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
firing_dir = os.path.join(base_dir, "firing_rates")
os.makedirs(firing_dir, exist_ok=True)

input_file = os.path.join(firing_dir, input_file_name)
output_file = os.path.join(firing_dir, output_file_name)

# -----------------------------
# LOAD DATA
# -----------------------------
if not os.path.exists(input_file):
    raise FileNotFoundError(f"Input file not found: {input_file}")

data = np.load(input_file)
print(f"Original data shape: {data.shape}")

# -----------------------------
# SELECT NEURONS
# -----------------------------
# Controllo validità degli indici
neuron_indices = sorted(set(neuron_indices))
max_idx = max(neuron_indices)
if max_idx >= data.shape[1]:
    raise ValueError(f"Index {max_idx} out of bounds. Data has only {data.shape[1]} neurons.")

subset_data = data[:, neuron_indices]
print(f"Reduced data shape: {subset_data.shape}")

# -----------------------------
# SAVE REDUCED DATA
# -----------------------------
np.save(output_file, subset_data.astype(np.float32))
print(f"Saved reduced data to {output_file}")

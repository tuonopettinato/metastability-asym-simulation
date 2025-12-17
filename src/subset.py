"""
This script selects a subset of neurons from stored firing rates data.
It loads the concatenated firing rates from all_firing_rates.npy,
applies neuron selection based on indices produced by selection.py,
and saves the reduced dataset into a new .npy file.

The list 'neuron_indices' must contain exactly the final chosen neurons:
K active neurons + K/4 low neurons.
"""

import os
import numpy as np
from parameters import N, multiple_dir_name, test_set

# -----------------------------
# SETTINGS
# -----------------------------
input_file_name = "all_firing_rates.npy"
if not test_set:
    output_file_name = "all_firing_rates_undersampled_subset.npy"
else:
    output_file_name = "test_all_firing_rates_undersampled_subset.npy"

# Final selection: K active neurons + K/4 low neurons (100 total if K=80)
neuron_indices = [
    11,24,32,41,56,63,64,70,77,109,123,180,199,218,237,243,253,280,348,354,
    359,364,384,388,407,419,422,423,448,481,514,530,532,541,563,581,598,599,
    600,622,637,645,688,696,707,712,736,762,774,775,800,810,862,872,873,876,
    897,934,943,969,990,1036,1053,1068,1069,1075,1091,1107,1160,1184,1189,
    1194,1200,1223,1226,1245,1251,1254,1278,1294,1305,1329,1343,1346,1356,
    1358,1365,1373,1385,1394,1409,1422,1433,1449,1461,1466,1472,1479,1487,1489
]

# -----------------------------
# PATHS
# -----------------------------
if not test_set:
    base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
else:
    base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "test_set", "npy")
firing_dir = os.path.join(base_dir, "firing_rates")

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
# NEURON SELECTION
# -----------------------------
neuron_indices = sorted(set(neuron_indices))  # enforce unique sorted indices

# Validate indices
if len(neuron_indices) == 0:
    raise ValueError("The neuron_indices list is empty.")

if max(neuron_indices) >= data.shape[1]:
    raise ValueError(
        f"Index {max(neuron_indices)} is out of bounds. "
        f"Dataset contains {data.shape[1]} neurons."
    )

# Slice the dataset over the selected neurons
subset_data = data[:, neuron_indices]
print(f"Reduced data shape: {subset_data.shape}")

# -----------------------------
# SAVE OUTPUT
# -----------------------------
np.save(output_file, subset_data.astype(np.float32))
print(f"Saved reduced data to {output_file}")


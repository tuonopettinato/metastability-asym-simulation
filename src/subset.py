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

# Final selection:
neuron_indices = [
    4,39,51,66,68,70,73,81,83,93,101,113,132,133,137,142,144,148,149,165,167,176,189,
    192,196,223,
    224,233,235,244,246,258,271,287,295,303,313,316,320,325,327,329,334,340,345,352,
    356,360,362,374,378,384,403,419,420,426,433,442,443,449,454,458,465,468,474,484,
    485,492,493,501,523,528,544,549,555,558,559,565,574,585,591,593,594,606,608,610,
    616,628,641,646,647,656,661,665,673,682,685,704,708,715,719,725,728,729,746,749,
    754,756,759,765,770,771,782,790,812,813,817,825,830,831,833,838,859,868,872,883,
    894,898,906,907,908,935,947,948,952,963,990,1011,1012,1022,1028,1031,1035,1051,1059,
    1100,1101,1107,1119,1123,1129,1130,1131,1140,1143,1158,1160,1161,1164,1171,1179,1183,
    1191,1194,1196,1203,1213,1216,1223,1237,1241,1255,1270,1273,1277,1280,1282,1291,1301,
    1305,1308,1311,1312,1331,1337,1350,1357,1365,1374,1376,1389,1391,1395,1403,1430,1454,
    1466,1469,1482,1493
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


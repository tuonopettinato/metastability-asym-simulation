import os
import numpy as np


# --------------------------------------------------
# File path
data_dir = "memory_patterns"


# paths to the 4 attractor files
files = [
    os.path.join(data_dir, "memory1.npy"),
    os.path.join(data_dir, "memory2.npy"),
    os.path.join(data_dir, "memory3.npy"),
    os.path.join(data_dir, "memory4.npy"),
]

last_rates = []

for f in files:
    data = np.load(f)        # shape (T, N)
    last_rates.append(data[-1])  # take last time step -> (N,)

# stack into shape (4, N)
attractor_coordinates = np.stack(last_rates, axis=0)

# save result
np.save(os.path.join(data_dir, "patterns.npy"), attractor_coordinates)

print("Saved memory_patterns/patterns.npy with shape:", attractor_coordinates.shape)
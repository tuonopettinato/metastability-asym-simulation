"""
This script selects a balanced subset of neurons from stored memory patterns.
It identifies K neurons that have an activity distribution across patterns
within a specified range (20-30% active per pattern) and combines them with
a set of low-activity neurons to form a final array of K + K/p neurons.
"""

import numpy as np
from modules.activation import step_function
from parameters import g_q, g_x, multiple_dir_name, N, p

q_f = g_q
x_f = g_x

pattern_path = f'{multiple_dir_name}_{N}/npy/phi_memory_patterns.npy'

# Load stored patterns
phi = np.load(pattern_path).transpose()

# Apply the step activation to patterns
phi_step = step_function(phi, q_f=q_f, x_f=x_f)

# N = number of neurons, P = number of stored patterns
N, P = phi_step.shape

# Identify neurons that are inactive for all patterns
low_neurons = np.where(phi_step.sum(axis=1) <= 0)[0]

print("Low neurons (inactive in all patterns):")
print(low_neurons)
print(f"Total number of low neurons: {len(low_neurons)}")

# Try to find K neurons whose activity distribution across patterns is balanced
K = 80
selected_neurons = []

max_trials = 10000
found = False

for _ in range(max_trials):
    # Randomly sample K neurons
    candidate = np.random.choice(N, K, replace=False)

    # Compute the percentage of active neurons per pattern
    perc_active = (phi_step[candidate, :] > 0).sum(axis=0) / K * 100

    # Accept only if each pattern has 20–30% active neurons among the selected K
    if all(20 <= perc_active[p] <= 30 for p in range(P)):
        selected_neurons = candidate
        found = True
        break

if found:
    print("\nBalanced neuron subset (active neurons):")
    print(selected_neurons)
    print("Activity percentages per pattern:", perc_active)
else:
    print("\nA balanced set of K neurons could not be found.")

# ----------------------------------------------------
# FINAL PART: build a 100-neuron array including low neurons
# ----------------------------------------------------

if found:
    final_size = K + K // p      # Total target size (e.g., 80 + 20 = 100)
    n_low = K // p               # Number of low neurons to include

    if len(low_neurons) < n_low:
        raise RuntimeError("Not enough low neurons to sample K/4 of them.")

    # Select K/4 low neurons randomly
    chosen_low = np.random.choice(low_neurons, n_low, replace=False)

    # Merge the active set (K neurons) with K/4 low neurons
    final_array = np.concatenate([selected_neurons, chosen_low])

    # Sort the final list for readability
    final_array = np.sort(final_array)

    print("\nFinal 100-neuron array (80 active + 20 low):")
    print(final_array)

    print("\nComma-separated output:")
    print(",".join(str(x) for x in final_array))



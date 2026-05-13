import os
import numpy as np
import matplotlib.pyplot as plt

# === PATH ===
file_path = "/Users/valeriocaporioni/Documents/thesis/metastability-asym-simulation/simulation_results/npy/firing_rates.npy"

# === PARAMETRI ===
n_neurons = 200
seed = 0

# === LOAD ===
h = np.load(file_path)   # shape: (T, N)
print("Shape:", h.shape)

T, N = h.shape

# === SELECT NEURONS ===
np.random.seed(seed)
idx = np.random.choice(N, n_neurons, replace=False)

# Prendi solo i neuroni selezionati e trasponi → (neuroni, tempo)
h_sel = h[:, idx].T

# === PLOT ===
plt.figure(figsize=(10, 6))

plt.imshow(
    h_sel,
    aspect='auto',
    cmap='gray',     # bianco/nero
    interpolation='nearest'
)

plt.xlabel("$t$")
plt.ylabel("Neuron")
#plt.title(f"Firing rates ({n_neurons} neurons)")

plt.colorbar()

plt.tight_layout()
plt.show()
import os
import numpy as np
import matplotlib.pyplot as plt

from modules.dynamics import calculate_pattern_overlaps
from parameters import (
    N, phi_beta, phi_r_m, phi_x_r,
    g_q, g_x, use_numba, use_g,
    multiple_dir_name,
    seed
)

# ===============================================================
# Paths
# ===============================================================
base_dir = os.path.join(
    os.path.dirname(__file__), "..",
    f"{multiple_dir_name}_{N}", "npy"
)

patterns_path = os.path.join(base_dir, "memory_patterns.npy")
firing_dir = os.path.join(base_dir, "firing_rates")
ou_dir = os.path.join(base_dir, "ou_process")

# ===============================================================
# Load patterns
# ===============================================================
eta = np.load(patterns_path)

phi_params = {'r_m': phi_r_m, 'beta': phi_beta, 'x_r': phi_x_r}
g_params   = {'q_f': g_q, 'x_f': g_x}

# ===============================================================
# Utilities
# ===============================================================
def zscore(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

# ===============================================================
# Compute C and d
# ===============================================================
def compute_C_d(firing):
    overlaps = calculate_pattern_overlaps(
        firing, eta, phi_params, g_params,
        use_numba=use_numba, use_g=use_g
    )

    # derivata semplice (robusta al downsampling)
    dO = overlaps[1:] - overlaps[:-1]
    P = dO.shape[1]

    C = np.zeros(dO.shape[0])
    for i in range(P):
        for j in range(i + 1, P):
            C -= dO[:, i] * dO[:, j]

    d = np.sum(np.abs(dO), axis=1)

    return zscore(C), zscore(d)

# ===============================================================
# Zeta events (percentile)
# ===============================================================
def detect_zeta_events(zeta, perc=98):
    thr = np.percentile(zeta, perc)
    events = (zeta > thr).astype(float)
    return events, thr

# ===============================================================
# Event-triggered average
# ===============================================================
def event_triggered_average(x, events, w=10):
    idx = np.where(events > 0)[0]
    snippets = []

    for i in idx:
        if i - w >= 0 and i + w < len(x):
            snippets.append(x[i - w:i + w + 1])

    if len(snippets) == 0:
        return None

    return np.mean(snippets, axis=0)

# ===============================================================
# Surrogate test (block shuffle)
# ===============================================================
def surrogate_test(
    x, events, w=3, block_size=50, n_surr=500, seed=None
):
    rng = np.random.default_rng(seed)

    eta_true = event_triggered_average(x, events, w)
    if eta_true is None:
        return None, None

    peak_true = np.max(np.abs(eta_true))

    T = len(events)
    n_blocks = T // block_size
    cut = n_blocks * block_size

    blocks = np.split(events[:cut], n_blocks)
    remainder = events[cut:]

    surr_peaks = np.zeros(n_surr)

    for i in range(n_surr):
        rng.shuffle(blocks)
        surr_events = np.concatenate(blocks)

        if len(remainder) > 0:
            surr_events = np.concatenate([surr_events, remainder])

        eta_s = event_triggered_average(x, surr_events, w)
        if eta_s is not None:
            surr_peaks[i] = np.max(np.abs(eta_s))

    pval = (np.sum(surr_peaks >= peak_true) + 1) / (n_surr + 1)
    return peak_true, pval

# ===============================================================
# Load files
# ===============================================================
files_firing = sorted(f for f in os.listdir(firing_dir) if f.endswith(".npy"))
files_ou = sorted(f for f in os.listdir(ou_dir) if f.endswith(".npy"))

# ===============================================================
# Single-run visualization
# ===============================================================
firing = np.load(os.path.join(firing_dir, files_firing[5]))
zeta   = np.load(os.path.join(ou_dir, files_ou[5]))

C, d = compute_C_d(firing)
events, thr = detect_zeta_events(zeta[:-1], perc=98)

w = 6
lags = np.arange(-w, w + 1)

eta_C = event_triggered_average(C, events, w)
eta_d = event_triggered_average(d, events, w)

plt.figure(figsize=(7, 4))
plt.plot(lags, eta_C, "-o", label="C | zeta events")
plt.plot(lags, eta_d, "-o", label="d | zeta events")
plt.axvline(0, color="k", ls="--")
plt.xlabel("Lag from event [steps]", fontsize = 20)
plt.ylabel("Response (z-score)", fontsize = 20)
plt.legend(fontsize = 18)
plt.tight_layout()
plt.show()

# ===============================================================
# Single-run surrogate test
# ===============================================================
peak_C, pC = surrogate_test(C, events, w)
peak_d, pd = surrogate_test(d, events, w)
print(f"Single run surrogate test results:")
print(f"Observable C: peak={peak_C:.4f}, p-value={pC:.4f}")
print(f"Observable d: peak={peak_d:.4f}, p-value={pd:.4f}")

# ===============================================================
# Statistics across runs
# ===============================================================
results = {"C": [], "d": []}

for f_fire, f_ou in zip(files_firing, files_ou):
    firing = np.load(os.path.join(firing_dir, f_fire))
    zeta   = np.load(os.path.join(ou_dir, f_ou))

    C, d = compute_C_d(firing)
    events, _ = detect_zeta_events(zeta[:-1], perc=98)

    _, pC = surrogate_test(C, events, w)
    _, pd = surrogate_test(d, events, w)

    results["C"].append(pC)
    results["d"].append(pd)

# ===============================================================
# Summary
# ===============================================================
for obs in ["C", "d"]:
    pvals = np.array(results[obs], dtype=float)
    sig = np.sum(pvals < 0.05) # 0.05

    print(f"\nObservable {obs}")
    print(f"Significant runs: {sig} / {len(pvals)}")
    print("p-values:", pvals)

# ===============================================================
# Histogram of p-values for observable C and d
# ===============================================================
pvals_C = np.array(results["C"], dtype=float)
pvals_d = np.array(results["d"], dtype=float)

xmax = max(pvals_C.max(), pvals_d.max())

# bin width such that there are 3 bins below 0.05
bin_width = 0.05 / 3

bins = np.arange(0, xmax + bin_width, bin_width)

plt.figure(figsize=(10, 4))

plt.hist(pvals_d, bins=bins, alpha=0.7, edgecolor='k', label='d')
plt.hist(pvals_C, bins=bins, alpha=0.7, edgecolor='k', label='C')


plt.axvline(0.05, color='r', ls='--', lw=2, label='p = 0.05')

plt.xlim(0, xmax)
plt.xlabel('p-value', fontsize = 20)
plt.ylabel('Count', fontsize = 20)
plt.title('Distribution of p-values', fontsize = 20)
plt.legend(fontsize = 18)

plt.tight_layout()
plt.show()
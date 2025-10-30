"""
Cross-correlation between ζ(t) and pattern overlaps ΔO(t) across multiple runs.
Each run has its own 'firing_rates/*.npy' and 'ou_process/*.npy'.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from modules.dynamics import calculate_pattern_overlaps
from parameters import (
    N,
    dt,
    phi_beta,
    phi_r_m,
    phi_x_r,
    g_q,
    g_x,
    use_numba,
    use_g,
    multiple_dir_name
)

# ----------------------------
# Paths
# ----------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
patterns_path = os.path.join(base_dir, "memory_patterns.npy")
firing_dir = os.path.join(base_dir, "firing_rates")
ou_dir = os.path.join(base_dir, "ou_process")

# ----------------------------
# Phi and g function parameters
# ----------------------------
phi_params = {'r_m': phi_r_m, 'beta': phi_beta, 'x_r': phi_x_r}
g_params = {'q_f': g_q, 'x_f': g_x}

# ----------------------------
# Load memory patterns
# ----------------------------
eta = np.load(patterns_path)

# ----------------------------
# Cross-correlation on ΔO
# ----------------------------
def compute_cross_correlation(firing_file, ou_file, max_lag_steps):
    """Compute normalized cross-correlation between ΔO(t) and ζ(t)."""
    firing = np.load(firing_file)
    zeta = np.load(ou_file)

    # compute overlaps and average over patterns
    overlaps = calculate_pattern_overlaps(firing, eta, phi_params, g_params,
                                          use_numba=use_numba, use_g=use_g)
    O_mean = overlaps.mean(axis=1)

    # compute ΔO(t)
    dO = np.diff(O_mean)
    z = zeta[:len(dO)]  # align lengths

    # standardize
    dO_std = (dO - dO.mean()) / dO.std()
    z_std = (z - z.mean()) / z.std()

    # full correlation
    corr = np.correlate(dO_std, z_std, mode="full") / len(z_std)
    center = len(corr) // 2
    return corr[center - max_lag_steps:center + max_lag_steps + 1]


def cross_correlation_multi_run(firing_dir, ou_dir, max_lag_ms=1000):
    """Compute average cross-correlation across runs."""
    files = sorted([f for f in os.listdir(firing_dir) if f.endswith(".npy")])
    lag_steps = int(max_lag_ms / (dt * 1000))
    lags = np.arange(-lag_steps, lag_steps + 1) * dt

    corr_list = []

    for fname in files:
        firing_file = os.path.join(firing_dir, fname)
        ou_file = os.path.join(ou_dir, fname)
        if not os.path.exists(ou_file):
            print(f"[!] Missing OU file for {fname}")
            continue

        corr = compute_cross_correlation(firing_file, ou_file, lag_steps)
        corr_list.append(corr)

    if len(corr_list) == 0:
        raise RuntimeError("No valid runs found for correlation analysis.")

    corr_mean = np.mean(np.stack(corr_list), axis=0)
    corr_std = np.std(np.stack(corr_list), axis=0)
    return lags, corr_mean, corr_std


# ----------------------------
# Plot
# ----------------------------
if __name__ == "__main__":
    max_lag_ms = 1000  # adjust window for cross-correlation
    lags, corr_mean, corr_std = cross_correlation_multi_run(firing_dir, ou_dir, max_lag_ms=max_lag_ms)

    plt.figure(figsize=(8, 4))
    plt.plot(lags, corr_mean, label="Mean correlation", color='b')
    plt.fill_between(lags, corr_mean - corr_std, corr_mean + corr_std, color='b', alpha=0.3, label="±1 std")
    plt.axvline(0, color='k', linestyle='--', label='Zero lag')
    plt.xlabel("Lag [s]")
    plt.ylabel("Correlation (ΔO vs ζ)")
    plt.title(f"Cross-correlation across runs (window={max_lag_ms} ms)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

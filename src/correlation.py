"""
Analyze the relationship between noise (ζ) and the variation of pattern overlaps using cross-correlation.
Load ζ and firing rate files from disk, compute overlaps, and correlate with lag.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from modules.dynamics import calculate_pattern_overlaps
from modules.activation import sigmoid_function
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
    multiple_dir_name,
    single_dir_name
)

# ----------------------------
# Paths
# ----------------------------
patterns_path = os.path.join(os.path.dirname(__file__), "..", f'{single_dir_name}', 'npy', 'memory_patterns.npy')
firing_rates_path = os.path.join(os.path.dirname(__file__), "..", f'{single_dir_name}', 'npy', 'firing_rates')
zeta_path = os.path.join(os.path.dirname(__file__), "..", f'{single_dir_name}', 'npy', 'ou_process.npy')

# ----------------------------
# Phi and g function parameters
# ----------------------------
phi_params = {'r_m': phi_r_m, 'beta': phi_beta, 'x_r': phi_x_r}
g_params = {'q_f': g_q, 'x_f': g_x}

# ----------------------------
# Load patterns
# ----------------------------
eta = np.load(patterns_path)

# ----------------------------
# Cross-correlation analysis using variation of overlaps
# ----------------------------
def cross_correlation_analysis(firing_dir, zeta_file, max_lag_ms=500, use_derivative=True):
    """
    Compute cross-correlation between zeta(t) and variation of pattern overlaps.
    
    Parameters
    ----------
    firing_dir : str
        Directory containing firing rate files (.npy)
    zeta_file : str
        File containing zeta(t) array
    max_lag_ms : int
        Maximum lag to consider in milliseconds (will be converted using dt)
    use_derivative : bool
        If True, correlate ζ(t) with temporal derivative of overlaps
    
    Returns
    -------
    lags : np.ndarray
        Lag vector in time units
    corr_mean : np.ndarray
        Cross-correlation averaged over all files
    """
    # load zeta
    zeta = np.load(zeta_file)
    
    fnames = [f for f in os.listdir(firing_dir) if f.endswith(".npy")]
    corr_all = []
    
    n_lag = int(max_lag_ms / (dt * 1000))  # convert ms to steps
    lags = np.arange(-n_lag, n_lag + 1) * dt
    
    for fname in fnames:
        firing = np.load(os.path.join(firing_dir, fname))
        overlaps = calculate_pattern_overlaps(firing, eta, phi_params, g_params, use_numba=use_numba, use_g=use_g)
        
        # mean over patterns
        overlaps_mean = overlaps.mean(axis=1)
        
        if use_derivative:
            # temporal derivative to capture variation
            overlaps_var = np.gradient(overlaps_mean, dt)
        else:
            overlaps_var = overlaps_mean
        
        # standardize
        z_std = (zeta - np.mean(zeta)) / np.std(zeta)
        o_std = (overlaps_var - np.mean(overlaps_var)) / np.std(overlaps_var)
        
        # full cross-correlation
        corr = np.correlate(o_std, z_std, mode='full') / len(z_std)
        
        # select only lags within ±n_lag
        center = len(corr) // 2
        corr_all.append(corr[center - n_lag:center + n_lag + 1])
    
    # average across files
    corr_mean = np.mean(np.stack(corr_all, axis=0), axis=0)
    
    return lags, corr_mean

# ----------------------------
# Plot
# ----------------------------
if __name__ == "__main__":
    lags, corr_mean = cross_correlation_analysis(firing_rates_path, zeta_path, max_lag_ms=500, use_derivative=True)
    
    plt.figure(figsize=(8,4))
    plt.plot(lags, corr_mean, color='b')
    plt.axvline(0, color='k', linestyle='--', label='zero lag')
    plt.xlabel("Lag [s]")
    plt.ylabel("Cross-correlation")
    plt.title("Average cross-correlation between ζ(t) and variation of pattern overlaps")
    plt.grid(True)
    plt.legend()
    plt.show()
# -----------------------------
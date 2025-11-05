import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from modules.dynamics import calculate_pattern_overlaps
from parameters import (
    N, phi_beta, phi_r_m, phi_x_r, g_q, g_x,
    use_numba, use_g, multiple_dir_name, zeta_bar
)

# ===============================================================
# Paths and load memory patterns
# ===============================================================
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
patterns_path = os.path.join(base_dir, "memory_patterns.npy")
firing_dir = os.path.join(base_dir, "firing_rates")
ou_dir = os.path.join(base_dir, "ou_process")

eta = np.load(patterns_path)
phi_params = {'r_m': phi_r_m, 'beta': phi_beta, 'x_r': phi_x_r}
g_params = {'q_f': g_q, 'x_f': g_x}

# ===============================================================
# Helper functions
# ===============================================================
def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-12)

def compute_observables(firing_file, ou_file):
    firing = np.load(firing_file)
    zeta_raw = np.load(ou_file)
    overlaps = calculate_pattern_overlaps(firing, eta, phi_params, g_params,
                                          use_numba=use_numba, use_g=use_g)
    dO = (overlaps[2:] - overlaps[:-2]) / 2.0
    d = np.sum(np.abs(dO), axis=1)
    P = dO.shape[1]
    C = np.zeros(dO.shape[0])
    for i in range(P):
        for j in range(i+1, P):
            C -= dO[:, i] * dO[:, j]
    z_dev = np.abs(zeta_raw[1:-1] - zeta_bar)
    z_signed = zeta_raw[1:-1] - zeta_bar
    return {'C': C, 'd': d, 'z_dev': z_dev, 'z_signed': z_signed}

def crosscorr_peak(x, y, max_lag=200):
    x_std, y_std = standardize(x), standardize(y)
    corr_full = correlate(x_std, y_std, mode='full', method='fft') / len(x_std)
    lags_full = correlation_lags(len(x_std), len(y_std), mode='full')
    center = len(corr_full)//2
    corr = corr_full[center-max_lag:center+max_lag+1]
    lags = lags_full[center-max_lag:center+max_lag+1]
    idx = np.argmax(corr)
    return int(lags[idx]), float(corr[idx]), lags, corr

def surrogate_pvalue(x, y, n_surr=1000, max_lag=200, block_size=200, seed=None):
    rng = np.random.default_rng(seed)
    _, obs_val, _, _ = crosscorr_peak(x, y, max_lag=max_lag)
    T = len(y)
    n_blocks = T // block_size
    surr_vals = np.empty(n_surr)
    for i in range(n_surr):
        blocks = np.array_split(y[:n_blocks*block_size], n_blocks)
        rng.shuffle(blocks)
        y_s = np.concatenate(blocks)
        if T > n_blocks*block_size:
            y_s = np.concatenate([y_s, y[n_blocks*block_size:]])
        _, surr_vals[i], _, _ = crosscorr_peak(x, y_s, max_lag=max_lag)
    p = (np.sum(np.abs(surr_vals) >= np.abs(obs_val)) + 1) / (n_surr + 1)
    return obs_val, p

# ===============================================================
# Load files and set observables
# ===============================================================
files_firing = sorted([f for f in os.listdir(firing_dir) if f.endswith('.npy')])
files_ou = sorted([f for f in os.listdir(ou_dir) if f.endswith('.npy')])
observables = ['C', 'd']

# ===============================================================
# Single-run cross-correlation (first run) and plotting
# ===============================================================
f_fire, f_ou = files_firing[0], files_ou[0]
sigs = compute_observables(os.path.join(firing_dir, f_fire), os.path.join(ou_dir, f_ou))
results = {}
file_stats = {}
plt.figure(figsize=(8,4))

for obs in observables:
    x = sigs[obs]
    z = sigs['z_dev']
    lag, peak_val, lags, corr = crosscorr_peak(x, z, max_lag=200)
    _, pval = surrogate_pvalue(x, z, n_surr=1000, max_lag=200)
    file_stats[obs] = {'lag': lag, 'peak_val': peak_val, 'pval': pval}
    plt.plot(lags, corr, label=f'$z{obs}$, peak={peak_val:.2f} at s={lag}')

plt.axvline(0, color='k', linestyle='--', label='s = 0')
plt.xlabel('s [steps]', fontsize=20)
plt.ylabel('Cross-correlation', fontsize=20)
plt.legend(fontsize=18, loc='lower left')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(base_dir, "..", "plots", "crosscorr_single.png"))
plt.show()
results[f_fire] = file_stats

# ===============================================================
# P-value computation across all runs
# ===============================================================
sig_results = {}
for f_fire, f_ou in zip(files_firing, files_ou):
    sigs = compute_observables(os.path.join(firing_dir, f_fire), os.path.join(ou_dir, f_ou))
    file_stats = {}
    for obs in observables:
        x = sigs[obs]
        z = sigs['z_dev']
        _, pval = surrogate_pvalue(x, z, n_surr=1000, max_lag=200)
        file_stats[obs] = {'pval': pval}
    sig_results[f_fire] = file_stats

# Print number of significant runs
for obs in observables:
    pvals = [sig_results[f][obs]['pval'] for f in sig_results]
    sig_count = np.sum(np.array(pvals) < 0.05)
    print(f"\nObservable: {obs}")
    print(f"Number of runs with p-value < 0.05: {sig_count} / {len(files_firing)}")
    print("P-values:", pvals)

# ===============================================================
# Cross-correlation aggregated across runs (C and d together)
# ===============================================================
def compute_crosscorr_all_runs(files_firing, files_ou, observable, max_lag=200):
    all_corrs = []
    for f_fire, f_ou in zip(files_firing, files_ou):
        sigs = compute_observables(os.path.join(firing_dir, f_fire), os.path.join(ou_dir, f_ou))
        x = standardize(sigs[observable])
        z = standardize(sigs['z_dev'])
        corr_full = correlate(x, z, mode='full', method='fft') / len(x)
        lags_full = correlation_lags(len(x), len(z), mode='full')
        center = len(corr_full)//2
        corr = corr_full[center-max_lag:center+max_lag+1]
        lags = lags_full[center-max_lag:center+max_lag+1]
        all_corrs.append(corr)
    if len(all_corrs) == 0:
        return None
    all_corrs = np.stack(all_corrs)
    mean_corr = np.mean(all_corrs, axis=0)
    sem_corr = np.std(all_corrs, axis=0) / np.sqrt(len(all_corrs))
    return lags, mean_corr, sem_corr, len(all_corrs)

plt.figure(figsize=(8,4))
for obs in observables:
    cross_all = compute_crosscorr_all_runs(files_firing, files_ou, obs, max_lag=200)
    if cross_all:
        lags, mean_corr, sem_corr, n_runs = cross_all
        plt.fill_between(lags, mean_corr - sem_corr, mean_corr + sem_corr, alpha=0.2, label=f'${obs}$ SEM')
        plt.plot(lags, mean_corr, label=f'{obs} (mean across {n_runs} runs)')

plt.axvline(0, color='k', linestyle='--', label='s = 0')
plt.xlabel('s [steps]', fontsize=20)
plt.ylabel('Cross-correlation', fontsize=20)
plt.legend(fontsize=18)
plt.grid(True)
plt.tight_layout()
plt.show()

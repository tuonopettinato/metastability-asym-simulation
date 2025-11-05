import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags, find_peaks
from modules.dynamics import calculate_pattern_overlaps
from parameters import (
    N, phi_beta, phi_r_m, phi_x_r, g_q, g_x,
    use_numba, use_g, multiple_dir_name, zeta_bar
)

# ----------------------------
# Paths
# ----------------------------
base_dir = os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "npy")
patterns_path = os.path.join(base_dir, "memory_patterns.npy")
firing_dir = os.path.join(base_dir, "firing_rates")
ou_dir = os.path.join(base_dir, "ou_process")

eta = np.load(patterns_path)
phi_params = {'r_m': phi_r_m, 'beta': phi_beta, 'x_r': phi_x_r}
g_params = {'q_f': g_q, 'x_f': g_x}

# ----------------------------
# Helper functions
# ----------------------------
def standardize(x):
    return (x - np.mean(x)) / (np.std(x) + 1e-12)  # avoid division by zero

def compute_observables(firing_file, ou_file):
    """Compute all observables from firing rates and OU process files using central difference."""
    firing = np.load(firing_file)
    zeta_raw = np.load(ou_file)

    overlaps = calculate_pattern_overlaps(firing, eta, phi_params, g_params,
                                          use_numba=use_numba, use_g=use_g)
    
    # Central difference: ΔO(t) = (O(t+1) - O(t-1)) / 2
    dO = (overlaps[2:] - overlaps[:-2]) / 2.0

    # Overlap drops
    fall_signal = np.maximum(-dO, 0)
    fall_total = np.sum(fall_signal, axis=1)

    # Total intensity of changes
    d = np.sum(np.abs(dO), axis=1)
    
    # C(t): cross-overlap measure of competition
    P = dO.shape[1]
    C = np.zeros(dO.shape[0])
    for i in range(P):
        for j in range(i+1, P):
            C -= dO[:, i] * dO[:, j]
    
    # overlap variance
    var_O = np.var(overlaps[1:-1], axis=1)  # trimmed to match central diff

    # deviation of zeta from mean
    # zbar = np.mean(zeta_raw) # computing the mean from data
    zbar = zeta_bar # use predefined mean from parameters
    z_dev = np.abs(zeta_raw[1:-1] - zbar)  # trimmed to match central diff
    z_signed = zeta_raw[1:-1] - zbar       # signed version

    return {
        'fall_total': fall_total,
        'C': C,
        'd': d,
        'var_O': var_O,
        'z_dev': z_dev,
        'z_signed': z_signed
    }

def crosscorr_peak(x, y, max_lag=200):
    """Computes the peak of the cross-correlation between x and y within ±max_lag."""
    x_std, y_std = standardize(x), standardize(y)
    corr_full = correlate(x_std, y_std, mode='full', method='fft') / len(x_std)
    lags_full = correlation_lags(len(x_std), len(y_std), mode='full')
    center = len(corr_full)//2
    corr = corr_full[center-max_lag:center+max_lag+1]
    lags = lags_full[center-max_lag:center+max_lag+1]
    idx = np.argmax(corr)
    return int(lags[idx]), float(corr[idx]), lags, corr

def surrogate_pvalue(x, y, n_surr=1000, max_lag=200, seed=None):
    """Circular shift surrogate test for p-value of cross-correlation."""
    rng = np.random.default_rng(seed)
    _, obs_val, _, _ = crosscorr_peak(x, y, max_lag=max_lag)
    surr_vals = np.empty(n_surr)
    T = len(y)
    for i in range(n_surr):
        shift = rng.integers(T)
        y_s = np.roll(y, shift)
        _, surr_vals[i], _, _ = crosscorr_peak(x, y_s, max_lag=max_lag)
    p = (np.sum(np.abs(surr_vals) >= np.abs(obs_val)) + 1) / (n_surr + 1) # two-tailed 
    return obs_val, p

def event_triggered_average(event_signal, response, pre=50, post=150, height_pct=80):
    """Event-triggered average of the response aligned to the peaks of event_signal."""
    event_signal = standardize(event_signal)
    response = standardize(response)
    height = np.percentile(event_signal, height_pct)
    peaks, _ = find_peaks(event_signal, height=height)
    epochs = []
    for p in peaks:
        if p - pre < 0 or p + post >= len(response):
            continue
        epochs.append(response[p-pre:p+post+1])
    if len(epochs) == 0:
        return None, peaks
    epochs = np.stack(epochs)
    mean_epoch = np.mean(epochs, axis=0)
    sem_epoch = np.std(epochs, axis=0) / np.sqrt(len(epochs))
    times = np.arange(-pre, post+1)
    return (times, mean_epoch, sem_epoch), peaks

# ----------------------------
# Loop over all runs
# ----------------------------
files_firing = sorted([f for f in os.listdir(firing_dir) if f.endswith('.npy')])
files_ou = sorted([f for f in os.listdir(ou_dir) if f.endswith('.npy')])
observables = ['C', 'd']

results = {}
for f_fire, f_ou in zip(files_firing, files_ou):
    sigs = compute_observables(os.path.join(firing_dir, f_fire), os.path.join(ou_dir, f_ou))
    file_stats = {}
    for obs in observables:
        x = sigs[obs]
        z = sigs['z_dev']
        lag, peak_val, _, _ = crosscorr_peak(x, z, max_lag=200)
        peak, pval = surrogate_pvalue(x, z, n_surr=1000, max_lag=200)
        file_stats[obs] = {'lag': lag, 'peak_val': peak_val, 'pval': pval}
    results[f_fire] = file_stats

# ----------------------------
# Summary printout
# ----------------------------
for obs in observables:
    lags_list = [results[f][obs]['lag'] for f in results]
    peaks = [results[f][obs]['peak_val'] for f in results]
    pvals = [results[f][obs]['pval'] for f in results]
    print(f"\nObservable: {obs}")
    print("lag median, mean, min, max:", np.median(lags_list), np.mean(lags_list), np.min(lags_list), np.max(lags_list))
    print("peak median:", np.median(peaks))
    print("pval < 0.05:", np.sum(np.array(pvals)<0.05), "/", len(pvals))
    print("p-values:", pvals)

# ----------------------------
# Event-triggered average per l'osservabile scelta
# ----------------------------
observable_to_plot = 'C'

example = files_firing[0]
sigs = compute_observables(os.path.join(firing_dir, example), os.path.join(ou_dir, files_ou[0]))
response = sigs[observable_to_plot]

eta_evt, peaks = event_triggered_average(sigs['z_dev'], response, pre=50, post=150, height_pct=90)
if eta_evt:
    times, mean_epoch, sem_epoch = eta_evt
    plt.figure(figsize=(8,4))
    plt.fill_between(times, mean_epoch-sem_epoch, mean_epoch+sem_epoch, alpha=0.3, label='SEM')
    plt.plot(times, mean_epoch, label=f'{observable_to_plot} (aligned to peak)')
    plt.axvline(0, color='k', linestyle='--', label='peak t=0')
    plt.xlabel('steps relative to peak', fontsize = 20)
    plt.ylabel(observable_to_plot, fontsize = 20)
    # plt.title(f'Event-triggered average (file={example}) — n_peaks={len(peaks)}')
    plt.legend(fontsize = 18)
    plt.grid(True)
    plt.show()
else:
    print("No events found for event-triggered average in example file.")

# --- Cross-correlazioni tra z_dev e tutte le osservabili (tranne var_O) ---
plt.figure(figsize=(10,5))

for obs in [o for o in observables if o != 'var_O']:
    response = sigs[obs]
    lag_peak, peak_val, lags, corr = crosscorr_peak(response, sigs['z_dev'], max_lag=200)
    plt.plot(lags, corr, label=f'${obs}(s)$ - peak=({peak_val:.2f} $\\pm$ {np.std(corr):.2f}) at lag={lag_peak}')
    plt.fill_between(lags, corr - np.std(corr), corr + np.std(corr), alpha=0.2)
    plt.axvline(lag_peak, linestyle=':', alpha=0.4)

plt.axvline(0, color='k', linestyle='--')
plt.xlabel('s (lag)', fontsize = 20)
plt.ylabel('R(s)', fontsize = 20)
# plt.title(f'Cross-correlation ({observable_to_plot} vs $|\\zeta - \\bar\\zeta|$) — peak={peak_val:.3f} at lag={lag_peak}')
plt.legend(fontsize = 12)
# plt.grid(True)
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "plots", "crosscorr.png"), dpi=300)
plt.show()

# ----------------------------
# Confronto ETA per vari tipi di eventi
# ----------------------------
def event_triggered_average_custom(event_signal, response, pre=100, post=100, mode="high", pct=80):
    """ETA per eventi high/low o positivi/negativi definiti dal percentile."""
    event_signal = standardize(event_signal)
    response = standardize(response)
    if mode == "high":
        thr = np.percentile(event_signal, pct)
        peaks, _ = find_peaks(event_signal, height=thr)
    elif mode == "low":
        thr = np.percentile(event_signal, 100 - pct)
        peaks, _ = find_peaks(-event_signal, height=-thr)
    else:
        raise ValueError("mode must be 'high' or 'low'")
    epochs = []
    for p in peaks:
        if p - pre < 0 or p + post >= len(response):
            continue
        epochs.append(response[p-pre:p+post+1])
    if len(epochs) == 0:
        return None, peaks
    epochs = np.stack(epochs)
    mean_epoch = np.mean(epochs, axis=0)
    sem_epoch = np.std(epochs, axis=0) / np.sqrt(len(epochs))
    times = np.arange(-pre, post+1)
    return (times, mean_epoch, sem_epoch), peaks

# --- ETA: alti e bassi di |ζ - ζ̄| ---
eta_high, _ = event_triggered_average_custom(sigs['z_dev'], response, mode="high", pct=80)
eta_low, _ = event_triggered_average_custom(sigs['z_dev'], response, mode="low", pct=20)

# --- ETA: positivi e negativi di (ζ - ζ̄) ---
eta_pos, _ = event_triggered_average_custom(sigs['z_signed'], response, mode="high", pct=90)
eta_neg, _ = event_triggered_average_custom(sigs['z_signed'], response, mode="low", pct=90)

# --- Plot comparativo ---
plt.figure(figsize=(10,6))

if eta_high:
    t, m, e = eta_high
    plt.plot(t, m, label='$z(t) > Q_{80}$', color='C0')
    plt.fill_between(t, m-e, m+e, alpha=0.2, color='C0')

if eta_low:
    t, m, e = eta_low
    plt.plot(t, m, label='$z(t) < Q_{20}$', color='C1')
    plt.fill_between(t, m-e, m+e, alpha=0.2, color='C1')

if eta_pos:
    t, m, e = eta_pos
    plt.plot(t, m, label='$\\zeta-\\bar{\\zeta} > Q_{90}$', color='C2')
    plt.fill_between(t, m-e, m+e, alpha=0.2, color='C2')

if eta_neg:
    t, m, e = eta_neg
    plt.plot(t, m, label='$\\zeta-\\bar{\\zeta} < Q_{10}$', color='C3')
    plt.fill_between(t, m-e, m+e, alpha=0.2, color='C3')

plt.axvline(0, color='k', linestyle='--')
plt.xlabel('$t-t_\\text{event}$', fontsize = 20)
plt.ylabel(f'Standardized ${observable_to_plot}(t)$ [$\\sigma$ units]', fontsize=20)
plt.legend(fontsize = 14)
# plt.title(f'Event-triggered average of {observable_to_plot} for event conditions')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", f"{multiple_dir_name}_{N}", "plots", "eta_comparison.png"), dpi=300)
plt.show()

# ----------------------------
# ETA aggregata su tutte le run
# ----------------------------
def compute_eta_all_runs(files_firing, files_ou, observable, mode="high", pct=90, pre=50, post=150):
    """Compute and average ETA of observable w.r.t. events on |zeta - zbar| across runs."""
    all_etas = []
    for f_fire, f_ou in zip(files_firing, files_ou):
        sigs = compute_observables(os.path.join(firing_dir, f_fire), os.path.join(ou_dir, f_ou))
        response = standardize(sigs[observable])
        z_dev = standardize(sigs['z_dev'])
        eta_evt, peaks = event_triggered_average_custom(z_dev, response, pre=pre, post=post, mode=mode, pct=pct)
        if eta_evt is not None:
            times, mean_epoch, _ = eta_evt
            # Ri-standardizza la curva della run per evitare run dominate
            mean_epoch_std = (mean_epoch - np.mean(mean_epoch)) / (np.std(mean_epoch) + 1e-12)
            all_etas.append(mean_epoch_std)
    if len(all_etas) == 0:
        return None
    all_etas = np.stack(all_etas)
    mean_eta = np.mean(all_etas, axis=0)
    sem_eta = np.std(all_etas, axis=0) / np.sqrt(len(all_etas))
    return times, mean_eta, sem_eta, len(all_etas)

# --- Calcola ETA su tutte le run ---
eta_all = compute_eta_all_runs(files_firing, files_ou, observable_to_plot, mode="high", pct=90, pre=50, post=150)

if eta_all:
    times, mean_eta, sem_eta, n_runs = eta_all
    plt.figure(figsize=(8,4))
    plt.fill_between(times, mean_eta - sem_eta, mean_eta + sem_eta, alpha=0.3, label='SEM across runs')
    plt.plot(times, mean_eta, label=f'Mean ETA of {observable_to_plot} across {n_runs} runs')
    plt.axvline(0, color='k', linestyle='--', label='event t=0')
    plt.xlabel('$t-t_{event}$', fontsize=20)
    plt.ylabel(f'Standardized ${observable_to_plot}(t)$  [$\\sigma$ units]', fontsize=20)
    plt.legend(fontsize=18)
    # plt.title(f'ETA across runs ({observable_to_plot} vs |ζ−ζ̄| > 90th pct)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid ETA segments found across runs.")

# ----------------------------
# Cross-correlazione aggregata su tutte le run
# ----------------------------
def compute_crosscorr_all_runs(files_firing, files_ou, observable, max_lag=200):
    """Compute and average cross-correlation across runs (standardized per run)."""
    all_corrs = []
    for f_fire, f_ou in zip(files_firing, files_ou):
        sigs = compute_observables(os.path.join(firing_dir, f_fire), os.path.join(ou_dir, f_ou))
        x = standardize(sigs[observable])
        z = standardize(sigs['z_dev'])
        # cross-correlation normalizzata
        corr_full = correlate(x, z, mode='full', method='fft') / len(x)
        lags_full = correlation_lags(len(x), len(z), mode='full')
        center = len(corr_full) // 2
        corr = corr_full[center - max_lag : center + max_lag + 1]
        lags = lags_full[center - max_lag : center + max_lag + 1]
        # standardizza la curva della singola run
        corr_std = (corr - np.mean(corr)) / (np.std(corr) + 1e-12)
        all_corrs.append(corr_std)

    if len(all_corrs) == 0:
        return None

    all_corrs = np.stack(all_corrs)
    mean_corr = np.mean(all_corrs, axis=0)
    sem_corr = np.std(all_corrs, axis=0) / np.sqrt(len(all_corrs))
    return lags, mean_corr, sem_corr, len(all_corrs)

# --- Calcola cross-correlation media su tutte le run ---
cross_all = compute_crosscorr_all_runs(files_firing, files_ou, observable_to_plot, max_lag=200)

if cross_all:
    lags, mean_corr, sem_corr, n_runs = cross_all
    plt.figure(figsize=(8,4))
    plt.fill_between(lags, mean_corr - sem_corr, mean_corr + sem_corr, alpha=0.3, label='SEM across runs')
    plt.plot(lags, mean_corr, label=f'Mean cross-corr. of {observable_to_plot} across {n_runs} runs')
    plt.axvline(0, color='k', linestyle='--', label='lag = 0')
    plt.xlabel('Lag [steps]', fontsize=20)
    plt.ylabel(f'Standardized $R_{{{observable_to_plot}, |\\zeta-\\bar\\zeta|}}(\\tau)$ [$\\sigma$ units]', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid cross-correlations found across runs.")

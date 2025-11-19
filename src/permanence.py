"""Calculate the mean time of permanence for overlaps above a threshold."""

import os
import numpy as np
import matplotlib.pyplot as plt
from modules.dynamics import calculate_pattern_overlaps
from modules.activation import sigmoid_function, inverse_sigmoid_function
from parameters import (
        N,
        t_end,
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

npy_path = os.path.join(os.path.dirname(__file__), "..", f'{multiple_dir_name}{"_"}{N}', 'npy')
patterns_path = os.path.join(npy_path, 'memory_patterns.npy')
firing_rates_path = os.path.join(npy_path, 'firing_rates')
threshold = 0.8

import numpy as np

def calculate_permanence_single_trial(overlaps, threshold=0.8, dt=None, smooth_window=5):
    """
    Compute permanence times for overlaps above a threshold,
    ensuring that each time point contributes to at most one pattern:
    the pattern with the highest overlap above threshold.
    Episodes that remain active at the end are excluded.

    Parameters
    ----------
    overlaps : np.ndarray
        Array of shape (n_timepoints, n_patterns) containing pattern overlaps over time.
    threshold : float, default=0.8
        Minimum overlap value to consider a pattern "active".
    dt : float or None, default=None
        Time step between consecutive time points. If None, durations
        are expressed in number of steps instead of time units.

    Returns
    -------
    permanence_times : np.ndarray
        1D array containing the durations of all closed permanence episodes.
        The last episode is ignored if still active at the end of the trial.
    """
    # Validate input
    if overlaps.ndim != 2:
        raise ValueError("`overlaps` must be a 2D array of shape (n_timepoints, n_patterns).")

    n_timepoints, n_patterns = overlaps.shape
    step_value = dt if dt is not None else 1.0

    # smoothing
    if smooth_window > 1:
        cumsum = np.cumsum(np.insert(overlaps, 0, 0, axis=0), axis=0)
        overlaps = (cumsum[smooth_window:] - cumsum[:-smooth_window]) / smooth_window

    permanence_times = []
    current_pattern = None
    current_duration = 0.0

    for t in range(overlaps.shape[0]):
        max_idx = np.argmax(overlaps[t])

        if current_pattern is None:
            # start new dwell
            current_pattern = max_idx
            current_duration = step_value
        else:
            if max_idx == current_pattern:
                # continue dwell
                current_duration += step_value
            else:
                # close dwell and start new one if above threshold
                permanence_times.append(current_duration)
                if overlaps[t, max_idx] > threshold:
                    current_pattern = max_idx
                    current_duration = step_value
                else:
                    current_pattern = None
                    current_duration = 0.0

    # do not add last dwell if still ongoing
    return np.array(permanence_times)

# Prepare phi function parameters
phi_params = {
            'r_m': phi_r_m,
            'beta': phi_beta,
            'x_r': phi_x_r,
        }

# Prepare g function parameters
g_params = {
            'q_f': g_q,
            'x_f': g_x
        }

eta = np.load(patterns_path)


def calculate_permanence_statistics(threshold=0.8, dt=None, file_path="firing_rates", smooth_window=5):
    """
    Calculate permanence time statistics for all overlap data in a directory.

    Parameters
    ----------
    threshold : float
        Threshold for detecting high-overlap episodes.
    dt : float or None
        Time step between consecutive points.
    file_path : str
        Directory containing firing rate files (.npy format expected).
    smooth_window : int
        Window size for smoothing overlaps with a moving average.

    Returns
    -------
    mean_permanence : float
        Mean permanence time across all files, patterns, and episodes.
    std_permanence : float
        Standard deviation of permanence times.
    """
    permanence_all = []
    fnames = os.listdir(file_path)
    print(f"Found {len(fnames)} files in {file_path} for permanence calculation.")

    for fname in fnames:
        if fname.endswith(".npy"):
            firing_rates = np.load(os.path.join(file_path, fname))
            overlaps = calculate_pattern_overlaps(firing_rates, eta, phi_params, g_params, use_numba=use_numba, use_g=use_g)
            permanence = calculate_permanence_single_trial(overlaps, threshold=threshold, dt=dt, smooth_window=smooth_window)
            permanence_all.extend(permanence)

    if len(permanence_all) == 0:
        return np.nan, np.nan

    permanence_all = np.array(permanence_all)
    return np.mean(permanence_all), np.std(permanence_all), permanence_all

if __name__ == "__main__":
    # check if permanence_times.npy already exists
    if os.path.exists(os.path.join(npy_path, "permanence_times.npy")):
        print("permanence_times.npy already exists. Loading data...")
        all_perm = np.load(os.path.join(npy_path, "permanence_times.npy"))
        mean_perm = np.mean(all_perm)
        std_perm = np.std(all_perm)
    else:
        mean_perm, std_perm, all_perm = calculate_permanence_statistics(threshold=threshold, dt=dt, file_path=firing_rates_path, smooth_window=100)
    print(f"Mean permanence time above threshold {threshold}: {mean_perm:.2f}")
    print(f"Standard deviation of permanence times: {std_perm:.2f}")
    np.save(os.path.join(npy_path, "permanence_times.npy"), all_perm)
    # draw the histogram of all permanence times
    # print the fraction of permanence times lower than 200
    fraction_below_200 = np.sum(all_perm < 100) / len(all_perm)
    print(f"Fraction of permanence times below 200: {fraction_below_200:.2f}") 
    plt.axvline(mean_perm, color='r', linestyle='--', label='Mean = {:.2f}'.format(mean_perm))
    plt.hist(all_perm, bins=50, density=False)
    plt.xlabel("Dwell time", fontsize=20)
    plt.ylabel("Counts", fontsize=20)
    # fill between mean +/- std
    plt.fill_betweenx([0, plt.ylim()[1]], np.maximum(mean_perm - std_perm, 0), mean_perm + std_perm, alpha=0.2, label=f"Std={std_perm:.2f}")
    plt.title(f"Dwell Times (threshold={threshold}, total={len(all_perm)})", fontsize=20)
    plt.legend(fontsize=18)
    plt.savefig(os.path.join(npy_path, "..", "plots", "dwell.png"))
    plt.show()


    # plot the smoothed overlaps for a sample trial
    sample_firing_rates = np.load(os.path.join(firing_rates_path, os.listdir(firing_rates_path)[0]))
    sample_overlaps = calculate_pattern_overlaps(sample_firing_rates, eta, phi_params, g_params, use_numba=use_numba, use_g=use_g)
    n_timepoints, n_patterns = sample_overlaps.shape
    time = np.arange(n_timepoints) * dt
    # smoothing overlaps with a moving average of window size 100
    window_size = 100
    cumsum = np.cumsum(np.insert(sample_overlaps, 0, 0, axis=0), axis=0)
    smoothed_overlaps = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    time_smoothed = time[window_size - 1:]
    for p in range(n_patterns):
        plt.plot(time_smoothed, smoothed_overlaps[:, p], label=f"Pattern {p}")
    plt.axhline(threshold, color='k', linestyle='--', label='Threshold')
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Overlap", fontsize=20)
    plt.title("Smoothed Pattern Overlaps (Sample Trial)", fontsize=20)
    plt.legend(fontsize=12)
    plt.show()

"""
energy_plots.py 
This module generates energy plots for the neural network simulation.
"""
import os
from flask import app
import numpy as np
import matplotlib.pyplot as plt
from sklearn import naive_bayes
from sympy import E, use
from modules.activation import step_function
from modules.connectivity import generate_connectivity_matrix, plot_matrix
from modules.dynamics import simulate_network, calculate_pattern_overlaps
from modules.energy import compute_energy

# Import all parameters from parameters.py
from parameters import (
    # Connectivity matrix parameters
    N,
    p,
    q,
    c,
    A_S,
    phi_function_type,
    phi_amplitude,
    phi_beta,
    phi_r_m,
    phi_x_r,
    f_type,
    f_r_m,
    f_beta,
    f_x_r,
    f_q,
    f_x,
    g_type,
    g_r_m,
    g_beta,
    g_x_r,
    g_q,
    g_x,
    pattern_mean,
    pattern_sigma,
    enforce_max_correlation,
    max_correlation,
    alpha,
    apply_sigma_cutoff,
    apply_phi_to_patterns,
    apply_er_to_asymmetric,

    # Simulation parameters
    tau,
    t_start,
    t_end,
    dt,
    init_cond_type,
    pattern_idx,
    noise_level,
    use_symmetric_only,
    model_type,
    use_numba,
    use_g,
    use_ou,
    tau_zeta,
    zeta_bar,
    sigma_zeta,
    constant_zeta,

    # Visualization parameters
    n_display,
    show_sim_plots,

    # Seed for reproducibility
    seed
    )

def main():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "simulation_results")
    npy_dir = os.path.join(output_dir, "npy")

    # Calculate and plot energy
    print("Calculating and plotting energy...")
    # Loading connectivity matrices and history from npy files
    connectivity_path = os.path.join(npy_dir, "connectivity_total.npy")
    overlaps_path = os.path.join(npy_dir, "pattern_overlaps.npy")
    original_symm_path = os.path.join(npy_dir, "connectivity_symmetric.npy")
    history_path = os.path.join(npy_dir, "firing_rates.npy")
    phi_memory_patterns_path = os.path.join(npy_dir, "phi_memory_patterns.npy")
    overlaps = np.transpose(np.load(overlaps_path))
    W, h = np.load(connectivity_path), np.load(history_path)
    W_symm = np.load(original_symm_path)
    phi_memory_patterns = np.load(phi_memory_patterns_path)

    # Apply step function to phi_memory_patterns (maybe it's necessary: memories compose the matrix with the step applied)
    # phi_memory_patterns = step_function(phi_memory_patterns, f_q, f_x)
    # h = step_function(h, f_q, f_x)

    # compute energy for each pattern and print it
    pattern_energies, _, _ = compute_energy(W_symm, phi_memory_patterns, 'sigmoid', phi_beta, phi_r_m, phi_x_r, activation_term=False)
    for i in range(pattern_energies.shape[0]):
        print(f"Energy for pattern {i+1}: {pattern_energies[i]:.4f}")

    # plot energy trajectory (computing symm, asymm, total)
    E_symm_traj, syn_terms, act_terms = compute_energy(W_symm, h, 'sigmoid', phi_beta, phi_r_m, phi_x_r, activation_term=True)
    E_asymm_traj, _, _ = compute_energy(W - W_symm, h, 'sigmoid', phi_beta, phi_r_m, phi_x_r, activation_term=True)
    E_total_traj = E_symm_traj + E_asymm_traj

    # time steps
    t = np.arange(E_symm_traj.shape[0])  # but time steps are dt seconds apart so multiply by dt
    t = t * dt

    # plot energy and overlaps in a subplot under the first one
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, E_symm_traj, label='Symmetric Energy', color='green')
    plt.ylim(min(E_symm_traj[100:])-10, max(E_symm_traj[100:])+10)
    for i in range(3):
        # pattern_energies[i] = 4.47 * pattern_energies[i] - 33150
        pattern_energies[i] =  pattern_energies[i] 
        plt.plot(t, np.full_like(t, pattern_energies[i]), label=f'Pattern {i+1} Energy', linestyle='--')
    plt.subplot(2, 1, 2)
    for i in range(p):
        plt.plot(t, overlaps[i], label=f'Pattern {i+1} Overlap')
    plt.xlabel('time steps')
    plt.savefig(os.path.join(output_dir, "energy_and_overlaps.png"))
    if show_sim_plots:
        plt.show()

"""
    # Plot total, symmetric, and asymmetric energy for original and 'new' matrices
    plt.figure(figsize=(12, 6))
    plt.plot(t, E_total_traj, label='Total Energy', color='blue')
    plt.plot(t, E_symm_traj, label='Symmetric Energy', color='green')
    plt.plot(t, E_asymm_traj, label='Asymmetric Energy', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    plt.title('Energy Trajectories (Original symmetric and asymmetric components)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "energy_trajectories_original.png"))
    plt.show()
    plt.figure(figsize=(12, 6))
    W_symm_new = 0.5 * (W + W.T)
    W_antisym_new = 0.5 * (W - W.T)
    E_symm_new_traj, _, _ = compute_energy(W_symm_new, h, 'sigmoid', phi_beta, phi_r_m, phi_x_r)
    E_asymm_new_traj, _, _ = compute_energy(W_antisym_new, h, 'sigmoid', phi_beta, phi_r_m, phi_x_r)
    plt.plot(t, E_total_traj, label='Total Energy', color='blue')
    plt.plot(t, E_symm_new_traj, label='Symmetric Energy (New)', color='green')
    plt.plot(t, E_asymm_new_traj, label='Antisymmetric Energy (New)', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    plt.title('Energy Trajectories (New symmetric and antisymmetric components)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(output_dir, "energy_trajectories_new.png"))
    if show_sim_plots:
        plt.show()
"""



if __name__ == "__main__":
    main()

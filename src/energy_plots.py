"""
energy_plots.py 
This module generates energy plots for the neural network simulation.
"""
from math import e
import os
import numpy as np
import matplotlib.pyplot as plt
from modules.activation import step_function
from modules.connectivity import generate_connectivity_matrix, plot_matrix
from modules.dynamics import simulate_network, calculate_pattern_overlaps
from modules.energy import compute_energy, compute_forces, project_flux_on_dynamics

# Import all parameters from parameters.py
from parameters import (
    # Connectivity matrix parameters
    N,
    p,
    q,
    c,
    A_S,
    phi_beta,
    phi_r_m,
    phi_x_r,
    f_q,
    f_x,
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
    output_dir = os.path.join(os.path.dirname(__file__), "..", "simulation_results_silly")
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
    pattern_energies, _, _ = compute_energy(W_symm, phi_memory_patterns, tau, phi_beta, phi_r_m, phi_x_r, activation_term=False)
    for i in range(pattern_energies.shape[0]):
        print(f"Energy for pattern {i+1}: {pattern_energies[i]:.4f}")

    # energy trajectory (computing symm, asymm, total)
    E_symm_traj, syn_terms, act_terms = compute_energy(W_symm, h, tau, phi_beta, phi_r_m, phi_x_r, activation_term=True)
    E_asymm_traj, _, _ = compute_energy(W - W_symm, h, tau, phi_beta, phi_r_m, phi_x_r, activation_term=True)
    E_total_traj = E_symm_traj + E_asymm_traj

    # compute symmetric force and asymmetric force (flux = F_ASYMM_AVG)
    F_symm, F_asymm, _, flux = compute_forces(W_symm, W - W_symm, h, tau, phi_beta, phi_r_m, phi_x_r)
    # compute the energy gradient (numerically)
    energy_gradient = -np.gradient(E_total_traj)


    # time in time steps (not seconds)
    t = np.arange(E_symm_traj.shape[0])  # but time steps are dt seconds apart so multiply by dt
    t = t * dt

    # compute norms
    F_symm_norm = np.linalg.norm(F_symm, axis=1)
    F_asymm_norm = np.linalg.norm(F_asymm, axis=1)

    # plot energy, flux and overlaps
    plt.figure(figsize=(12, 8))

    # --- subplot: Energy (left) and Flux (right) ---
    ax1 = plt.subplot(411)
    ax2 = ax1.twinx()  # create second y-axis

    # Energy on left y-axis
    ax1.plot(t, E_symm_traj, label='E (symm)', color='green')
    ax1.set_ylabel("Energy")
    ax1.set_ylim(min(E_symm_traj[300:]) - 10, max(E_symm_traj[300:]) + 10)
    ax2.plot(t, energy_gradient, label='- $\\nabla$ E', color='orange', alpha=0.7)
    ax2.set_ylabel("Energy Gradient")
    ax2.set_ylim(min(energy_gradient[300:]) - 1, max(energy_gradient[300:]) + 1)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # --- Subplot: Forces (symmetrical and asymmetrical) --- 
    ax1 = plt.subplot(412)
    ax2 = ax1.twinx()  # create second y-axis 
    ax1.plot(t, F_symm_norm, label='$|F^{symm}|$', color='blue', alpha=0.7)
    ax2.plot(t, F_asymm_norm, label='$|F^{asymm}|$', color='red', alpha=0.7)
    ax1.plot(t, np.full_like(t, 0.0), linestyle='--', color='gray', alpha=0.5)  # horizontal line at y=0
    ax1.set_ylabel("$|F^{symm}|$")
    ax2.set_ylabel("$|F^{asymm}|$")
    # ylims with min and max
    ax1.set_ylim(-100, max(F_symm_norm[100:]) + 1)
    ax2.set_ylim(-100, max(F_symm_norm[100:]) + 1)
    # ax2.set_ylim(min(F_asymm_norm[100:]) - 1, max(F_asymm_norm[100:]) + 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    # --- subplot: Overlaps ---
    plt.subplot(413)
    for i in range(p):
        plt.plot(t, overlaps[i], label=f'P {i+1}')
    plt.ylabel("Overlap")
    plt.legend()

    plt.subplot(414)
    # projection of the flux on the dynamics
    print(np.shape(h), np.shape(F_symm), np.shape(F_asymm))
    proj_flux_symm = project_flux_on_dynamics(h, F_symm)
    proj_flux_asymm = project_flux_on_dynamics(h, F_asymm)
    plt.plot(t, proj_flux_symm, label='Projection of flux on dynamics (symm)', color='purple')
    plt.plot(t, proj_flux_asymm, label='Projection of flux on dynamics (asymm)', color='orange')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel("Projection")
    plt.xlabel("Time (s)")
    plt.legend()


    # Save + show
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "energy_and_overlaps.png"))
    if show_sim_plots:
        plt.show()

    plt.figure(figsize=(12, 8))
    plt.subplot(211)
    for i in range(p):
        plt.plot(t, overlaps[i], label=f'P {i+1}')
    plt.ylabel("Overlap")
    plt.legend()
    plt.subplot(212)
    plt.plot(t, proj_flux_symm, label='Symm', color='purple')
    plt.plot(t, proj_flux_asymm, label='Asymm', color='orange')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.ylabel("Flux Proj on Dynamics")
    plt.xlabel("$t$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overlaps_and_flux_projection.png"))
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
    E_symm_new_traj, _, _ = compute_energy(W_symm_new, h, tau, 'sigmoid', phi_beta, phi_r_m, phi_x_r)
    E_asymm_new_traj, _, _ = compute_energy(W_antisym_new, h, tau, 'sigmoid', phi_beta, phi_r_m, phi_x_r)
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

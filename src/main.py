"""
This file intends to simulate the dynamics of the NN with many different initial conditions
"""

import os
import time

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from sympy import im
from torch import inverse, mul


from modules.connectivity import generate_connectivity_matrix, plot_matrix
from modules.dynamics import simulate_network, calculate_pattern_overlaps, initial_condition_creator 
from modules.activation import sigmoid_function, relu_function, inverse_sigmoid_function

# ================ Simulation Parameters ================ 

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
    ou_non_neg,
    tau_zeta,
    zeta_bar,
    sigma_zeta,
    fixed_zeta,

    # Visualization parameters
    n_display,
    show_sim_plots,
    verbose,
    single_dir_name,
    multiple_dir_name,

    # number of runs
    runs,
    # Import connectivity or not
    import_connectivity,
    # Seed for reproducibility
    seed
    )

# ===================================================

def single_simulation(addition, W_S, W_A_sim, W, eta, phi_eta, t_span, ou_params, seed = seed, verbose = False):
    """
    Single simulation of the network dynamics.
    seed and addition are needed in order to run multiple simulations
    addition is a string that is used to save and identify the simulation run
    """
    time_start = time.time() # Start time measurement
    np.random.seed(seed)  # for reproducibility
    # names of the output directories (already created in multiple_simulations, see below)
    output_dir = os.path.join(os.path.dirname(__file__), "..", f'{multiple_dir_name}{"_"}{N}')
    plot_dir = os.path.join(output_dir, 'plots')
    npy_dir = os.path.join(output_dir, "npy")

    # Prepare g function parameters
    g_params = {
        'q_f': g_q,
        'x_f': g_x
    }
    # Prepare phi function parameters
    phi_params = {
        'r_m': phi_r_m,
        'beta': phi_beta,
        'x_r': phi_x_r,
    }
    
    # Set up initial condition with proper noise calculation, no pattern index here (random) (i am changing that)
    initial_condition = initial_condition_creator(init_cond_type, N, g_params=g_params, phi_params=phi_params, p=p, eta=eta, noise_level=noise_level, seed = seed)

    # Run simulation with same φ parameters used in connectivity generation
    t, u, zeta = simulate_network(
        W_S=W_S,
        W_A=W_A_sim,
        t_span=t_span,
        dt=dt,
        tau=tau,
        initial_condition=initial_condition,
        use_ou=use_ou,
        ou_params=ou_params,
        r_m=phi_r_m,
        beta=phi_beta,
        x_r=phi_x_r,
        model_type=model_type,
        fixed_zeta=fixed_zeta if not use_ou else None,
        use_numba=use_numba,
        ou_non_neg=ou_non_neg,
        seed=seed
    )

    logger.info(f"\n {addition} - simulation completed successfully!")

    # Calculate pattern overlaps
    overlaps = None
    if p > 0:

        overlaps = calculate_pattern_overlaps(u,
                                              eta,
                                              phi_params,
                                              g_params,
                                              use_numba=use_numba,
                                              use_g=use_g)

    # # Save OU process
    # zeta_array = np.asarray(zeta)
    # if zeta_array.size == 1:
    #     # Save constant zeta as array
    #     zeta_full = np.full_like(t, float(zeta_array.item()))
    #     np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_full)
    # else:
    #     np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_array)
    
    # Calculate and save firing rates
    phi_u = sigmoid_function(u, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)
    np.save(os.path.join(npy_dir, "firing_rates" , f'{"firing_rates_"}{addition}{".npy"}'), phi_u)

    # Save OU process
    zeta_array = np.asarray(zeta)
    if zeta_array.size == 1:
        # Save constant zeta as array
        zeta_full = np.full_like(t, float(zeta_array.item()))
        np.save(os.path.join(npy_dir, "ou_process" , f'{"ou_process_"}{addition}{".npy"}'), zeta_full)
    else:
        np.save(os.path.join(npy_dir, "ou_process" , f'{"ou_process_"}{addition}{".npy"}'), zeta_array)

    # # Save pattern overlaps if available
    # if p > 0 and overlaps is not None:
    #     np.save(os.path.join(npy_dir, "pattern_overlaps.npy"), overlaps)
    
    plt.figure()
    if p > 0 and overlaps is not None:
        for i in range(p):
            plt.plot(t, overlaps[:, i], label=f'Pattern {i+1}', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Pattern Overlap')
        plt.title(f'Overlaps ($\\bar{{\\zeta}} = {zeta_bar} $, $\\sigma_\\zeta = {sigma_zeta}$, $\\tau_\\zeta = {tau_zeta}$, A={A_S}, seed = {seed})')
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No patterns to display', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Pattern Overlaps')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{"pattern_overlaps_"}{addition}{".png"}'), dpi=300)
    plt.close()

    logger.info(f"Saved plots to {plot_dir}")
    time_end = time.time()
    elapsed_time = time_end - time_start
    logger.info(f"Elapsed time for this simulation: {elapsed_time:.2f} seconds")

# =================================================================


def multiple_simulations():
    """
    Run multiple simulations with different initial conditions but same connectivity matrix.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create the main output directory
    output_dir = os.path.join(os.path.dirname(__file__), "..", f'{multiple_dir_name}{"_"}{N}')
    os.makedirs(output_dir, exist_ok=True)
    # Create the npy and plots subdirectories
    npy_dir = os.path.join(output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    logger.info(f"Output directories created at: {output_dir}")

    # firing rates subdirectory
    firing_rates_dir = os.path.join(npy_dir, "firing_rates")
    os.makedirs(firing_rates_dir, exist_ok=True)

    # ou process subdirectory
    ou_process_dir = os.path.join(npy_dir, "ou_process")
    os.makedirs(ou_process_dir, exist_ok=True)

    connectivity_path_symmetric = os.path.join(npy_dir, "connectivity_symmetric.npy")
    connectivity_path_asymmetric = os.path.join(npy_dir, "connectivity_asymmetric.npy")
    phi_eta_path = os.path.join(npy_dir, "phi_memory_patterns.npy")
    eta_path = os.path.join(npy_dir, "memory_patterns.npy")

    def load_connectivity():
        W_S = np.load(connectivity_path_symmetric)
        W_A = np.load(connectivity_path_asymmetric)
        W = W_S + W_A
        phi_eta = np.load(phi_eta_path)
        eta = np.load(eta_path)
        logger.info(f"Imported connectivity matrices from {npy_dir}")
        return W_S, W_A, W, eta, phi_eta
    
    def generate_connectivity():
        logger.info(f"Generating connectivity matrices...")
        W_S, W_A, W, eta, phi_eta = generate_connectivity_matrix(
            N=N,
            p=p,
            q=q,
            c=c,
            A_S=A_S,
            f_q=f_q,
            f_x=f_x,
            g_q=g_q,
            g_x=g_x,
            pattern_mean=pattern_mean,
            pattern_sigma=pattern_sigma,
            phi_beta=phi_beta,
            phi_r_m=phi_r_m,
            phi_x_r=phi_x_r,
            apply_phi_to_patterns=apply_phi_to_patterns,
            apply_er_to_asymmetric=apply_er_to_asymmetric,
            enforce_max_correlation=enforce_max_correlation)
        logger.info("Matrices generated successfully!")
        return W_S, W_A, W, eta, phi_eta

    # Generate or import connectivity matrices
    if import_connectivity:
        try:
            W_S, W_A, W, eta, phi_eta = load_connectivity()
        except Exception as e:
            logger.warning(f"Connectivity import failed ({e}). Generating new matrices.")
            W_S, W_A, W, eta, phi_eta = generate_connectivity()
    else:
        W_S, W_A, W, eta, phi_eta = generate_connectivity()

    # Display matrix statistics if verbose 
    if verbose:
        logger.info("\nMatrix statistics:")
        logger.info(
            f"W_S: Mean={W_S.mean():.6f}, Std={W_S.std():.4f}, Min={W_S.min():.4f}, Max={W_S.max():.4f}"
        )
        logger.info(
        f"W_A: Mean={W_A.mean():.8f}, Std={W_A.std():.4f}, Min={W_A.min():.4f}, Max={W_A.max():.4f}"
        )
        logger.info(
        f"W: Mean={W.mean():.8f}, Std={W.std():.4f}, Min={W.min():.4f}, Max={W.max():.4f}"
        )

        # Check symmetry
        symmetry_diff = W_S - W_S.T
        max_symmetry_diff = np.max(np.abs(symmetry_diff))
        logger.info(f"Max |W_S - W_S^T|: {max_symmetry_diff:.8f}")

        # Check asymmetry
        asymmetry_diff = W_A - W_A.T
        max_asymmetry_diff = np.max(np.abs(asymmetry_diff))
        logger.info(f"Max |W_A - W_A^T|: {max_asymmetry_diff:.8f}")
    else:
        None

    # Calculate and print pattern correlation analysis
    if not import_connectivity:
        from modules.connectivity import calculate_pattern_correlation_matrix, plot_pattern_correlation_matrix
        correlation_matrix, actual_max_correlation = calculate_pattern_correlation_matrix(eta)


        if verbose:
            logger.info(f"\nMemory Pattern Correlation Analysis:")
            logger.info(f"Number of patterns: {eta.shape[0]}")
            logger.info(f"Correlation constraint enforced: {enforce_max_correlation}")
            logger.info(f"Actual maximum correlation: {actual_max_correlation:.3f}")
        else:
            None

        if N < 2001: # Only plot matrices for smaller networks
            # Plot matrices
            _, _, _ = plot_pattern_correlation_matrix(eta,
                                                    enforce_max_correlation=enforce_max_correlation,
                                                    ax=None, output_dir=output_dir)

            fig1 = plt.figure(figsize=(15, 5))
            ax1 = fig1.add_subplot(131)
            ax2 = fig1.add_subplot(132)
            ax3 = fig1.add_subplot(133)

            plot_matrix(W_S, "Symmetric Component", ax=ax1)
            plot_matrix(W_A, "Asymmetric Component", ax=ax2)
            plot_matrix(W, "Total Connectivity", ax=ax3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "connectivity_matrices.png"), dpi=150)
            logger.info(f"Saved matrices visualization")
        else:
            logger.info(
                "\nSkipping matrix plots for large network (N > 2000) to save time and resources."
            )
    else:
        None

    # Print initial condition type
    logger.info(f"Initial condition type: {init_cond_type}")

     # Simulation time span
    t_span = (t_start, t_end)

    # Set up OU parameters
    if use_ou:
        ou_params = {
            'tau_zeta': tau_zeta,
            'zeta_bar': zeta_bar,
            'sigma_zeta': sigma_zeta
        }
        if verbose:
            logger.info(
                f"Using Ornstein-Uhlenbeck process: τ_ζ={tau_zeta}, ζ̄={zeta_bar}, σ_ζ={sigma_zeta}"
            )
        else: None
    else:
        ou_params = None
        logger.info(f"Using fixed ζ") if verbose else None

    # Choose which connectivity matrix to use
    if use_symmetric_only:
        W_A_sim = np.zeros_like(W_A)
        logger.info("Using only the symmetric component (W^S) for dynamics.")
    else:
        W_A_sim = W_A
        logger.info("Using both symmetric and asymmetric components for dynamics.") if verbose else None

    logger.info(f"\nRunning {model_type} dynamics simulation, time spanning from {t_start} to {t_end}, τ = {tau}, φ function: sigmoid (β={phi_beta}, r_m={phi_r_m}, x_r={phi_x_r})"
    ) 

    if verbose:
        if use_numba and N > 1000:
            logger.info(f"Numba optimization enabled for {N} neurons")
        elif not use_numba:
            logger.info("Numba optimization disabled")
    else:
        None

    # Save simulation parameters
    params = {
        'N': N,
        'p': p,
        'q': q,
        'c': c,
        'A_S': A_S,
        'phi_beta': phi_beta,
        'phi_r_m': phi_r_m,
        'phi_x_r': phi_x_r,
        'f_q': f_q,
        'f_x': f_x,
        'g_q': g_q,
        'g_x': g_x,
        'pattern_mean': pattern_mean,
        'pattern_sigma': pattern_sigma,
        'enforce_max_correlation': enforce_max_correlation,
        'apply_phi_to_patterns': apply_phi_to_patterns,
        'apply_er_to_asymmetric': apply_er_to_asymmetric,
        'use_ou': use_ou,
        'use_g': use_g,
        'tau_zeta': tau_zeta if use_ou else None,  # Only include if using OU
        'zeta_bar': zeta_bar if use_ou else None,  # Only include if using OU
        'sigma_zeta': sigma_zeta if use_ou else None,  # Only include if using OU
        'fixed_zeta': fixed_zeta if not use_ou else None,  # Only include if not using OU
        't_start': t_start,
        't_end': t_end,
        'dt': dt,
        'tau': tau,
        'init_cond_type': init_cond_type,
        'pattern_idx': pattern_idx,
        'noise_level': noise_level,
        'use_symmetric_only': use_symmetric_only,
        'model_type': model_type,
        'use_numba': use_numba,
        'n_display': n_display,
        'seed': seed
    }
    # Save parameters to npy file
    np.save(os.path.join(npy_dir, "simulation_parameters.npy"), params)
    # Save parameters to txt file
    with open(os.path.join(output_dir, "parameters.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.close()

    # Save simulation data as .npy files
    print(f"\nSaving simulation data to '{npy_dir}'...")
    np.save(os.path.join(npy_dir, "connectivity_symmetric.npy"), W_S)
    np.save(os.path.join(npy_dir, "connectivity_asymmetric.npy"), W_A)
    np.save(os.path.join(npy_dir, "phi_memory_patterns.npy"), phi_eta)
    np.save(os.path.join(npy_dir, "memory_patterns.npy"), eta)

    for seed_index in range(334, 334+runs):  # Run multiple simulations
        # transform the number of the seed into a string called addition
        addition = str(seed_index)
        single_simulation(addition, W_S, W_A_sim, W, eta, phi_eta, t_span, ou_params, seed=seed_index, verbose=verbose)

if __name__ == "__main__":
    multiple_simulations()
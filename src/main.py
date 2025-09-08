"""
This file intends to simulate the dynamics of the NN with many different initial conditions
"""

import os

import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from torch import mul


from modules.connectivity import generate_connectivity_matrix, plot_matrix
from modules.dynamics import simulate_network, calculate_pattern_overlaps

# ================ Simulation Parameters ================ 

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
    verbose,
    single_dir_name,
    multiple_dir_name,

    # number of runs
    runs,
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
    np.random.seed(seed)  # for reproducibility
    # Create output directory and show error message if addition is not provided
    try:
        specific_output_dir = os.path.join(os.path.dirname(__file__), "..", multiple_dir_name, addition)
    except Exception as e:
        logger.error(f"Error occurred while setting output directory: {e}")
    os.makedirs(specific_output_dir, exist_ok=True)

    # Set up initial condition with proper noise calculation
    if init_cond_type == "Random":
        initial_condition = np.random.normal(0, 0.1, N)
    elif init_cond_type == "Zero":
        initial_condition = np.zeros(N)
    elif init_cond_type == "Memory Pattern":
        if p > 0:
            pattern_index = np.random.randint(0, p)
            pattern = eta[pattern_index]
            initial_condition = pattern.copy()
        else:
            initial_condition = np.random.normal(0, 0.1, N)
    else:  # Near Memory Pattern
        if p > 0:
            pattern_index = np.random.randint(0, p)
            pattern = eta[pattern_index % p]  # Getting the pattern based on index
            # Add noise scaled relative to pattern magnitude
            pattern_std = np.std(pattern)
            noise = np.random.normal(0, noise_level * pattern_std, N)
            initial_condition = pattern + noise
        else:
            initial_condition = np.random.normal(0, 0.1, N)

    # Run simulation with same φ parameters used in connectivity generation
    t, u, zeta = simulate_network(
        W_S=W_S,
        W_A=W_A_sim,
        t_span=t_span,
        dt=dt,
        tau=tau,
        activation_type=phi_function_type,
        activation_param=phi_beta
        if phi_function_type == "sigmoid" else phi_amplitude,
        initial_condition=initial_condition,
        use_ou=use_ou,
        ou_params=ou_params,
        r_m=phi_r_m,
        theta=0.0,
        x_r=phi_x_r,
        model_type=model_type,
        constant_zeta=constant_zeta if not use_ou else None,
        use_numba=use_numba)

    logger.info(f"\n {addition} - simulation completed successfully!")

    # Calculate pattern overlaps
    overlaps = None
    if p > 0:
        # Prepare phi function parameters
        phi_params = {
            'r_m': phi_r_m,
            'beta': phi_beta,
            'x_r': phi_x_r,
            'amplitude': phi_amplitude
        }

        # Prepare g function parameters
        g_params = {
            'r_m': g_r_m,
            'beta': g_beta,
            'x_r': g_x_r,
            'q': g_q,
            'x': g_x
        }

        overlaps = calculate_pattern_overlaps(u,
                                              eta,
                                              phi_function_type,
                                              phi_params,
                                              g_type,
                                              g_params,
                                              use_numba=use_numba,
                                              use_g=use_g)
        
    # Create data storage directory
    npy_dir = os.path.join(specific_output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)
    

    # # Save OU process
    # zeta_array = np.asarray(zeta)
    # if zeta_array.size == 1:
    #     # Save constant zeta as array
    #     zeta_full = np.full_like(t, float(zeta_array.item()))
    #     np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_full)
    # else:
    #     np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_array)
    
    # Calculate and save firing rates
    from modules.activation import sigmoid_function, relu_function
    if phi_function_type == "sigmoid":
        phi_u = sigmoid_function(u, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)
    else:  # relu
        phi_u = relu_function(u, amplitude=phi_amplitude)
    np.save(os.path.join(npy_dir, "firing_rates.npy"), phi_u)
    
    # # Save pattern overlaps if available
    # if p > 0 and overlaps is not None:
    #     np.save(os.path.join(npy_dir, "pattern_overlaps.npy"), overlaps)
    
    plt.figure()
    if p > 0 and overlaps is not None:
        for i in range(p):
            plt.plot(t, overlaps[:, i], label=f'Pattern {i+1}', linewidth=2)
        plt.xlabel('Time')
        plt.ylabel('Pattern Overlap')
        plt.title(f'Memory Pattern Overlaps (A={A_S}, seed = {seed})')
        plt.grid(True)
        plt.legend()
    else:
        plt.text(0.5, 0.5, 'No patterns to display', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Pattern Overlaps')
    plt.tight_layout()
    plt.savefig(os.path.join(specific_output_dir, "pattern_overlaps.png"), dpi=300)
    plt.close()


    """
    # Create complete 2x2 figure AND individual plots
    print(f"\nCreating dynamics figure and individual plots...")
    
    # Create 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Neural Currents (top-left)
    n_plot = min(n_display, N)
    ax = axs[0, 0]
    for i in range(n_plot):
        ax.plot(t, u[:, i], alpha=0.7, label=f'u_{i+1}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Current')
    ax.set_title(f'Neural Currents $u_i$ (first {n_plot} neurons)')
    ax.grid(True)
    if n_plot <= 5:
        ax.legend()

    # 2. Firing Rates (top-right)
    ax = axs[0, 1]
    for i in range(n_plot):
        ax.plot(t, phi_u[:, i], alpha=0.7, label=f'φ(u_{i+1})')
    ax.set_xlabel('Time')
    ax.set_ylabel('FR')
    ax.set_title(f'Firing Rates φ($u_i$) (first {n_plot} neurons)')
    ax.grid(True)
    if n_plot <= 5:
        ax.legend()

    # 3. Pattern Overlaps (bottom-left)
    ax = axs[1, 0]
    if p > 0 and overlaps is not None:
        for i in range(p):
            ax.plot(t, overlaps[:, i], label=f'Pattern {i+1}', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Pattern Overlap')
        ax.set_title(f'Memory Pattern Overlaps (A={A_S}, seed = {seed})')
        ax.grid(True)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No patterns to display', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Pattern Overlaps')

    # 4. OU Process (bottom-right)
    ax = axs[1, 1]
    try:
        if zeta_array.size == 1:
            # Constant zeta case
            zeta_val = float(zeta_array.item())
            ax.axhline(y=zeta_val, color='r', linestyle='-', linewidth=2)
            ax.set_ylim([zeta_val - 0.1, zeta_val + 0.1])
            ax.set_title(f'Control Signal ζ(t) = {zeta_val:.2f} (constant)')
        else:
            # Time-varying zeta case with OU parameters in legend
            if use_ou and ou_params is not None:
                # Include OU parameters in legend
                tau_zeta_val = ou_params.get('tau_zeta', tau_zeta)
                zeta_bar_val = ou_params.get('zeta_bar', zeta_bar)
                sigma_zeta_val = ou_params.get('sigma_zeta', sigma_zeta)
                legend_label = f'ζ(t) ($τ_ζ$={tau_zeta_val:.1f}, $\\bar{{ζ}}$={zeta_bar_val:.1f}, $σ_ζ$={sigma_zeta_val:.2f})'
                ax.set_title('Ornstein-Uhlenbeck Control Signal ζ(t)')
            else:
                legend_label = 'ζ(t)'
                ax.set_title('Control Signal ζ(t)')
            ax.plot(t, zeta_array, color='red', linewidth=1.5, label=legend_label)
            ax.legend()

        ax.set_xlabel('Time')
        ax.set_ylabel('ζ(t)')
        ax.grid(True)
    except Exception as e:
        print(f"Warning: Could not create ζ(t) plot - {e}")
        ax.text(0.5, 0.5, f'OU Process Error: {e}', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('OU Process')
    
    plt.tight_layout()
    
    # Save complete 2x2 figure
    complete_fig_path = os.path.join(specific_output_dir, "complete_simulation_results.png")
    plt.savefig(complete_fig_path, dpi=300)
    
    # Extract and save individual plots from the 2x2 figure
    titles = ['neural_currents', 'firing_rates', 'pattern_overlaps', 'ou_process']
    
    for i, ax in enumerate(axs.flat):
        fig_single, ax_single = plt.subplots(figsize=(10, 6))
        
        # Copy all lines from original subplot
        for line in ax.get_lines():
            ax_single.plot(line.get_xdata(), line.get_ydata(), 
                          label=line.get_label(), color=line.get_color(),
                          alpha=line.get_alpha() if line.get_alpha() is not None else 1.0, 
                          linewidth=line.get_linewidth())
        
        # Copy title, labels, and legend
        ax_single.set_title(ax.get_title())
        ax_single.set_xlabel(ax.get_xlabel())
        ax_single.set_ylabel(ax.get_ylabel())
        ax_single.grid(True)
        
        # Add legend if original had one
        if ax.get_legend() is not None:
            ax_single.legend()
        
        # Copy text annotations if any
        for text in ax.texts:
            ax_single.text(text.get_position()[0], text.get_position()[1], 
                          text.get_text(), ha=text.get_ha(), va=text.get_va(),
                          transform=ax_single.transAxes if text.get_transform() == ax.transAxes else ax_single.transData)
        
        # Copy axis limits for proper scaling
        ax_single.set_xlim(ax.get_xlim())
        ax_single.set_ylim(ax.get_ylim())
        
        plt.tight_layout()
        
        # Save individual plot
        fig_single.savefig(os.path.join(specific_output_dir, f"{titles[i]}.png"), dpi=300)
        plt.close(fig_single)  # Close to free memory
        """

    logger.info(f"Saved plots to {specific_output_dir}")

# =================================================================


def multiple_simulations():
    """
    Run multiple simulations with different initial conditions but same connectivity matrix.
    """

    # Set random seed for reproducibility
    np.random.seed(seed)

    # Create the main output directory
    output_dir_name = multiple_dir_name
    output_dir = os.path.join(os.path.dirname(__file__), "..", output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output directory created at: {output_dir}")


    # Generate connectivity matrix with all parameters
    logger.info(
        f"Generating connectivity matrices for {N} neurons with {p} patterns..."
    )
    W_S, W_A, W, eta, phi_eta = generate_connectivity_matrix(
        N=N,
        p=p,
        q=q,
        c=c,
        A_S=A_S,
        f_r_m=f_r_m,
        f_beta=f_beta,
        f_x_r=f_x_r,
        f_type=f_type,
        f_q=f_q,
        f_x=f_x,
        g_r_m=g_r_m,
        g_beta=g_beta,
        g_x_r=g_x_r,
        g_type=g_type,
        g_q=g_q,
        g_x=g_x,
        pattern_mean=pattern_mean,
        pattern_sigma=pattern_sigma,
        apply_sigma_cutoff=apply_sigma_cutoff,
        phi_function_type=phi_function_type,
        phi_amplitude=phi_amplitude,
        phi_beta=phi_beta,
        phi_r_m=phi_r_m,
        phi_x_r=phi_x_r,
        apply_phi_to_patterns=apply_phi_to_patterns,
        apply_er_to_asymmetric=apply_er_to_asymmetric,
        alpha=alpha,
        enforce_max_correlation=enforce_max_correlation,
        max_correlation=max_correlation)
    logger.info("Matrices generated successfully!")

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
    from modules.connectivity import calculate_pattern_correlation_matrix, plot_pattern_correlation_matrix
    correlation_matrix, actual_max_correlation = calculate_pattern_correlation_matrix(
        eta)


    if verbose:
        logger.info(f"\nMemory Pattern Correlation Analysis:")
        logger.info(f"Number of patterns: {eta.shape[0]}")
        logger.info(f"Correlation constraint enforced: {enforce_max_correlation}")
        if enforce_max_correlation:
            logger.info(f"Maximum correlation threshold: {max_correlation:.3f}")
        logger.info(f"Actual maximum correlation: {actual_max_correlation:.3f}")
    else:
        None

    if N < 2001: # Only plot matrices for smaller networks
        # Plot matrices
        _, _, _ = plot_pattern_correlation_matrix(eta,
                                                enforce_max_correlation,
                                                max_correlation,
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
        logger.info(f"Using constant ζ = {constant_zeta}") if verbose else None

    # Choose which connectivity matrix to use
    if use_symmetric_only:
        W_A_sim = np.zeros_like(W_A)
        logger.info("Using only the symmetric component (W^S) for dynamics.")
    else:
        W_A_sim = W_A
        logger.info("Using both symmetric and asymmetric components for dynamics.") if verbose else None

    logger.info(f"\nRunning {model_type} dynamics simulation...")
    logger.info(f"Time span: {t_start} to {t_end}, τ = {tau}")
    logger.info(
        f"φ function: {phi_function_type} (β={phi_beta}, r_m={phi_r_m}, x_r={phi_x_r})"
    ) 

    if verbose:
        if use_numba and N > 1000:
            logger.info(f"Numba optimization enabled for {N} neurons")
        elif not use_numba:
            logger.info("Numba optimization disabled")
    else:
        None
    
    # Saving common data in main npy dir
    npy_dir = os.path.join(output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    # Save simulation parameters
    params = {
        'N': N,
        'p': p,
        'q': q,
        'c': c,
        'A_S': A_S,
        'phi_function_type': phi_function_type,
        'phi_amplitude': phi_amplitude,
        'phi_beta': phi_beta,
        'phi_r_m': phi_r_m,
        'phi_x_r': phi_x_r,
        'f_type': f_type,
        'f_r_m': f_r_m,
        'f_beta': f_beta,
        'f_x_r': f_x_r,
        'f_q': f_q,
        'f_x': f_x,
        'g_type': g_type,
        'g_r_m': g_r_m,
        'g_beta': g_beta,
        'g_x_r': g_x_r,
        'g_q': g_q,
        'g_x': g_x,
        'pattern_mean': pattern_mean,
        'pattern_sigma': pattern_sigma,
        'enforce_max_correlation': enforce_max_correlation,
        'max_correlation': max_correlation,
        'alpha': alpha,
        'apply_sigma_cutoff': apply_sigma_cutoff,
        'apply_phi_to_patterns': apply_phi_to_patterns,
        'apply_er_to_asymmetric': apply_er_to_asymmetric,
        'use_ou': use_ou,
        'use_g': use_g,
        'tau_zeta': tau_zeta if use_ou else None,  # Only include if using OU
        'zeta_bar': zeta_bar if use_ou else None,  # Only include if using OU
        'sigma_zeta': sigma_zeta if use_ou else None,  # Only include if using OU
        'constant_zeta': constant_zeta if not use_ou else None,  # Only include if not using OU
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

    for seed_index in range(8, runs):  # Run multiple simulations
        # transform the number of the seed into a string called addition
        addition = str(seed_index)
        single_simulation(addition, W_S, W_A_sim, W, eta, phi_eta, t_span, ou_params, seed=seed_index, verbose=verbose)

if __name__ == "__main__":
    multiple_simulations()
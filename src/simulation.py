#!/usr/bin/env python3
"""
Neural network dynamics simulation for local execution
This script allows you to generate connectivity matrices and run simulations.

All parameters are configured in parameters.py - edit that file to change simulation settings.
"""
from math import log
import os
import numpy as np
import matplotlib.pyplot as plt
from sympy import use
from loguru import logger
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
    plot_connectivity_matrices,
    plot_heatmap,
    single_dir_name,
    verbose,

    # Seed for reproducibility
    seed
    )



def simulation():
    """
    Simulation of the network dynamics. 
    """
    np.random.seed(seed)  # for reproducibility
    output_dir = os.path.join(os.path.dirname(__file__), "..", single_dir_name)
    os.makedirs(output_dir, exist_ok=True) 
    logger.info(
        f"Generating connectivity matrices for {N} neurons with {p} patterns..."
    )

    # Generate connectivity matrix with all parameters
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

    # Display matrix statistics
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
        logger.info(f"Correlation matrix:")
        for i in range(correlation_matrix.shape[0]):
            row_str = " ".join([
                f"{correlation_matrix[i,j]:6.3f}"
                for j in range(correlation_matrix.shape[1])
            ])
            logger.info(f"  η{i+1}: [{row_str}]")

    if N < 2001 and plot_connectivity_matrices: # Only plot matrices for smaller networks
        logger.info("\nPlotting connectivity matrices...")
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
            "\nSkipping matrix plots for large network (N > 2000) or to save time and resources."
        )

    # Set up initial condition with proper noise calculation 
    if init_cond_type == "Random":
        initial_condition = np.random.normal(0, 0.1, N)
        logger.info("Initialized with random values.")
    elif init_cond_type == "Zero":
        initial_condition = np.zeros(N)
        logger.info("Initialized with zeros.")
    elif init_cond_type == "Memory Pattern":
        if p > 0:
            pattern = eta[pattern_idx % p]
            initial_condition = pattern.copy()
            logger.info(
                f"Initialized with memory pattern {(pattern_idx % p)+1} of {p}"
            )
        else:
            initial_condition = np.random.normal(0, 0.1, N)
            logger.info("No patterns available. Using random initialization.")
    else:  # Near Memory Pattern
        if p > 0:
            pattern = eta[pattern_idx % p] # Getting the pattern based on index
            # Add noise scaled relative to pattern magnitude
            pattern_std = np.std(pattern)
            noise = np.random.normal(0, noise_level * pattern_std, N)
            initial_condition = pattern + noise

            # Calculate similarity
            norm_pattern = pattern / np.linalg.norm(pattern)
            norm_initial = initial_condition / np.linalg.norm(
                initial_condition)
            similarity = np.dot(norm_pattern, norm_initial)
            logger.info(
                f"Initialized near memory pattern {(pattern_idx % p)+1} of {p}. Similarity: {similarity:.4f}"
            )
        else:
            initial_condition = np.random.normal(0, 0.1, N)
            logger.info("No patterns available. Using random initialization.")

    # Simulation time span
    t_span = (t_start, t_end)

    # Set up OU parameters
    if use_ou:
        ou_params = {
            'tau_zeta': tau_zeta,
            'zeta_bar': zeta_bar,
            'sigma_zeta': sigma_zeta
        }
        logger.info(
            f"Using Ornstein-Uhlenbeck process: τ_ζ={tau_zeta}, ζ̄={zeta_bar}, σ_ζ={sigma_zeta}"
        )
    else:
        ou_params = None
        logger.info(f"Using constant ζ = {constant_zeta}")

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

    if use_numba and N > 1000:
        logger.info(f"Numba optimization enabled for {N} neurons")
    elif not use_numba:
        logger.info("Numba optimization disabled")

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

    logger.info("\nSimulation completed successfully!")

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
        logger.info(f"\nFinal pattern overlaps:")
        for i in range(p):
            logger.info(f"  Pattern {i+1}: {overlaps[-1, i]:.4f}")
        logger.info(
            f" \nHighest final overlap: {np.max(overlaps[-1, :]):.4f} (Pattern {np.argmax(overlaps[-1, :]) + 1})"
        )
        
    # Create data storage directory
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
    # Save simulation data as .npy files
    print(f"\nSaving simulation data to '{npy_dir}'...")
    np.save(os.path.join(npy_dir, "time.npy"), t)
    np.save(os.path.join(npy_dir, "neural_currents.npy"), u)
    np.save(os.path.join(npy_dir, "connectivity_symmetric.npy"), W_S)
    np.save(os.path.join(npy_dir, "connectivity_asymmetric.npy"), W_A)
    np.save(os.path.join(npy_dir, "connectivity_total.npy"), W)
    np.save(os.path.join(npy_dir, "memory_patterns.npy"), eta)
    np.save(os.path.join(npy_dir, "phi_memory_patterns.npy"), phi_eta)
    
    # Save OU process
    zeta_array = np.asarray(zeta)
    if zeta_array.size == 1:
        # Save constant zeta as array
        zeta_full = np.full_like(t, float(zeta_array.item()))
        np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_full)
    else:
        np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_array)
    
    # Calculate and save firing rates
    from modules.activation import sigmoid_function, relu_function
    if phi_function_type == "sigmoid":
        phi_u = sigmoid_function(u, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)
    else:  # relu
        phi_u = relu_function(u, amplitude=phi_amplitude)
    np.save(os.path.join(npy_dir, "firing_rates.npy"), phi_u)
    
    # Save pattern overlaps if available
    if p > 0 and overlaps is not None:
        np.save(os.path.join(npy_dir, "pattern_overlaps.npy"), overlaps)
    
    logger.info(f"Saved: time, neural_currents, firing_rates, ou_process, connectivity matrices, memory_patterns")
    if p > 0 and overlaps is not None:
        logger.info(f"Saved: pattern_overlaps")
        if use_g:
            logger.info('Overlaps calculated using g function')
        else:
            logger.info('Overlaps calculated using default function')
    else:
        logger.info(f"No patterns available, skipping overlaps saving.")

    # Create plot for Gaussian distribution and activation functions
    fig3, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create x range for plotting
    x_range = np.linspace(-4, 4, 1000)

    # Plot N(μ,σ) distribution
    gaussian_pdf = (1 / np.sqrt(2 * np.pi * pattern_sigma**2)) * np.exp(
        -0.5 * ((x_range - pattern_mean) / pattern_sigma)**2)
    ax_twin = ax.twinx()
    ax_twin.plot(x_range,
                 gaussian_pdf,
                 'b-',
                 alpha=0.7,
                 linewidth=2,
                 label=f'N({pattern_mean},{pattern_sigma}²)')
    ax_twin.set_ylabel('PDF', color='blue')
    ax_twin.tick_params(axis='y', labelcolor='blue')

    # Plot φ (phi) activation function
    if phi_function_type == "sigmoid":
        phi_values = sigmoid_function(x_range,
                                      r_m=phi_r_m,
                                      beta=phi_beta,
                                      x_r=phi_x_r)
        ax.plot(x_range,
                phi_values,
                'r-',
                linewidth=2,
                label=f'φ sigmoid (β={phi_beta}, $x_r$={phi_x_r})')
        # Inflection point for sigmoid
        ax.axvline(x=phi_x_r,
                   color='red',
                   linestyle='--',
                   alpha=0.7,
                   linewidth=1)
        ax.text(phi_x_r,
                phi_r_m / 2,
                f'  $x_r$={phi_x_r}',
                rotation=90,
                verticalalignment='center')
    else:  # relu
        phi_values = relu_function(x_range, amplitude=phi_amplitude)
        ax.plot(x_range,
                phi_values,
                'r-',
                linewidth=2,
                label=f'φ ReLU (amp={phi_amplitude})')
        # Threshold at 0 for ReLU
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax.text(0,
                phi_amplitude / 2,
                '  threshold=0',
                rotation=90,
                verticalalignment='center')

    # Add 1-sigma cutoff line if enabled
    if apply_sigma_cutoff:
        cutoff_value = pattern_mean + pattern_sigma
        ax.axvline(x=cutoff_value,
                   color='green',
                   linestyle=':',
                   alpha=0.8,
                   linewidth=2)
        ax.text(cutoff_value,
                ax.get_ylim()[1] * 0.8,
                f'  1σ cutoff\n  ({cutoff_value:.1f})',
                rotation=90,
                verticalalignment='top',
                color='green')

    ax.set_xlabel('Input Value')
    ax.set_ylabel('φ(x)', color='red')
    ax.tick_params(axis='y', labelcolor='red')
    ax.set_title('Pattern Distribution N(μ,σ) & Activation Function φ(x)')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax_twin.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gaussian_activation_plot.png'), dpi=150)
    logger.info(
        "Saved Gaussian distribution and activation function plot to 'simulation_results/gaussian_activation_plot.png'"
    )

    # Create complete 2x2 figure AND individual plots
    logger.info(f"\nCreating dynamics figure and individual plots...")
    
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
    complete_fig_path = os.path.join(output_dir, "complete_simulation_results.png")
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
        fig_single.savefig(os.path.join(output_dir, f"{titles[i]}.png"), dpi=300)
        plt.close(fig_single)  # Close to free memory
        
        if plot_heatmap:
            # Plot the firing rates of ALL neurons as a heatmap in another figure
            plt.figure(figsize=(10, 6))
            plt.imshow(phi_u.T, aspect='auto', cmap='viridis', origin='lower')
            plt.colorbar(label='FR')
            plt.title(f'Firing Rates of all {N} neurons')
            plt.xlabel('Time')
            plt.ylabel('Neuron Index')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "firing_rates_heatmap.png"), dpi=300) # Save heatmap
            plt.close()
        else: None

    logger.info(f"Saved complete figure: complete_simulation_results.png")
    logger.info(f"Saved individual plots: {', '.join([f'{title}.png' for title in titles])}")
    if show_sim_plots:
        plt.show()
    else:
        logger.info("Plots not displayed, only saved to files.")
    plt.close('all')  # Close all figures to free memory

if __name__ == "__main__":
    simulation()
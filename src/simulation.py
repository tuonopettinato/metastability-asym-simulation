#!/usr/bin/env python3
"""
Neural network dynamics simulation for local execution
This script allows you to generate connectivity matrices and run simulations.

All parameters are configured in parameters.py - edit that file to change simulation settings.
"""
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from modules.connectivity import generate_connectivity_matrix, plot_matrix
from modules.dynamics import simulate_network, calculate_pattern_overlaps, initial_condition_creator
from modules.activation import sigmoid_function, relu_function
from modules.energy import compute_energy
from modules.connectivity import calculate_pattern_correlation_matrix, plot_pattern_correlation_matrix


def simulation():
    time_start = time.time()
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
    ou_non_neg,
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
    ou_threshold,

    # Seed for reproducibility
    seed
    )
    """
    Simulation of the network dynamics. 
    """
    np.random.seed(seed)  # for reproducibility
    output_dir = os.path.join(os.path.dirname(__file__), "..", single_dir_name)
    os.makedirs(output_dir, exist_ok=True) 
    logger.info(
        f"Generating connectivity matrices for {N} neurons with {p} patterns..."
    )

    #====== Generate connectivity matrix with all parameters =====================================
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
        apply_sigma_cutoff=apply_sigma_cutoff,
        phi_beta=phi_beta,
        phi_r_m=phi_r_m,
        phi_x_r=phi_x_r,
        apply_phi_to_patterns=apply_phi_to_patterns,
        apply_er_to_asymmetric=apply_er_to_asymmetric,
        alpha=alpha,
        enforce_max_correlation=enforce_max_correlation,
        max_correlation=max_correlation)
    logger.info("Matrices generated successfully!")

    # Convert to float32 =========================================================================
    W_S = W_S.astype(np.float32)
    W_A = W_A.astype(np.float32)
    W = W.astype(np.float32)
    eta = eta.astype(np.float32)
    phi_eta = phi_eta.astype(np.float32)
    # ============================================================================================

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

    # Calculate and show pattern correlation analysis
    
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

    # Prepare g function parameters
    g_params = {
        'q_f': g_q,
        'x_f': g_x
    }

    # Set up initial condition with proper noise calculation 
    initial_condition = initial_condition_creator(
        init_cond_type=init_cond_type,
        N=N,
        g_params=g_params,
        p=p,
        xi=phi_eta,
        pattern_idx=pattern_idx,
        noise_level=noise_level,
        seed=seed+19
    )

    # Simulation time span
    t_span = (t_start, t_end)

    # Convert time span to float32 =========================================
    t_span = (t_start, t_end)
    # ============================================================================================

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
        f"φ function: sigmoid (β={phi_beta}, r_m={phi_r_m}, x_r={phi_x_r})"
    )

    if use_numba and N > 1000:
        logger.info(f"Numba optimization enabled for {N} neurons")
    elif not use_numba:
        logger.info("Numba optimization disabled")

    # ===============================================
    # Convert all arrays and scalars to float32
    u = np.ascontiguousarray(initial_condition, dtype=np.float32)
    W_S = np.ascontiguousarray(W_S, dtype=np.float32)
    W_A_sim = np.ascontiguousarray(W_A_sim, dtype=np.float32)

    tau = np.float32(tau)
    dt = np.float32(dt)
    phi_r_m = np.float32(phi_r_m)
    phi_beta = np.float32(phi_beta)
    phi_x_r = np.float32(phi_x_r)
    zeta_bar = np.float32(zeta_bar)
    sigma_zeta = np.float32(sigma_zeta)
    tau_zeta = np.float32(tau_zeta)
    constant_zeta = np.float32(constant_zeta)

    # ===============================================

    # Create data storage directory
    npy_dir = os.path.join(output_dir, "npy")
    os.makedirs(npy_dir, exist_ok=True)

    # Saving connectivity matrices and patterns as .npy files

    np.save(os.path.join(npy_dir, "connectivity_symmetric.npy"), W_S)
    np.save(os.path.join(npy_dir, "connectivity_asymmetric.npy"), W_A)
    np.save(os.path.join(npy_dir, "connectivity_total.npy"), W)
    np.save(os.path.join(npy_dir, "memory_patterns.npy"), eta)
    np.save(os.path.join(npy_dir, "phi_memory_patterns.npy"), phi_eta)
    logger.info(f"Saved connectivity matrices and memory patterns to '{npy_dir}'")

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
        constant_zeta=constant_zeta if not use_ou else None,
        use_numba=use_numba,
        seed=seed,
        ou_non_neg=ou_non_neg)

    logger.info("\nSimulation completed successfully!")

    # Calculate pattern overlaps
    overlaps = None
    if p > 0:
        # Prepare phi function parameters
        phi_params = {
            'r_m': phi_r_m,
            'beta': phi_beta,
            'x_r': phi_x_r
        }

        overlaps = calculate_pattern_overlaps(u,
                                              eta,
                                              phi_params,
                                              g_params,
                                              use_numba=use_numba,
                                              use_g=use_g)
        logger.info(f"\nFinal pattern overlaps:")
        for i in range(p):
            logger.info(f"  Pattern {i+1}: {overlaps[-1, i]:.4f}")
        logger.info(
            f" \nHighest final overlap: {np.max(overlaps[-1, :]):.4f} (Pattern {np.argmax(overlaps[-1, :]) + 1})"
        )
        

    
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
    logger.info(f"\nSaving simulation data to '{npy_dir}'...")
    np.save(os.path.join(npy_dir, "time.npy"), t)
    
    # Save OU process
    zeta_array = np.asarray(zeta)
    if zeta_array.size == 1:
        # Save constant zeta as array
        zeta_full = np.full_like(t, float(zeta_array.item()))
        np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_full)
    else:
        np.save(os.path.join(npy_dir, "ou_process.npy"), zeta_array)
    
    # Calculate and save firing rates
    np.save(os.path.join(npy_dir, "neural_currents.npy"), u)
    phi_u = sigmoid_function(u, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)
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
    phi_values = sigmoid_function(x_range, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)

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

    figs = {}
    titles = ['neural_currents', 'firing_rates', 'ou_process', 'pattern_overlaps']

    # =====================================
    # 1. Neural Currents
    # =====================================
    logger.info("Plotting neural currents...")
    n_plot = min(n_display, u.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_plot):
        ax.plot(t, u[:, i], alpha=0.7, label=f'$u_{{{i+1}}}$')
    ax.set_xlabel('$t$', fontsize=20)
    ax.set_ylabel('$u_i$', fontsize=20)
    ax.grid(True)
    if n_plot <= 5:
        ax.legend(fontsize=16)
    figs[titles[0]] = (fig, ax)
    path = os.path.join(output_dir, f"{titles[0]}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {titles[0]} → {path}")

    # =====================================
    # 2. Firing Rates
    # =====================================
    logger.info("Plotting firing rates...")
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_plot):
        ax.plot(t, phi_u[:, i], alpha=0.7, label=f'$\\phi(u_{{{i+1}}})$')
    ax.set_xlabel('$t$', fontsize=20)
    ax.set_ylabel('FR', fontsize=20)
    ax.grid(True)
    if n_plot <= 5:
        ax.legend(fontsize=16)
    figs[titles[1]] = (fig, ax)
    path = os.path.join(output_dir, f"{titles[1]}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {titles[1]} → {path}")

    # =====================================
    # 3. OU Process
    # =====================================
    logger.info("Plotting OU process...")
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        if zeta_array.size == 1:
            zeta_val = float(zeta_array.item())
            ax.axhline(y=zeta_val, color='r', linestyle='-', linewidth=2)
            ax.set_ylim([zeta_val - 0.1, zeta_val + 0.1])
            legend_label = f'$\\zeta(t)={zeta_val:.2f}$ (constant)'
        else:
            if use_ou and ou_params is not None:
                tau_zeta_val = ou_params.get('tau_zeta', tau_zeta)
                zeta_bar_val = ou_params.get('zeta_bar', zeta_bar)
                sigma_zeta_val = ou_params.get('sigma_zeta', sigma_zeta)
                legend_label = f'$τ_ζ$={tau_zeta_val:.2f}, $\\bar{{ζ}}$={zeta_bar_val:.2f}, $σ_ζ$={sigma_zeta_val:.2f}'
            else:
                legend_label = '$\\zeta(t)$'
            ax.plot(t, zeta_array, color='k', linewidth=1.5, label=legend_label)
            ax.legend(fontsize=16)
        ax.set_xlabel('$t$', fontsize=20)
        ax.set_ylabel('$\\zeta(t)$', fontsize=20)
        ax.grid(True)
        figs[titles[2]] = (fig, ax)
        path = os.path.join(output_dir, f"{titles[2]}.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved {titles[2]} → {path}")
    except Exception as e:
        logger.error(f"OU plot error: {e}")

    # =====================================
    # 4. Pattern Overlaps
    # =====================================
    logger.info("Plotting pattern overlaps...")
    fig, ax = plt.subplots(figsize=(10, 6))
    # plot the times when the ou noise is above the threshold
    if zeta_array is not None:
        above_threshold = zeta_array > ou_threshold
        ax.fill_between(t, -0.1, 1.1, where=above_threshold, color='lightgray', alpha=0.7, ls = 'dashed', transform=ax.get_xaxis_transform())
    if overlaps is not None and overlaps.shape[1] > 0:
        p = overlaps.shape[1]
        for i in range(p):
            ax.plot(t, overlaps[:, i], label=f'P {i+1}', linewidth=2)
        ax.axhline(y=0.8, color='gray', alpha=0.5)
        ax.set_xlabel('$t$', fontsize=20)
        ax.set_ylabel('Overlaps', fontsize=20)
        ax.grid(False)
        ax.legend(fontsize=16)
    else:
        ax.text(0.5, 0.5, 'No patterns to display', ha='center', va='center', transform=ax.transAxes)
    figs[titles[3]] = (fig, ax)
    path = os.path.join(output_dir, f"{titles[3]}.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved {titles[3]} → {path}")

    # =====================================
    # Optional Heatmap
    # =====================================
    if plot_heatmap == True:
        logger.info("Plotting firing rate heatmap...")
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(phi_u.T, aspect='auto', cmap='viridis', origin='lower')
        plt.colorbar(im, ax=ax, label='FR')
        ax.set_xlabel('Time', fontsize=18)
        ax.set_ylabel('Neuron Index', fontsize=18)
        path = os.path.join(output_dir, "firing_rates_heatmap.png")
        fig.savefig(path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved firing_rates_heatmap → {path}")

    # =====================================
    # 5. Build 2x2 composite figure
    # =====================================
    logger.info("Building 2×2 composite summary figure...")
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    for idx, key in enumerate(titles):
        r, c = divmod(idx, 2)
        src_fig, src_ax = figs[key]
        ax = axs[r, c]
        for line in src_ax.get_lines():
            ax.plot(line.get_xdata(), line.get_ydata(),
                    label=line.get_label(), color=line.get_color(),
                    linewidth=line.get_linewidth(), alpha=line.get_alpha() or 1.0)
        ax.set_xlabel(src_ax.get_xlabel())
        ax.set_ylabel(src_ax.get_ylabel())
        if src_ax.get_legend() is not None:
            ax.legend(fontsize=12)
        ax.grid(True)
    plt.tight_layout()
    complete_fig_path = os.path.join(output_dir, "complete_simulation_results.png")
    fig.savefig(complete_fig_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved composite figure → {complete_fig_path}")

    plt.close('all')  # making sure to close all figures to free memory
    time_end = time.time()
    elapsed_time = time_end - time_start
    logger.info(f"\nTotal execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    simulation()
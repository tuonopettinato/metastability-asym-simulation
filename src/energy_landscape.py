import os
import numpy as np
from matplotlib import pyplot as plt
from modules.energy import plot_energy_from_npy

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
    multiple_sim_dir = os.path.join(os.path.dirname(__file__), "..", f'{"multiple_simulations_"}{N}', "npy")
    W_symm = np.load(os.path.join(multiple_sim_dir, "connectivity_symmetric.npy"))
    W_asymm = np.load(os.path.join(multiple_sim_dir, "connectivity_asymmetric.npy"))
    files_dir = os.path.join(os.path.join(multiple_sim_dir, "firing_rates"))
    files = [os.path.join(files_dir, f) 
                for f in os.listdir(files_dir) if f.endswith('.npy')]
    fig = plot_energy_from_npy(files, W_symm, W_asymm, tau, phi_beta=phi_beta, phi_r_m=phi_r_m, phi_x_r=phi_x_r)
    fig.savefig(os.path.join(multiple_sim_dir, "landscape.png"))
if __name__ == "__main__":
    main()
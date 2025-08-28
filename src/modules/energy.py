""" energy.py
This module computes the energy trajectories of a neural network simulation.
It loads the connectivity matrix and firing rates from .npy files and calculates the energy
based on the connectivity and firing rates. The energy is computed as the quadratic form
E = h^T W h, where h is the firing rate vector and W is the connectivity matrix.
The results are saved in the 'loaded_results' directory.
It also computes the energy trajectories for both the original symmetric and asymmetric components
and the new symmetric and antisymmetric components derived from the total connectivity matrix.
"""

import re
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from modules.activation import inverse_sigmoid_function, inverse_relu_function


def compute_energy(
    W, h,
    phi_function_type="sigmoid",
    phi_amplitude=1.0,
    phi_beta=1.5,
    phi_r_m=30.0,
    phi_x_r=2.0,
    num_interp_points=1000,
    activation_term=True
):
    """
    Compute total energy E(h) = -0.5 * h^T W h + sum_i ∫₀^{h_i} phi⁻¹(x) dx
    using lookup-table interpolation for fast integration.

    Args:
        W: (N x N) connectivity matrix
        h: (M x N) array of firing rates (M states), or (N,) single state
        phi_function_type: "sigmoid" or "relu"
        phi_amplitude, phi_beta, phi_r_m, phi_x_r: activation inverse parameters
        num_interp_points: resolution of lookup table
        activation_term: whether to include activation term in energy computation

    Returns:
        energies: (M,) energy values for each state, or scalar if h was 1D
    """
    # Normalize shape: (M, N)
    if h.ndim == 1:
        h = h.reshape(1, -1)
        single_state = True
    else:
        single_state = False
    M, N = h.shape

    # Select inverse function and clipping range
    if phi_function_type == "sigmoid":
        inv_func = lambda v: inverse_sigmoid_function(v, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)
        h_clip_min = 1e-4
        h_clip_max = phi_r_m - 1e-4
    elif phi_function_type == "relu":
        inv_func = lambda v: inverse_relu_function(v, amplitude=phi_amplitude)
        h_clip_min = 0.0
        h_clip_max = phi_amplitude * 1.5
    else:
        raise ValueError("Unsupported phi_function_type")

    # Precompute lookup table
    x_vals = np.linspace(h_clip_min, h_clip_max, num_interp_points)
    integral_vals = np.array([quad(inv_func, 0, x)[0] for x in x_vals])
    interp_integral = interp1d(
        x_vals, integral_vals,
        kind="linear",
        bounds_error=False,
        fill_value=(integral_vals[0], integral_vals[-1])  # Avoid extrapolation
    )

    # Compute energy for each state
    energies = np.zeros(M)
    syn_terms = np.zeros(M)
    act_terms = np.zeros(M)
    for t in range(M):
        h_t = h[t, :]
        h_t_clipped = np.clip(h_t, h_clip_min, h_clip_max)

        # Synaptic term
        energy_syn = -0.5 * h_t @ W @ h_t
        syn_terms[t] = energy_syn

        # Activation term via lookup-table integration
        energy_act = np.sum(interp_integral(h_t_clipped))
        act_terms[t] = energy_act

        energies[t] = energy_syn + energy_act if activation_term else energy_syn

    if single_state:
        return energies[0], syn_terms[0], act_terms[0]
    else:
        return energies, syn_terms, act_terms



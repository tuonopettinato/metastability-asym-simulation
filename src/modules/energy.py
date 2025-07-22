""" energy.py
This module computes the energy trajectories of a neural network simulation.
It loads the connectivity matrix and firing rates from .npy files and calculates the energy
based on the connectivity and firing rates. The energy is computed as the quadratic form
E = h^T W h, where h is the firing rate vector and W is the connectivity matrix.
The results are saved in the 'loaded_results' directory.
It also computes the energy trajectories for both the original symmetric and asymmetric components
and the new symmetric and antisymmetric components derived from the total connectivity matrix.
"""

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
    num_interp_points=1000
):
    """
    Compute total energy E(t) = -0.5 * h_t^T W h_t + sum_i int_0^{h_i} g^{-1}(x) dx
    using lookup-table interpolation for fast integration.
    It can handle both the single state case (h is a vector) and the multi-state case (h is a matrix).
    
    Args:
        W: (N x N) connectivity matrix
        h: (T x N) array of firing rates OR memory pattern (the single state case)
        phi_function_type: "sigmoid" or "relu"
        phi_amplitude, phi_beta, phi_r_m, phi_x_r: parameters of inverse activation
        num_interp_points: number of points for lookup table grid
        
    Returns:
        E: (T,) energy at each time step if h is a matrix, or a single value if h is a vector.
    """
    # Ensure h shape is (T, N)
    if h.ndim == 1:
        h.reshape(-1, 1)  # If h is 1D, convert to 2D column vector
        single_state = True
        T, N = 1, h.shape[0]
    else:
        single_state = False
        T, N = h.shape

    # Select inverse function
    if phi_function_type == "sigmoid":
        inv_func = lambda v: inverse_sigmoid_function(v, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)
        h_clip_min = 1e-4
        h_clip_max = phi_r_m - 1e-4
    elif phi_function_type == "relu":
        inv_func = lambda v: inverse_relu_function(v, amplitude=phi_amplitude)
        h_clip_min = 0.0
        h_clip_max = phi_amplitude * 1.5  # estensione sicura
    else:
        raise ValueError("Unsupported phi_function_type")

    # Precompute lookup table
    x_vals = np.linspace(h_clip_min, h_clip_max, num_interp_points)
    integral_vals = np.array([quad(inv_func, 0, x)[0] for x in x_vals])
    interp_integral = interp1d(x_vals, integral_vals, kind="linear", bounds_error=False, fill_value="extrapolate")

    # Compute energy
    energies = np.zeros(T)
    for t in range(T):
        h_t = h[t, :]
        h_t_clipped = np.clip(h_t, h_clip_min, h_clip_max)

        # First term: -0.5 * h_t^T W h_t
        energy_syn = -0.5 * h_t @ W @ h_t

        # Second term: sum_i ∫0^{h_i} g^{-1}(x) dx via interpolation
        energy_act =  np.sum(interp_integral(h_t_clipped))
        energies[t] = energy_syn + energy_act

    return energies if not single_state else energies[0]


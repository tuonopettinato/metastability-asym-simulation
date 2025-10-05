""" energy.py
This module computes the energy trajectories of a neural network simulation.
It loads the connectivity matrix and firing rates from .npy files and calculates the energy
based on the connectivity and firing rates. The energy is computed as the quadratic form
E = h^T W h, where h is the firing rate vector and W is the connectivity matrix.
The results are saved in the 'loaded_results' directory.
It also computes the energy trajectories for both the original symmetric and asymmetric components
and the new symmetric and antisymmetric components derived from the total connectivity matrix.
"""
import os
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from modules.activation import inverse_sigmoid_function, sigmoid_function, derivative_sigmoid_function


def normalize_shape(h):
    # Normalize shape: (M, N)
    if h.ndim == 1:
        h = h.reshape(1, -1)
        single_state = True
    else:
        single_state = False
    M, N = h.shape
    return h, single_state, M, N

def compute_energy(
    W, h,
    tau=1.0,
    phi_beta=1.5,
    phi_r_m=30.0,
    phi_x_r=2.0,
    num_interp_points=1000,
    activation_term=True
):
    """
    h = phi(u) is the firing rate vector
    Compute total energy E(u) = -0.5 * phi(u)^T W phi(u) + sum_i ∫₀^{phi(u_i)} phi⁻¹(x) dx 
    using lookup-table interpolation for fast integration.
    

    Args:
        W: (N x N) connectivity matrix
        h: (M x N) array of firing rates (M states), or (N,) single state
        phi_beta, phi_r_m, phi_x_r: activation inverse parameters
        num_interp_points: resolution of lookup table
        activation_term: whether to include activation term in energy computation

    Returns:
        energies: (M,) energy values for each state, or scalar if h was 1D
    """
    h, single_state, M, N = normalize_shape(h)

    # Select inverse function and clipping range
    inv_func = lambda v: inverse_sigmoid_function(v, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)
    h_clip_min = 1e-4
    h_clip_max = phi_r_m - 1e-4

    # Precompute lookup table
    x_vals = np.linspace(h_clip_min, h_clip_max, num_interp_points)
    integral_vals = np.array([quad(lambda x: inv_func(x), 0, x)[0] for x in x_vals])
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
        energy_act = np.sum(interp_integral(h_t_clipped))  if activation_term else 0.0
        act_terms[t] = energy_act

        energies[t] = energy_syn + energy_act # Total energy

    if single_state:
        return energies[0], syn_terms[0], act_terms[0]
    else:
        return energies, syn_terms, act_terms
    

def compute_forces(W_symm, W_asymm, h, tau=1.0, phi_beta=1.5, phi_r_m=30.0, phi_x_r=2.0):
    """
    Compute the symmetric and asymmetric forces for all neurons:
    F_symm_i(u) = phi'(u_i) * [sum_j W^S_ij * phi_j(u_j) - u_i]
    F_asymm_i(u) = sum_j W^A_ij * phi_j(u_j)
    
    Args:
        W: (N x N) connectivity matrix
        h: (M x N) array of firing rates (M states), or (N,) single state
        tau: time constant
        phi_beta, phi_r_m, phi_x_r: activation parameters
    Returns:
        F_symm: (M x N) array of symmetric forces for each state, or (N,) single state
        F_asymm: (M x N) array of asymmetric forces for each state, or (N,) single state
    """
    h, single_state, M, N = normalize_shape(h)
    u = inverse_sigmoid_function(h, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)

    # Compute activation function values
    phi_deriv_u = derivative_sigmoid_function(u, r_m=phi_r_m, beta=phi_beta, x_r=phi_x_r)

    # Compute forces
    F_symm =  phi_deriv_u * ((W_symm @ h.T).T - h) # Shape: (M, N)
    F_asymm = phi_deriv_u * (W_asymm @ h.T).T         # Shape: (M, N)

    # Compute the average value of the forces (integrate over the path) F_avg = ∫ F(h) dh / ∫ dh
    dh = np.linalg.norm(np.diff(h, axis=0), axis=1)  # Shape: (M-1,)
    dh = np.append(dh, dh[-1])  # Assume last step same as second last for simplicity
    F_symm_avg = np.sum(F_symm * dh[:, np.newaxis], axis=0) / np.sum(dh)
    F_asymm_avg = np.sum(F_asymm * dh[:, np.newaxis], axis=0) / np.sum(dh)

    if single_state:
        return F_symm[0], F_asymm[0]
    else:
        return F_symm, F_asymm, F_symm_avg, F_asymm_avg

def plot_energy_from_npy(files, W_symm, W_asymm, tau, neuron1=0, neuron2=1, phi_beta=1.5, phi_r_m=30.0, phi_x_r=2.0):
    """
    Load firing rate data from multiple .npy files, compute energy, and plot 3D scatter
    of energy vs two selected neurons.

    Args:
        files: list of str, paths to .npy files (shape: N x T)
        W: connectivity matrix for energy computation
        neuron1, neuron2: indices of neurons to plot
        phi_beta, phi_r_m, phi_x_r: parameters for compute_energy
    """
    energies_list = []
    f_symm_list1 = []
    f_asymm_list1 = []
    f_symm_list2 = []
    f_asymm_list2 = []
    rates_n1 = []
    rates_n2 = []

    for fpath in files:
        h = np.load(fpath)  # shape: (N, T) or (T, N)? adjust if needed
        # Ensure h shape is (states, neurons)
        if h.ndim == 1:
            h = h.reshape(1, -1)
        elif h.shape[0] == W_symm.shape[0]:
            h = h.T  # transpose (neurons x time) → (time x neurons)
        elif h.shape[1] != W_symm.shape[0]:
            raise ValueError(f"Incompatible shape: {h.shape} vs W {W_symm.shape}")
        
        # Compute energy for each time point/state
        E, _, _ = compute_energy(W_symm, h, phi_beta=phi_beta, phi_r_m=phi_r_m, phi_x_r=phi_x_r)
        # Compute forces and plot the vectors (project on the plane of neuron1 and neuron2)
        F_symm, F_asymm, _, _ = compute_forces(W_symm, W_asymm, h, tau= tau, phi_beta=phi_beta, phi_r_m=phi_r_m, phi_x_r=phi_x_r)
        # Store energies and selected neuron rates
        
        energies_list.extend(E)
        # Force component for neuron1 and neuron2
        f_symm_list1.extend(F_symm[:, neuron1])  # Force component for neuron1 
        f_asymm_list1.extend(F_asymm[:, neuron1])  # Force component for neuron1 
        f_symm_list2.extend(F_symm[:, neuron2])  # Force component for neuron2
        f_asymm_list2.extend(F_asymm[:, neuron2])  # Force component for neuron2
        rates_n1.extend(h[:, neuron1])
        rates_n2.extend(h[:, neuron2])

    energies_list = np.array(energies_list)
    f_symm_list1 = np.array(f_symm_list1)
    f_asymm_list1 = np.array(f_asymm_list1)
    f_symm_list2 = np.array(f_symm_list2)
    f_asymm_list2 = np.array(f_asymm_list2)
    rates_n1 = np.array(rates_n1)
    rates_n2 = np.array(rates_n2)

    # 3D scatter plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(rates_n1, rates_n2, energies_list, c=energies_list, cmap='viridis', s=40)
    # plot the vectors of the forces
    ax.quiver(rates_n1, rates_n2, energies_list, f_symm_list1, f_symm_list2, np.zeros_like(energies_list), color='blue', length=0.5, normalize=True, label='Symmetric Force')
    ax.quiver(rates_n1, rates_n2, energies_list, f_asymm_list1, f_asymm_list2, np.zeros_like(energies_list), color='red', length=0.5, normalize=True, label='Asymmetric Force')

    ax.set_xlabel(f'Neuron {neuron1+1} firing rate')
    ax.set_ylabel(f'Neuron {neuron2+1} firing rate')
    ax.set_zlabel('Energy')
    ax.set_title('Energy vs Neuron Firing Rates')
    fig.colorbar(sc, label='Energy')
    return fig

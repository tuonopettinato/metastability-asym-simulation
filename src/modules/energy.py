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

def compute_energy(W, h):
    """
    Compute the total energy of the system given the connectivity matrix W and history h.
    """
    # Ensure h is a column vector
    if h.ndim == 1:
        h = h[:, np.newaxis]
    # Total energy
    E_mat = h @ W @ h.T
    # Sum over the 0 axis to get a scalar
    E_traj = - np.sum(E_mat, axis=0)
    return E_traj
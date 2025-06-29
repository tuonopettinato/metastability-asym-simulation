import os
import numpy as np
import matplotlib.pyplot as plt

def load_data(connectivity_path, history_path):
    W = np.load(connectivity_path)
    h = np.load(history_path)
    return W, h

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
    E_traj = -np.sum(E_mat, axis=0)
    return E_traj

if __name__ == "__main__":
    # Use paths relative to this script's location
    base_dir = os.path.dirname(__file__)
    loaded_results_dir = os.path.join(base_dir, "..", "loaded_results")
    os.makedirs(loaded_results_dir, exist_ok=True)
    npy_dir = os.path.join(base_dir, "..", "simulation_results", "npy")
    connectivity_path = os.path.join(npy_dir, "connectivity_total.npy")
    original_symm_path = os.path.join(npy_dir, "connectivity_symmetric.npy")
    original_asymm_path = os.path.join(npy_dir, "connectivity_asymmetric.npy")
    history_path = os.path.join(npy_dir, "neural_currents.npy")
    W, h = load_data(connectivity_path, history_path)
    W_symm = np.load(original_symm_path)
    W_asymm = np.load(original_asymm_path)
    # plot energy trajectories both original and divided with symmetric and antisymmetric components
    E_total_traj = compute_energy(W, h)
    E_symm_traj = compute_energy(W_symm, h)
    E_asymm_traj = compute_energy(W_asymm, h)
    t = np.arange(E_total_traj.shape[0])  # Assuming time steps are
    plt.figure(figsize=(12, 6))
    plt.plot(t, E_total_traj, label='Total Energy', color='blue')
    plt.plot(t, E_symm_traj, label='Symmetric Energy', color='green')
    plt.plot(t, E_asymm_traj, label='Asymmetric Energy', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    plt.title('Energy Trajectories (Original symmetric and asymmetric components)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(loaded_results_dir, "energy_trajectories_original.png"))
    plt.show()
    plt.figure(figsize=(12, 6))
    W_symm_new = 0.5 * (W + W.T)
    W_antisym_new = 0.5 * (W - W.T)
    E_symm_new_traj, E_asymm_new_traj = compute_energy(W_symm_new, h), compute_energy(W_antisym_new, h)
    plt.plot(t, E_total_traj, label='Total Energy', color='blue')
    plt.plot(t, E_symm_new_traj, label='Symmetric Energy (New)', color='green')
    plt.plot(t, E_asymm_new_traj, label='Antisymmetric Energy (New)', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Energy')
    plt.title('Energy Trajectories (New symmetric and antisymmetric components)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(loaded_results_dir, "energy_trajectories_new.png"))
    plt.show()